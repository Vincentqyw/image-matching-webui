import torch
import torch.nn.functional as F

try:
    import cupy
except:
    print("Cupy not found, local correlation will not work")
import re
from ..dkm import ConvRefiner


class Stream:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        stream = torch.cuda.current_stream(device=device).cuda_stream
    else:
        stream = None


kernel_Correlation_rearrange = """
	extern "C" __global__ void kernel_Correlation_rearrange(
		const int n,
		const float* input,
		float* output
	) {
	  int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	  if (intIndex >= n) {
	    return;
	  }
	  int intSample = blockIdx.z;
	  int intChannel = blockIdx.y;
	  float dblValue = input[(((intSample * SIZE_1(input)) + intChannel) * SIZE_2(input) * SIZE_3(input)) + intIndex];
	  __syncthreads();
	  int intPaddedY = (intIndex / SIZE_3(input)) + 4;
	  int intPaddedX = (intIndex % SIZE_3(input)) + 4;
	  int intRearrange = ((SIZE_3(input) + 8) * intPaddedY) + intPaddedX;
	  output[(((intSample * SIZE_1(output) * SIZE_2(output)) + intRearrange) * SIZE_1(input)) + intChannel] = dblValue;
	}
"""

kernel_Correlation_updateOutput = """
	extern "C" __global__ void kernel_Correlation_updateOutput(
	  const int n,
	  const float* rbot0,
	  const float* rbot1,
	  float* top
	) {
	  extern __shared__ char patch_data_char[];
	  float *patch_data = (float *)patch_data_char;
	  // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
	  int x1 = blockIdx.x + 4;
	  int y1 = blockIdx.y + 4;
	  int item = blockIdx.z;
	  int ch_off = threadIdx.x;
	  // Load 3D patch into shared shared memory
	  for (int j = 0; j < 1; j++) { // HEIGHT
	    for (int i = 0; i < 1; i++) { // WIDTH
	      int ji_off = (j + i) * SIZE_3(rbot0);
	      for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
	        int idx1 = ((item * SIZE_1(rbot0) + y1+j) * SIZE_2(rbot0) + x1+i) * SIZE_3(rbot0) + ch;
	        int idxPatchData = ji_off + ch;
	        patch_data[idxPatchData] = rbot0[idx1];
	      }
	    }
	  }
	  __syncthreads();
	  __shared__ float sum[32];
	  // Compute correlation
	  for (int top_channel = 0; top_channel < SIZE_1(top); top_channel++) {
	    sum[ch_off] = 0;
	    int s2o = top_channel % 9 - 4;
	    int s2p = top_channel / 9 - 4;
	    for (int j = 0; j < 1; j++) { // HEIGHT
	      for (int i = 0; i < 1; i++) { // WIDTH
	        int ji_off = (j + i) * SIZE_3(rbot0);
	        for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
	          int x2 = x1 + s2o;
	          int y2 = y1 + s2p;
	          int idxPatchData = ji_off + ch;
	          int idx2 = ((item * SIZE_1(rbot0) + y2+j) * SIZE_2(rbot0) + x2+i) * SIZE_3(rbot0) + ch;
	          sum[ch_off] += patch_data[idxPatchData] * rbot1[idx2];
	        }
	      }
	    }
	    __syncthreads();
	    if (ch_off == 0) {
	      float total_sum = 0;
	      for (int idx = 0; idx < 32; idx++) {
	        total_sum += sum[idx];
	      }
	      const int sumelems = SIZE_3(rbot0);
	      const int index = ((top_channel*SIZE_2(top) + blockIdx.y)*SIZE_3(top))+blockIdx.x;
	      top[index + item*SIZE_1(top)*SIZE_2(top)*SIZE_3(top)] = total_sum / (float)sumelems;
	    }
	  }
	}
"""

kernel_Correlation_updateGradFirst = """
	#define ROUND_OFF 50000
	extern "C" __global__ void kernel_Correlation_updateGradFirst(
	  const int n,
	  const int intSample,
	  const float* rbot0,
	  const float* rbot1,
	  const float* gradOutput,
	  float* gradFirst,
	  float* gradSecond
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
	  int n = intIndex % SIZE_1(gradFirst); // channels
	  int l = (intIndex / SIZE_1(gradFirst)) % SIZE_3(gradFirst) + 4; // w-pos
	  int m = (intIndex / SIZE_1(gradFirst) / SIZE_3(gradFirst)) % SIZE_2(gradFirst) + 4; // h-pos
	  // round_off is a trick to enable integer division with ceil, even for negative numbers
	  // We use a large offset, for the inner part not to become negative.
	  const int round_off = ROUND_OFF;
	  const int round_off_s1 = round_off;
	  // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
	  int xmin = (l - 4 + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4)
	  int ymin = (m - 4 + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4)
	  // Same here:
	  int xmax = (l - 4 + round_off_s1) - round_off; // floor (l - 4)
	  int ymax = (m - 4 + round_off_s1) - round_off; // floor (m - 4)
	  float sum = 0;
	  if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(gradOutput)-1) && (ymin<=SIZE_2(gradOutput)-1)) {
	    xmin = max(0,xmin);
	    xmax = min(SIZE_3(gradOutput)-1,xmax);
	    ymin = max(0,ymin);
	    ymax = min(SIZE_2(gradOutput)-1,ymax);
	    for (int p = -4; p <= 4; p++) {
	      for (int o = -4; o <= 4; o++) {
	        // Get rbot1 data:
	        int s2o = o;
	        int s2p = p;
	        int idxbot1 = ((intSample * SIZE_1(rbot0) + (m+s2p)) * SIZE_2(rbot0) + (l+s2o)) * SIZE_3(rbot0) + n;
	        float bot1tmp = rbot1[idxbot1]; // rbot1[l+s2o,m+s2p,n]
	        // Index offset for gradOutput in following loops:
	        int op = (p+4) * 9 + (o+4); // index[o,p]
	        int idxopoffset = (intSample * SIZE_1(gradOutput) + op);
	        for (int y = ymin; y <= ymax; y++) {
	          for (int x = xmin; x <= xmax; x++) {
	            int idxgradOutput = (idxopoffset * SIZE_2(gradOutput) + y) * SIZE_3(gradOutput) + x; // gradOutput[x,y,o,p]
	            sum += gradOutput[idxgradOutput] * bot1tmp;
	          }
	        }
	      }
	    }
	  }
	  const int sumelems = SIZE_1(gradFirst);
	  const int bot0index = ((n * SIZE_2(gradFirst)) + (m-4)) * SIZE_3(gradFirst) + (l-4);
	  gradFirst[bot0index + intSample*SIZE_1(gradFirst)*SIZE_2(gradFirst)*SIZE_3(gradFirst)] = sum / (float)sumelems;
	} }
"""

kernel_Correlation_updateGradSecond = """
	#define ROUND_OFF 50000
	extern "C" __global__ void kernel_Correlation_updateGradSecond(
	  const int n,
	  const int intSample,
	  const float* rbot0,
	  const float* rbot1,
	  const float* gradOutput,
	  float* gradFirst,
	  float* gradSecond
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
	  int n = intIndex % SIZE_1(gradSecond); // channels
	  int l = (intIndex / SIZE_1(gradSecond)) % SIZE_3(gradSecond) + 4; // w-pos
	  int m = (intIndex / SIZE_1(gradSecond) / SIZE_3(gradSecond)) % SIZE_2(gradSecond) + 4; // h-pos
	  // round_off is a trick to enable integer division with ceil, even for negative numbers
	  // We use a large offset, for the inner part not to become negative.
	  const int round_off = ROUND_OFF;
	  const int round_off_s1 = round_off;
	  float sum = 0;
	  for (int p = -4; p <= 4; p++) {
	    for (int o = -4; o <= 4; o++) {
	      int s2o = o;
	      int s2p = p;
	      //Get X,Y ranges and clamp
	      // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
	      int xmin = (l - 4 - s2o + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4 - s2o)
	      int ymin = (m - 4 - s2p + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4 - s2o)
	      // Same here:
	      int xmax = (l - 4 - s2o + round_off_s1) - round_off; // floor (l - 4 - s2o)
	      int ymax = (m - 4 - s2p + round_off_s1) - round_off; // floor (m - 4 - s2p)
	      if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(gradOutput)-1) && (ymin<=SIZE_2(gradOutput)-1)) {
	        xmin = max(0,xmin);
	        xmax = min(SIZE_3(gradOutput)-1,xmax);
	        ymin = max(0,ymin);
	        ymax = min(SIZE_2(gradOutput)-1,ymax);
	        // Get rbot0 data:
	        int idxbot0 = ((intSample * SIZE_1(rbot0) + (m-s2p)) * SIZE_2(rbot0) + (l-s2o)) * SIZE_3(rbot0) + n;
	        float bot0tmp = rbot0[idxbot0]; // rbot1[l+s2o,m+s2p,n]
	        // Index offset for gradOutput in following loops:
	        int op = (p+4) * 9 + (o+4); // index[o,p]
	        int idxopoffset = (intSample * SIZE_1(gradOutput) + op);
	        for (int y = ymin; y <= ymax; y++) {
	          for (int x = xmin; x <= xmax; x++) {
	            int idxgradOutput = (idxopoffset * SIZE_2(gradOutput) + y) * SIZE_3(gradOutput) + x; // gradOutput[x,y,o,p]
	            sum += gradOutput[idxgradOutput] * bot0tmp;
	          }
	        }
	      }
	    }
	  }
	  const int sumelems = SIZE_1(gradSecond);
	  const int bot1index = ((n * SIZE_2(gradSecond)) + (m-4)) * SIZE_3(gradSecond) + (l-4);
	  gradSecond[bot1index + intSample*SIZE_1(gradSecond)*SIZE_2(gradSecond)*SIZE_3(gradSecond)] = sum / (float)sumelems;
	} }
"""


def cupy_kernel(strFunction, objectVariables):
    strKernel = globals()[strFunction]

    while True:
        objectMatch = re.search(r"(SIZE_)([0-4])(\()([^\)]*)(\))", strKernel)

        if objectMatch is None:
            break

        intArg = int(objectMatch.group(2))

        strTensor = objectMatch.group(4)
        intSizes = objectVariables[strTensor].size()

        strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))

    while True:
        objectMatch = re.search(r"(VALUE_)([0-4])(\()([^\)]+)(\))", strKernel)

        if objectMatch is None:
            break

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(",")

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = [
            "(("
            + strArgs[intArg + 1].replace("{", "(").replace("}", ")").strip()
            + ")*"
            + str(intStrides[intArg])
            + ")"
            for intArg in range(intArgs)
        ]

        strKernel = strKernel.replace(
            objectMatch.group(0), strTensor + "[" + str.join("+", strIndex) + "]"
        )

    return strKernel


try:

    @cupy.memoize(for_each_device=True)
    def cupy_launch(strFunction, strKernel):
        return cupy.RawModule(code=strKernel).get_function(strFunction)

except:
    pass


class _FunctionCorrelation(torch.autograd.Function):
    @staticmethod
    def forward(self, first, second):
        rbot0 = first.new_zeros(
            [first.size(0), first.size(2) + 8, first.size(3) + 8, first.size(1)]
        )
        rbot1 = first.new_zeros(
            [first.size(0), first.size(2) + 8, first.size(3) + 8, first.size(1)]
        )

        self.save_for_backward(first, second, rbot0, rbot1)

        first = first.contiguous()
        second = second.contiguous()

        output = first.new_zeros([first.size(0), 81, first.size(2), first.size(3)])

        if first.is_cuda == True:
            n = first.size(2) * first.size(3)
            cupy_launch(
                "kernel_Correlation_rearrange",
                cupy_kernel(
                    "kernel_Correlation_rearrange", {"input": first, "output": rbot0}
                ),
            )(
                grid=tuple([int((n + 16 - 1) / 16), first.size(1), first.size(0)]),
                block=tuple([16, 1, 1]),
                args=[n, first.data_ptr(), rbot0.data_ptr()],
                stream=Stream,
            )

            n = second.size(2) * second.size(3)
            cupy_launch(
                "kernel_Correlation_rearrange",
                cupy_kernel(
                    "kernel_Correlation_rearrange", {"input": second, "output": rbot1}
                ),
            )(
                grid=tuple([int((n + 16 - 1) / 16), second.size(1), second.size(0)]),
                block=tuple([16, 1, 1]),
                args=[n, second.data_ptr(), rbot1.data_ptr()],
                stream=Stream,
            )

            n = output.size(1) * output.size(2) * output.size(3)
            cupy_launch(
                "kernel_Correlation_updateOutput",
                cupy_kernel(
                    "kernel_Correlation_updateOutput",
                    {"rbot0": rbot0, "rbot1": rbot1, "top": output},
                ),
            )(
                grid=tuple([output.size(3), output.size(2), output.size(0)]),
                block=tuple([32, 1, 1]),
                shared_mem=first.size(1) * 4,
                args=[n, rbot0.data_ptr(), rbot1.data_ptr(), output.data_ptr()],
                stream=Stream,
            )

        elif first.is_cuda == False:
            raise NotImplementedError()

        return output

    @staticmethod
    def backward(self, gradOutput):
        first, second, rbot0, rbot1 = self.saved_tensors

        gradOutput = gradOutput.contiguous()

        assert gradOutput.is_contiguous() == True

        gradFirst = (
            first.new_zeros(
                [first.size(0), first.size(1), first.size(2), first.size(3)]
            )
            if self.needs_input_grad[0] == True
            else None
        )
        gradSecond = (
            first.new_zeros(
                [first.size(0), first.size(1), first.size(2), first.size(3)]
            )
            if self.needs_input_grad[1] == True
            else None
        )

        if first.is_cuda == True:
            if gradFirst is not None:
                for intSample in range(first.size(0)):
                    n = first.size(1) * first.size(2) * first.size(3)
                    cupy_launch(
                        "kernel_Correlation_updateGradFirst",
                        cupy_kernel(
                            "kernel_Correlation_updateGradFirst",
                            {
                                "rbot0": rbot0,
                                "rbot1": rbot1,
                                "gradOutput": gradOutput,
                                "gradFirst": gradFirst,
                                "gradSecond": None,
                            },
                        ),
                    )(
                        grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                        block=tuple([512, 1, 1]),
                        args=[
                            n,
                            intSample,
                            rbot0.data_ptr(),
                            rbot1.data_ptr(),
                            gradOutput.data_ptr(),
                            gradFirst.data_ptr(),
                            None,
                        ],
                        stream=Stream,
                    )

            if gradSecond is not None:
                for intSample in range(first.size(0)):
                    n = first.size(1) * first.size(2) * first.size(3)
                    cupy_launch(
                        "kernel_Correlation_updateGradSecond",
                        cupy_kernel(
                            "kernel_Correlation_updateGradSecond",
                            {
                                "rbot0": rbot0,
                                "rbot1": rbot1,
                                "gradOutput": gradOutput,
                                "gradFirst": None,
                                "gradSecond": gradSecond,
                            },
                        ),
                    )(
                        grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                        block=tuple([512, 1, 1]),
                        args=[
                            n,
                            intSample,
                            rbot0.data_ptr(),
                            rbot1.data_ptr(),
                            gradOutput.data_ptr(),
                            None,
                            gradSecond.data_ptr(),
                        ],
                        stream=Stream,
                    )

        elif first.is_cuda == False:
            raise NotImplementedError()

        return gradFirst, gradSecond


class _FunctionCorrelationTranspose(torch.autograd.Function):
    @staticmethod
    def forward(self, input, second):
        rbot0 = second.new_zeros(
            [second.size(0), second.size(2) + 8, second.size(3) + 8, second.size(1)]
        )
        rbot1 = second.new_zeros(
            [second.size(0), second.size(2) + 8, second.size(3) + 8, second.size(1)]
        )

        self.save_for_backward(input, second, rbot0, rbot1)

        input = input.contiguous()
        second = second.contiguous()

        output = second.new_zeros(
            [second.size(0), second.size(1), second.size(2), second.size(3)]
        )

        if second.is_cuda == True:
            n = second.size(2) * second.size(3)
            cupy_launch(
                "kernel_Correlation_rearrange",
                cupy_kernel(
                    "kernel_Correlation_rearrange", {"input": second, "output": rbot1}
                ),
            )(
                grid=tuple([int((n + 16 - 1) / 16), second.size(1), second.size(0)]),
                block=tuple([16, 1, 1]),
                args=[n, second.data_ptr(), rbot1.data_ptr()],
                stream=Stream,
            )

            for intSample in range(second.size(0)):
                n = second.size(1) * second.size(2) * second.size(3)
                cupy_launch(
                    "kernel_Correlation_updateGradFirst",
                    cupy_kernel(
                        "kernel_Correlation_updateGradFirst",
                        {
                            "rbot0": rbot0,
                            "rbot1": rbot1,
                            "gradOutput": input,
                            "gradFirst": output,
                            "gradSecond": None,
                        },
                    ),
                )(
                    grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                    block=tuple([512, 1, 1]),
                    args=[
                        n,
                        intSample,
                        rbot0.data_ptr(),
                        rbot1.data_ptr(),
                        input.data_ptr(),
                        output.data_ptr(),
                        None,
                    ],
                    stream=Stream,
                )

        elif second.is_cuda == False:
            raise NotImplementedError()

        return output

    @staticmethod
    def backward(self, gradOutput):
        input, second, rbot0, rbot1 = self.saved_tensors

        gradOutput = gradOutput.contiguous()

        gradInput = (
            input.new_zeros(
                [input.size(0), input.size(1), input.size(2), input.size(3)]
            )
            if self.needs_input_grad[0] == True
            else None
        )
        gradSecond = (
            second.new_zeros(
                [second.size(0), second.size(1), second.size(2), second.size(3)]
            )
            if self.needs_input_grad[1] == True
            else None
        )

        if second.is_cuda == True:
            if gradInput is not None or gradSecond is not None:
                n = second.size(2) * second.size(3)
                cupy_launch(
                    "kernel_Correlation_rearrange",
                    cupy_kernel(
                        "kernel_Correlation_rearrange",
                        {"input": gradOutput, "output": rbot0},
                    ),
                )(
                    grid=tuple(
                        [int((n + 16 - 1) / 16), gradOutput.size(1), gradOutput.size(0)]
                    ),
                    block=tuple([16, 1, 1]),
                    args=[n, gradOutput.data_ptr(), rbot0.data_ptr()],
                    stream=Stream,
                )

            if gradInput is not None:
                n = gradInput.size(1) * gradInput.size(2) * gradInput.size(3)
                cupy_launch(
                    "kernel_Correlation_updateOutput",
                    cupy_kernel(
                        "kernel_Correlation_updateOutput",
                        {"rbot0": rbot0, "rbot1": rbot1, "top": gradInput},
                    ),
                )(
                    grid=tuple(
                        [gradInput.size(3), gradInput.size(2), gradInput.size(0)]
                    ),
                    block=tuple([32, 1, 1]),
                    shared_mem=gradOutput.size(1) * 4,
                    args=[n, rbot0.data_ptr(), rbot1.data_ptr(), gradInput.data_ptr()],
                    stream=Stream,
                )

            if gradSecond is not None:
                for intSample in range(second.size(0)):
                    n = second.size(1) * second.size(2) * second.size(3)
                    cupy_launch(
                        "kernel_Correlation_updateGradSecond",
                        cupy_kernel(
                            "kernel_Correlation_updateGradSecond",
                            {
                                "rbot0": rbot0,
                                "rbot1": rbot1,
                                "gradOutput": input,
                                "gradFirst": None,
                                "gradSecond": gradSecond,
                            },
                        ),
                    )(
                        grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                        block=tuple([512, 1, 1]),
                        args=[
                            n,
                            intSample,
                            rbot0.data_ptr(),
                            rbot1.data_ptr(),
                            input.data_ptr(),
                            None,
                            gradSecond.data_ptr(),
                        ],
                        stream=Stream,
                    )

        elif second.is_cuda == False:
            raise NotImplementedError()

        return gradInput, gradSecond


def FunctionCorrelation(reference_features, query_features):
    return _FunctionCorrelation.apply(reference_features, query_features)


class ModuleCorrelation(torch.nn.Module):
    def __init__(self):
        super(ModuleCorrelation, self).__init__()

    def forward(self, tensorFirst, tensorSecond):
        return _FunctionCorrelation.apply(tensorFirst, tensorSecond)


def FunctionCorrelationTranspose(reference_features, query_features):
    return _FunctionCorrelationTranspose.apply(reference_features, query_features)


class ModuleCorrelationTranspose(torch.nn.Module):
    def __init__(self):
        super(ModuleCorrelationTranspose, self).__init__()

    def forward(self, tensorFirst, tensorSecond):
        return _FunctionCorrelationTranspose.apply(tensorFirst, tensorSecond)


class LocalCorr(ConvRefiner):
    def forward(self, x, y, flow):
        """Computes the relative refining displacement in pixels for a given image x,y and a coarse flow-field between them

        Args:
            x ([type]): [description]
            y ([type]): [description]
            flow ([type]): [description]

        Returns:
            [type]: [description]
        """
        with torch.no_grad():
            x_hat = F.grid_sample(y, flow.permute(0, 2, 3, 1), align_corners=False)
        corr = FunctionCorrelation(x, x_hat)
        d = self.block1(corr)
        d = self.hidden_blocks(d)
        d = self.out_conv(d)
        certainty, displacement = d[:, :-2], d[:, -2:]
        return certainty, displacement


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 128, 32, 32).to(device)
    y = torch.randn(2, 128, 32, 32).to(device)
    local_corr = LocalCorr(in_dim=81, hidden_dim=81 * 4)
    z = local_corr(x, y)
    print("hej")
