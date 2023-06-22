#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// #include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>

// int const CUDA_NUM_THREADS = sizeof(unsigned long long) * 8;
int const CUDA_NUM_THREADS = 1024;

inline int CUDA_GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// template<typename T>
float inline __device__ sgn(float x)
{
    return x>0?1.0:-1.0;
}
// template<typename T>
__global__ void afm_kernel(const int nthreads, const float* lines, const int* shape_info, const int num, const int height, const int width, float* afmap, int* aflabel)
{
    // aflabel[0] = 100;
    CUDA_1D_KERNEL_LOOP(index, nthreads){
        // printf("%d, %d\n",index,nthreads);
        // afmap[index]   = 1;
        // afmap[index+height*width] = 2;
        // aflabel[index] = index;
        int w = index % width;
        int h = (index / width) % height;
        int n = index / width / height;
        int x_index = n*2*height*width + h*width + w;
        int y_index = n*2*height*width + height*width + h*width + w;
        int label_index = n*height*width + h*width + w;
        // printf("%d, %d, %d, %d, %d\n",index,nthreads, n, h, w);

        
        float px = (float) w;
        float py = (float) h;
        int start = shape_info[n*4];
        int end   = shape_info[n*4+1];
        float min_dis = 1e30;
        for(int i = start; i < end; ++i) {
            float xs = (float)width  /(float)shape_info[n*4+3];
            float ys = (float)height /(float)shape_info[n*4+2];
            float x1 = lines[4*i]*xs;
            float y1 = lines[4*i+1]*ys;
            float x2 = lines[4*i+2]*xs;
            float y2 = lines[4*i+3]*ys;

            float dx = x2 - x1;
            float dy = y2 - y1;
            float norm2 = dx*dx + dy*dy;

            float t = ((px-x1)*dx + (py-y1)*dy)/(norm2+1e-6);
            t = t<1.0?t:1.0;
            t = t>0.0?t:0.0;

            float ax = x1   + t*(x2-x1) - px;
            float ay = y1   + t*(y2-y1) - py;

            float dis = ax*ax + ay*ay;
            if (dis < min_dis) {
                min_dis = dis;
                // ax_opt = -sgn(ax)*log(fabs(ax/float(width))  + 1e-6);
                // ay_opt = -sgn(ay)*log(fabs(ay/float(height)) + 1e-6);
                // afmap[x_index] = -sgn(ax)*log(fabs(ax/float(width))  + 1e-6);
                // afmap[y_index] = -sgn(ay)*log(fabs(ay/float(height)) + 1e-6);
                afmap[x_index] = ax;
                afmap[y_index] = ay;
                aflabel[label_index] = i - start;
            }            
        }
        // afmap[x_index]       = ax_opt;
        // afmap[y_index]       = ay_opt;
        // aflabel[label_index] = ind_opt-start;
    }
}

std::tuple<at::Tensor,at::Tensor> afm_cuda(
    const at::Tensor& lines,
    const at::Tensor& shape_info,
    const int height,
    const int width)
{
    auto batch_size = shape_info.size(0);
    auto afmap = at::zeros({batch_size,2,height,width}, lines.options());
    auto aflabel = at::zeros({batch_size,1,height,width}, lines.options().dtype(at::kInt));

    auto nthreads = batch_size*height*width;
    // printf("nthreads = %d\n",nthreads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    float* afmap_data = afmap.data<float>();
    int* aflabel_data = aflabel.data<int>();

    // printf("%.8f\n", log(1e-6));
    afm_kernel<<<CUDA_GET_BLOCKS(nthreads), CUDA_NUM_THREADS >>>(
            nthreads, 
            lines.contiguous().data<float>(),
            shape_info.contiguous().data<int>(),
            batch_size, height, width, 
            afmap_data,
            aflabel_data);    
    cudaDeviceSynchronize();
    // THCudaCheck(cudaMemcpy(&aflabel_host[0],aflabel_dev,
                // sizeof(int)*batch_size*height*width, cudaMemcpyDeviceToHost));
    // THCudaCheck(cudaMemcpy(&afmap_host[0],afmap_dev,
                // sizeof(int)*batch_size*2*height*width, cudaMemcpyDeviceToHost));
    
    // THCudaFree(state, aflabel_dev);
    // THCudaFree(state, afmap_dev);
    // THCudaCheck(cudaGetLastError());
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(afmap, aflabel);
}
