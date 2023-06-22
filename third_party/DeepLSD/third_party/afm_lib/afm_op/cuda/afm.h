#pragma once

#include <torch/extension.h>

std::tuple<at::Tensor,at::Tensor> afm_cuda(
    const at::Tensor& lines,
    const at::Tensor& shape_info,
    const int height,
    const int width);

    