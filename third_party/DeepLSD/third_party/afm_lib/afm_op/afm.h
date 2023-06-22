#pragma once
#include "cuda/afm.h"

std::tuple<at::Tensor,at::Tensor> afm(
    const at::Tensor& lines,
    const at::Tensor& shape_info,
    const int height,
    const int width)
{
    return afm_cuda(lines,shape_info,height,width);
}
