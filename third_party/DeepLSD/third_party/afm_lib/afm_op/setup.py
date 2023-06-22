from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os

extension_dir = '.'

main_file = glob.glob(os.path.join(extension_dir,'*.cpp'))
source_cuda = glob.glob(os.path.join(extension_dir,'cuda', '*.cu'))

sources = main_file + source_cuda

extra_compile_args = {'cxx': []}
defined_macros = []
extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        
extension = CUDAExtension

include_dirs = [extension_dir]

ext_module = [
    extension(
        "CUDA",
        sources,
        include_dirs=include_dirs,
        defined_macros=defined_macros,
        # extra_compile_args=extra_compile_args,
    )
]
setup(
    name='afm_op',
    ext_modules=ext_module,
    cmdclass={
        'build_ext': BuildExtension
    })
