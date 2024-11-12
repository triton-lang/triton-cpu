from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

xsmm_root = os.getenv("XSMM_ROOT_DIR")
xsmm_lib = os.getenv("XSMM_LIB_DIR")
print(f'Using LIBXSMM root: {xsmm_root}')
print(f'LIBXSMM lib location: {xsmm_lib}')

setup(name='xsmm_py',
      ext_modules=[
          cpp_extension.CppExtension('xsmm_py', ['xsmm_utils.cpp'],
          include_dirs=[
              f'{xsmm_root}/include',
              f'{xsmm_root}/src/template'
          ],
          library_dirs=[f'{xsmm_lib}'],
          libraries=['xsmm', 'omp'],
          extra_compile_args=['-fopenmp']
      )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
