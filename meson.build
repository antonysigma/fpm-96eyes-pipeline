project('u96eyes', 'cpp', 'c',
  version : '0.1',
  default_options : [
    'buildtype=debugoptimized',
    'warning_level=3',
    'cpp_std=c++17',
    'cuda_std=c++14',
])

cxxopts_dep = subproject('cxxopts').get_variable('cxxopts_dep')
armadillo_dep = subproject('armadillo-code').get_variable('armadillo_dep')
mpi_dep = dependency('mpi', language: 'cpp')
catch2_dep = subproject('catch2').get_variable('catch2_with_main_dep')

# Do not warn about non-CUDA pragma statements
add_project_arguments([
  '-Xcompiler=-Wno-pedantic',
], language: 'cuda')

# Common data structures shared among modules
subdir('common')

# Wavevector estimation over a curved air-to-liquid interface (i.e. meniscus)
subdir('wavevector-calibration')

# File formats
subdir('metadata')
subdir('storage')

# Image processing algorithms, potentially GPU-accelerated.
subdir('algorithms')

# Test logic
subdir('test-data')
subdir('tests')

# End user application
subdir('apps')
