project('u96eyes', 'cpp', 'cuda', 'c',
  version : '0.1',
  default_options : [
    'buildtype=debugoptimized',
    'warning_level=3',
    'cpp_std=c++17',
    'cuda_std=c++14',
])

cxxopts_dep = subproject('cxxopts').get_variable('cxxopts_dep')
pugixml_dep = subproject('pugixml').get_variable('pugixml_dep')
armadillo_dep = subproject('armadillo-code').get_variable('armadillo_dep')
mpi_dep = dependency('mpi', language: 'cpp')

# Do not warn about non-CUDA pragma statements
add_project_arguments([
  '-Xcompiler=-Wno-pedantic',
], language: 'cuda')

subdir('common')
subdir('storage')
subdir('algorithms')

subdir('test-data')
subdir('tests')
subdir('apps')
