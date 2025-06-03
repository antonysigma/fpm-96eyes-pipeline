highfive_dep = subproject('highfive').get_variable('highfive_dep')

cxx = meson.get_compiler('cpp')
hdf5_parallel_dep = cxx.find_library('hdf5_openmpi')

base64_dep = subproject('cpp-base64').get_variable('base64_dep')

storage_inc = include_directories('.')

fpm_datafile_dep = declare_dependency(
  include_directories: storage_inc,
  compile_args: [
    '-DH5_HAVE_PARALLEL',
  ],
  dependencies: [
    highfive_dep,
    dependency('mpi', language: 'c'),
    hdf5_parallel_dep,
  ],
)

save_xml_raw_dep = declare_dependency(
  include_directories: storage_inc,
  sources: 'save_xml_raw.cpp',
  dependencies: [
    dependency('zlib'),
    base64_dep,
  ],
)

read_slice_dep = declare_dependency(
  include_directories: storage_inc,
  sources: 'read-slice.cpp',
  dependencies: [
    highfive_dep,
  ],
)