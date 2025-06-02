wavevector_utils_dep = declare_dependency(
  include_directories: 'inc',
  sources: 'src/wavevector_utility.cpp',
  dependencies: armadillo_dep,
)

test_wavevector_exe = executable('test_wavevector',
    sources: 'tests/test-wavevector.cpp',
    include_directories: common_inc,
    dependencies: [
      wavevector_utils_dep,
      catch2_dep,
    ],
)

test('Wavevector', test_wavevector_exe,
  args: [
    '-r', 'tap',
  ],
  protocol: 'tap',
)
