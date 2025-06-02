wavevector_utils_dep = declare_dependency(
  include_directories: 'inc',
  dependencies: armadillo_dep,
)

test_wavevector_exe = executable('test_wavevector',
    sources: 'tests/test-wavevector.cpp',
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
