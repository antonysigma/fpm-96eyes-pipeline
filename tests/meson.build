test_autofocus_exe = executable('test_autofocus',
    sources: [
        'test_autofocus.cpp',
        halide_generated_bin['plls'],
    ],
    cpp_args: [
        '-DHDF5_FILE_PATH="@0@"'.format(datafile_path),
    ],
    dependencies: [
        highfive_dep,
        armadillo_dep,
        halide_runtime_dep,
    ],
)

test_phase_exe = executable('test_phase',
    include_directories: '../storage',
    sources: [
        'test_phase.cpp',
        halide_generated_bin['get_phase'],
    ],
    cpp_args: [
        '-DHDF5_FILE_PATH="@0@"'.format(datafile_path),
    ],
    dependencies: [
        read_slice_dep,
        armadillo_dep,
        halide_runtime_dep,
    ],
)

test_fluorescence_exe = executable('test_fluorescence',
    include_directories: [
        '../storage',
    ],
    sources: [
        'test_fluorescence.cpp',
        halide_generated_bin['plls'],
        halide_generated_bin['raw2bgr'],
    ],
    cpp_args: [
        '-DHDF5_FILE_PATH="@0@"'.format(datafile_path),
    ],
    dependencies: [
        read_slice_dep,
        armadillo_dep,
        halide_runtime_dep,
    ],
)

if get_option('has_caltech_data')

#test('FPM data format', test_hdf5_exe)
test('Autofocus algorithm', test_autofocus_exe)
test('Phase channel tile stitching', test_phase_exe)
test('Fluorescence channel RGB conversion', test_fluorescence_exe)

#test('Halide accelerated pipelines',
#    unittest_exe,
#    suite: 'halide',
#    args: [
#        '-r', 'tap',
#        '[halide]',
#    ],
#    protocol: 'tap',
#)

else

warning('Skipping validation of image processing algorithms. Please request the raw data from Caltech, or contribute a mock test.')

endif