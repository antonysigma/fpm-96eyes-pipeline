catch2_dep = subproject('catch2').get_variable('catch2_with_main_dep')

unittest_exe = executable('unittest',
    sources: [
        'test_metadata.cpp',
    ],
    cpp_args: [
        # Needed for define_extern and symload
        '-fPIC',
        '-DTEST_TILE_SIZE=@0@'.format(tile_size),
        '-DREGISTER_XML="@0@"'.format(register_xml_path),
        '-DIMAGE_URL_XML="@0@"'.format(image_url_xml_path),
    ],
    dependencies: [
        catch2_dep,
        halide_runtime_dep,
        armadillo_dep,
        metadata_parser_dep,
    ],
)

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

test_wavevector_exe = executable('test_wavevector',
    sources: 'test_armadillo.cpp',
    dependencies: wavevector_utils_dep,
)

#test('FPM data format', test_hdf5_exe)
test('Wavevector', test_wavevector_exe)

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

test('XML metadata decoding',
    unittest_exe,
    suite: 'metadata',
    args: [
        '-r', 'tap',
        '[metadata]',
    ],
    protocol: 'tap',
)