taskflow_dep = subproject('taskflow').get_variable('taskflow_dep')

export_exe = executable('hdf5toimg',
    include_directories: [
        'utils/',
        'export-images/',
    ],
    sources: [
        'export-images/main.cpp',
        'export-images/decode-fluorescence.cpp',
        'export-images/decode-phase.cpp',
        halide_generated_bin['raw2bgr'],
        halide_generated_bin['get_phase'],
    ],
    cpp_args: [
        '-DHDF5_FILE_PATH="@0@"'.format(datafile_path),
        '-DCONFIG_PATH="@0@"'.format(image_url_xml_path),
    ],
    dependencies: [
        taskflow_dep,
        halide_runtime_dep,
        read_slice_dep,
        armadillo_dep,
        metadata_parser_dep,
        save_xml_raw_dep,
        cxxopts_dep,
    ],
)
