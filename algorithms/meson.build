halide_toolchain = subproject('halide-@0@-@1@'.format(build_machine.cpu_family(), build_machine.system()))

halide_codegen_dep = halide_toolchain.get_variable('halide_generator_dep')
halide_library_path = halide_toolchain.get_variable('halide_library_path')
halide_runtime_dep = halide_toolchain.get_variable('halide_runtime_dep')

halide_fft_inc = subproject('halide-fft').get_variable('halide_fft_inc')

halide_codegen_exe = executable('codegen',
    include_directories: [
        '.',
        halide_fft_inc,
        common_inc,
    ],
    cpp_args: [
        '-Wno-unused-parameter',
        '-Wno-unused-function',
    ],
    sources: [
        'autofocus_generator.cpp',
        'fluorescence_generator.cpp',
        'phase_generator.cpp',
    ],
    dependencies: [
        halide_codegen_dep,
    ],
)

halide_pipelines = [
    {'name': 'plls'},
    {'name': 'get_phase'},
    {'name': 'raw2bgr'},
]

tile_size = 256

halide_generated_bin = {}

foreach p : halide_pipelines
    if p.has_key('auto_schedule') and not p['auto_schedule']
        halide_codegen_args = [
            'target=host-cuda-cuda_capability_75-no_asserts',
        ]
    else
        halide_codegen_args = [
            'target=host',
            '-p', 'autoschedule_mullapudi2016',
            'autoscheduler=Mullapudi2016',
            # Maximum level of CPU core, or GPU threads available
            'autoscheduler.parallelism=12',
            # Size of last level (L2) cache
            'autoscheduler.last_level_cache_size=@0@'.format(4 * 1024 * 1024),
            # Ratio of the cache read cost to compute cost
            'autoscheduler.balance=40',
        ]
    endif

    if p.has_key('specify_tile_size') and p['specify_tile_size']
        halide_codegen_args += ['tile_size=@0@'.format(tile_size)]
    endif

    halide_generated_bin += {p['name']: custom_target(
        p['name'] + '.[ah]',
        output: [
            p['name'] + '.a',
            p['name'] + '.h',
        ],
        input: halide_codegen_exe,
        env: { 'LD_LIBRARY_PATH': halide_library_path },
        command: [
            halide_codegen_exe,
            '-o', meson.current_build_dir(),
            '-g', p['name'],
            '-e', 'static_library,h,html',
        ] + halide_codegen_args,
    )}

    alias_target('halide_' + p['name'], halide_generated_bin[p['name']])
endforeach
