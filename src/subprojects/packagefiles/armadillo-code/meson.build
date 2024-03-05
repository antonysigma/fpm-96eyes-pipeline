project('armadillo', 'cpp',
    version: '10.1.x',
)

# Header-only library
armadillo_inc = include_directories('include')

armadillo_dep = declare_dependency(
    include_directories: armadillo_inc,
    dependencies: [
        dependency('openblas', required: false),
        dependency('lapack', required: false),
        dependency('arpack2', required: false),
    ]
)