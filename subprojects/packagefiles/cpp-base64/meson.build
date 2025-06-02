project('cpp-base64', 'cpp')

base64_dep = declare_dependency(
    include_directories: '.',
    sources: 'base64.cpp',
)

base64_test_exe = executable('base64-test',
    sources: 'test.cpp',
    dependencies: base64_dep,
)

test('Base64 test', base64_test_exe)