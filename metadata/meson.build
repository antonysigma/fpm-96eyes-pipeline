pugixml_dep = subproject('pugixml').get_variable('pugixml_dep')

metadata_parser_dep = declare_dependency(
  include_directories: 'inc',
  sources: 'src/metadata-parser.cpp',
  dependencies: pugixml_dep,
)

subdir('test-data')

test_metadata_exe = executable('test-metadata',
    sources: [
        'tests/test-metadata.cpp',
    ],
    cpp_args: [
        '-DREGISTER_XML="@0@"'.format(register_xml_path),
        '-DIMAGE_URL_XML="@0@"'.format(image_url_xml_path),
    ],
    dependencies: [
        catch2_dep,
        metadata_parser_dep,
    ],
)

test('XML metadata decoding',
    test_metadata_exe,
    suite: 'metadata',
    args: [
        '-r', 'tap',
    ],
    protocol: 'tap',
)
