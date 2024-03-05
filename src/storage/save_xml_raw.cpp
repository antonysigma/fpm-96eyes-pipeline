#include "save_xml_raw.h"

#include <base64.h>
#include <zlib.h>

#include <algorithm>
#include <fstream>

namespace {

std::pair<std::vector<uint8_t>, size_t>
compressZlib(storage::span_t<uint8_t> src) {
    std::vector<uint8_t> dest(src.size);

    size_t compressed_size = dest.size();
    compress(dest.data(), &compressed_size, src.data, src.size);

    return {dest, compressed_size};
}

std::string
encodeBase64(const std::vector<uint8_t> raw_bytes, size_t count) {
    return base64_encode(raw_bytes.data(), count, false);
}
}  // namespace

namespace storage {

void
saveXML(const span_t<uint8_t> buffer, const std::string& filename) {
    std::ofstream xml_file{filename};

    xml_file << R"(<BinData Compression="zlib" BigEndian="false" )";

    auto [compressed, count] = compressZlib(buffer);
    const auto encoded = encodeBase64(std::move(compressed), count);
    xml_file << "Length=\"" << encoded.size() << "\">" << encoded << "</BinData>\n";
}

}  // namespace storage
