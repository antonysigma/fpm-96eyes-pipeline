#include "metadata-parser.h"

#include <cassert>
#include <pugixml.hpp>

namespace {

storage::format_t
str2format(const std::string &path) {
    using namespace std::string_view_literals;
    using f = storage::format_t;
    assert(path.size() > 3);

    // TODO(Antony): Use regex to decode file extensions
    const auto extension = path.substr(path.size() - 3);

    if (extension == "tif"sv) {
        return f::TIF;
    }
    if (extension == "xml"sv) {
        return f::XML;
    }
    if (extension == "png"sv) {
        return f::PNG;
    }

    return f::UNKNOWN;
}

storage::channel_t
str2channel(const char value[]) {
    using namespace std::string_view_literals;
    using ch = storage::channel_t;

    if (value == "Phase"sv) {
        return ch::PHASE;
    }
    if (value == "Intensity"sv) {
        return ch::INTENSITY;
    }
    if (value == "Brightfield"sv) {
        return ch::BRIGHTFIELD;
    }
    if (value == "EGFP"sv) {
        return ch::EGFP;
    }
    if (value == "TXRED"sv) {
        return ch::TXRED;
    }

    return ch::UNKNOWN;
}
}  // namespace

namespace storage {
struct MetadataParser::PrivateImplementation {
    pugi::xml_document doc;
    pugi::xml_parse_status status;
};

MetadataParser::MetadataParser(const char filename[])
    : pimpl{std::make_unique<PrivateImplementation>()} {
    const auto result = pimpl->doc.load_file(filename);
    pimpl->status = result.status;
}

MetadataParser::MetadataParser(const std::vector<char> &xml_payload)
    : pimpl{std::make_unique<PrivateImplementation>()} {
    pimpl->doc.load_buffer(xml_payload.data(), xml_payload.size());
}

MetadataParser::~MetadataParser() = default;

bool
MetadataParser::isParseSuccess() const {
    return (pimpl->status == pugi::status_ok);
}

uint8_t
MetadataParser::getI2CAddr() const {
    const auto &doc = pimpl->doc;

    // Import slave address
    auto node = doc.select_node("/Protocol/FPGA/i2c");

    char *err;
    const uint8_t slave_addr = std::strtoul(node.node().attribute("slave").value() + 2, &err, 16);

    return slave_addr;
}

std::pair<uint16_t, uint16_t>
MetadataParser::getImageDimensions() const {
    const auto &doc = pimpl->doc;

    auto node = doc.select_node("/Protocol/FPGA/image");
    const uint16_t width = node.node().attribute("width").as_int();
    const uint16_t height = node.node().attribute("height").as_int();

    return {width, height};
}

std::vector<cmos_register_t>
MetadataParser::getRegisters() const {
    const auto &doc = pimpl->doc;
    auto register_set = doc.select_nodes("/Protocol/FPGA/i2c/register");

    std::vector<cmos_register_t> register_list;
    register_list.reserve(50);

    for (const auto &reg : register_set) {
        char *err;
        const uint16_t reg_addr = std::strtoul(reg.node().attribute("addr").value() + 2, &err, 16);
        const uint8_t reg_val = std::strtoul(reg.node().attribute("value").value() + 2, &err, 16);

        register_list.emplace_back(cmos_register_t{reg_addr, reg_val});
    }

    return register_list;
}

std::map<std::string, external_image_t>
MetadataParser::getImageURL() const {
    const auto &doc = pimpl->doc;
    auto external_set = doc.select_nodes("//External");

    std::map<std::string, external_image_t> image_list;

    for (const auto &tag : external_set) {
        auto current_node = tag.node();
        const size_t well_id = [parent_node = current_node.parent().parent()]() -> size_t {
            const size_t col_id = parent_node.attribute("Column").as_int();
            const size_t row_id = parent_node.attribute("Row").as_int();
            return row_id * 12 + col_id;
        }();

        const auto channel = str2channel(current_node.attribute("channel").value());

        std::string path{current_node.attribute("filename").value()};
        const auto format = str2format(path);

        auto zlayer_attr = current_node.attribute("zlayer");
        const uint8_t plane_id = (zlayer_attr) ? zlayer_attr.as_int() : 0;

        image_list.emplace(std::move(path), external_image_t{static_cast<uint8_t>(well_id),
                                                             plane_id, channel, format});
    }

    return image_list;
}
}  // namespace storage