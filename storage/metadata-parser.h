#pragma once
#include <array>
#include <bitset>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace storage {

struct cmos_register_t {
    uint16_t addr{};
    uint8_t value{};
};

enum channel_t { EGFP, TXRED, PHASE, INTENSITY, BRIGHTFIELD, UNKNOWN };

enum class format_t : uint8_t { TIF, XML, PNG, UNKNOWN };

constexpr size_t n_wells = 96;

struct pair_t {
    channel_t channel{channel_t::PHASE};
    uint8_t plane_id{};
};

struct external_image_t {
    uint8_t well_id{};
    uint8_t plane_id{};
    channel_t channel{PHASE};
    format_t format{format_t::PNG};
};

class MetadataParser {
   public:
    MetadataParser(const char filename[]);
    MetadataParser(const std::vector<char>& xml_payload);
    ~MetadataParser();

    bool isParseSuccess() const;
    uint8_t getI2CAddr() const;
    std::pair<uint16_t, uint16_t> getImageDimensions() const;
    std::vector<cmos_register_t> getRegisters() const;
    std::map<std::string, external_image_t> getImageURL() const;

   private:
    struct PrivateImplementation;
    std::unique_ptr<PrivateImplementation> pimpl;
};

}  // namespace storage