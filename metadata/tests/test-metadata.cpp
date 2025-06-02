#include <algorithm>
#include <catch2/catch_test_macros.hpp>

#include "metadata-parser.h"

SCENARIO("Can decode register list", "[metadata]") {
    GIVEN("Source XML config") {
        constexpr char register_xml_path[]{REGISTER_XML};

        WHEN("Decode XML") {
            storage::MetadataParser parser{register_xml_path};

            REQUIRE(parser.isParseSuccess());

            THEN("Found image dimensions") {
                const auto [width, height] = parser.getImageDimensions();

                REQUIRE(width == 2592);
                REQUIRE(height == 1944);
            }

            THEN("Found CMOS I2C slave addr") {
                const auto i2c_addr = parser.getI2CAddr();

                REQUIRE(i2c_addr != 0);
            }

            THEN("Found register list") {
                const auto reg_list = parser.getRegisters();

                REQUIRE(!reg_list.empty());
            }
        }
    }
}

SCENARIO("Can decode href of external images", "[metadata]") {
    GIVEN("Source XML config") {
        constexpr char path[]{IMAGE_URL_XML};

        WHEN("Decode image URL from XML") {
            storage::MetadataParser parser{path};
            REQUIRE(parser.isParseSuccess());

            const auto image_list = parser.getImageURL();

            // TODO(Antony): Simulate test-vector of all 96 wells.
            THEN("Meaningful URL") {
                const auto& path = image_list.begin()->first;
                INFO("Path = " << path);
                REQUIRE(!path.empty());
            }

            const auto image_param = image_list.begin()->second;
            const auto [well_id, plane_id, channel, format] = image_param;
            UNSCOPED_INFO("Image = {well:" << int(well_id) << ", ch:" << channel
                                           << ", z:" << int(plane_id) << "}");

            THEN("Known channel") { REQUIRE(channel != storage::channel_t::UNKNOWN); }
            THEN("Known format") { REQUIRE(format != storage::format_t::UNKNOWN); }
        }
    }
}