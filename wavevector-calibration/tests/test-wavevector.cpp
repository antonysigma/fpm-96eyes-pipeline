#include <algorithm>
#include <catch2/catch_test_macros.hpp>

#include "types.h"
#include "wavevector_utility.hpp"

SCENARIO("Can find wavevector by fixed-point iterations") {
    constexpr auto n_leds = 49;
    WavevectorOverMeniscus wavevector_engine{n_leds};

    GIVEN("LED xy coordinates") {
        wavevector_engine.led_position = {
            {0., 0.},        {0., -0.003},     {0.003, 0.},      {0., 0.003},
            {-0.003, 0.},    {-0.003, -0.003}, {0.003, -0.003},  {0.003, 0.003},
            {-0.003, 0.003}, {0., -0.006},     {0.006, 0.},      {0., 0.006},
            {-0.006, 0.},    {-0.006, -0.003}, {-0.003, -0.006}, {0.003, -0.006},
            {0.006, -0.003}, {0.006, 0.003},   {0.003, 0.006},   {-0.003, 0.006},
            {-0.006, 0.003}, {-0.006, -0.006}, {0.006, -0.006},  {0.006, 0.006},
            {-0.006, 0.006}, {0., -0.009},     {0.009, 0.},      {0., 0.009},
            {-0.009, 0.},    {-0.009, -0.003}, {-0.003, -0.009}, {0.003, -0.009},
            {0.009, -0.003}, {0.009, 0.003},   {0.003, 0.009},   {-0.003, 0.009},
            {-0.009, 0.003}, {-0.009, -0.006}, {-0.006, -0.009}, {0.006, -0.009},
            {0.009, -0.006}, {0.009, 0.006},   {0.006, 0.009},   {-0.006, 0.009},
            {-0.009, 0.006}, {-0.009, -0.009}, {0.009, -0.009},  {0.009, 0.009},
            {-0.009, 0.009}};

        wavevector_engine.led_height = 33e-3;
        wavevector_engine.medium_height = 3e-3;
        wavevector_engine.medium_refractive_index = 1.33;
        wavevector_engine.numerical_aperture = 0.23;

        WHEN("Solve for wavevector") {
            arma::cx_double tile_position{0.0, 0.0};
            REQUIRE_NOTHROW(wavevector_engine.solve(tile_position));
            REQUIRE(wavevector_engine.solution.is_finite());

            THEN("Wavevector of center LED has small values") {
                constexpr auto tile_width = 256;
                wavevector_engine.tile_width = tile_width;
                wavevector_engine.pixel_size = 0.4375e-6;
                wavevector_engine.wavelength = 533e-9;
                wavevector_engine.zeropad_factor = 2;

                const auto offset = wavevector_engine.getOffset(1);
                REQUIRE(offset.is_finite());

                {
                    using namespace types;
                    REQUIRE(std::abs(static_cast<int32_t>(offset(X)) - tile_width / 2) <= 20);
                    REQUIRE(std::abs(static_cast<int32_t>(offset(Y)) - tile_width / 2) <= 20);
                }

                AND_THEN("Valid wavevector values") {
                    arma::umat all_offset(2, n_leds);
                    for (auto i = 0; i < n_leds; i++) {
                        all_offset.col(i) = wavevector_engine.getOffset(i);
                    }

                    REQUIRE(all_offset.is_finite());
                }
            }
        }
    }
}