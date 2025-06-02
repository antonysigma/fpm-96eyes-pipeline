#pragma once

#include <HalideBuffer.h>

#include <array>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>
#include <variant>

#include "metadata-parser.h"
#include "read-slice.h"
#include "tasks.hpp"

class DecodePhase final : public Task {
    using image_list_t = std::map<std::string, storage::external_image_t>;

    using slice_t = ::storage::slice_t;
    using input_t = Halide::Runtime::Buffer<const float, 4>;

    using output_t = Halide::Runtime::Buffer<uint8_t, 2>;
    using pipe_t = std::variant<input_t, output_t>;

    static constexpr auto n_wells = storage::n_wells;
    static constexpr auto n_lines = 3;
    static constexpr size_t zsize = 4;

    const HighFive::DataSet dataset;

    struct path_t {
        uint8_t well_id{};
        storage::format_t format{storage::format_t::PNG};
        const std::string& path;
    };

    std::vector<path_t> image_list;
    std::array<pipe_t, n_lines> buffer;

    tf::Task convert;
    tf::Task cleanup;

    void definePipeflow();

   public:
    DecodePhase(const HighFive::File& f, const image_list_t& image_list);

    void emplace() override;
    void schedule() override;
};
