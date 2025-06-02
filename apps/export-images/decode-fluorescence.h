#pragma once

#include <HalideBuffer.h>

#include <array>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>
#include <variant>

#include "metadata-parser.h"
#include "read-slice.h"
#include "tasks.hpp"

class DecodeFluorescence final : public Task {
    using image_list_t = std::map<std::string, storage::external_image_t>;

    struct job_t {
        uint8_t well_id{};
        const std::string* egfp_path{nullptr};
        const std::string* txred_path{nullptr};
        storage::format_t format{storage::format_t::PNG};
    };
    std::vector<job_t> image_list;

    using slice_t = ::storage::slice_t;
    struct input_t {
        slice_t egfp;
        slice_t txred;
    };

    using output_t = Halide::Runtime::Buffer<uint8_t>;

    using pipe_t = std::variant<input_t, output_t>;

    static constexpr auto n_wells = 96;
    static constexpr auto n_lines = 3;
    static constexpr size_t zsize = 11;

    const HighFive::File& file;
    const HighFive::DataSet dataset;

    std::array<uint8_t, n_wells> autofocus_plane;
    std::array<pipe_t, n_lines> buffer;

    tf::Task autofocus;
    tf::Task convert;
    tf::Task cleanup;

    void defineAutofocusTask();
    void definePipeflow();

   public:
    DecodeFluorescence(const HighFive::File& f, const image_list_t& l);

    void emplace() override;
    void schedule() override;
};