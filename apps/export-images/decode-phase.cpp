#include "decode-phase.h"

#include <halide_image_io.h>

#include <taskflow/algorithm/pipeline.hpp>

#include "constants.h"
#include "get_phase.h"
#include "read-slice.h"
#include "save_xml_raw.h"

using cmos::height;
using cmos::width;

DecodePhase::DecodePhase(const HighFive::File& f, const image_list_t& list)
    : dataset(f.getDataSet("himr")) {
    image_list.reserve(list.size());

    for (const auto& [path, image_param] : list) {
        using storage::PHASE;
        if (image_param.channel != PHASE) {
            continue;
        }

        image_list.emplace_back(path_t{image_param.well_id, image_param.format, path});
    }

    image_list.shrink_to_fit();
}

void
DecodePhase::definePipeflow() {
    auto read_layers = [&](tf::Pipeflow& pf) {
        const auto job_id = pf.token();
        if (job_id >= image_list.size()) {
            pf.stop();
            return;
        }

        const auto well_id = image_list[job_id].well_id;
        const auto path = image_list[job_id].path;
        std::cout << "Well[" << int(well_id) << "] -> " << path << std::endl;

        const auto line_id = pf.line();
        buffer[line_id] = storage::readQPILayers(dataset, well_id, width, height);
    };

    auto flatten_layers = [&](const tf::Pipeflow& pf) {
        const auto line_id = pf.line();
        auto& raw = std::get<input_t>(buffer[line_id]);

        output_t phase(width, height);

        const auto error = get_phase(raw, phase);
        assert(!error && "Halide error.");

        buffer[line_id] = std::move(phase);
    };

    auto write_image = [&](const tf::Pipeflow& pf) {
        const auto line_id = pf.line();
        auto phase_image = std::move(std::get<output_t>(buffer[line_id]));

        const auto job_id = pf.token();
        const auto& image_param = image_list[job_id];
        const std::string& output_filename = image_param.path;

        using f = storage::format_t;
        switch (image_param.format) {
            case f::TIF:
                [[fallthrough]];
            case f::PNG:
                Halide::Tools::convert_and_save_image(phase_image, output_filename);
                break;
            case f::XML:
                storage::saveXML(
                    storage::span_t<uint8_t>{phase_image.begin(), phase_image.number_of_elements()},
                    output_filename);
                break;
            case f::UNKNOWN:
                // Not implemented
                break;
        }
    };

    using tf::Pipe;
    using p = tf::PipeType;
    auto pipeline = new tf::Pipeline{
        n_lines,  //
        Pipe{p::SERIAL, std::move(read_layers)},
        Pipe{p::PARALLEL, std::move(flatten_layers)},
        Pipe{p::PARALLEL, std::move(write_image)},
    };

    convert = taskflow.composed_of(*pipeline);
    cleanup = taskflow.emplace([=]() { delete pipeline; });

    convert.name("Decode phase");
    cleanup.name("Cleanup");
}

void
DecodePhase::emplace() {
    definePipeflow();
}

void
DecodePhase::schedule() {
    convert.precede(cleanup);
}