#include "decode-fluorescence.h"

#include <halide_image_io.h>

#include <armadillo>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>
#include <taskflow/algorithm/pipeline.hpp>

#include "constants.h"
#include "raw2bgr.h"
#include "save_xml_raw.h"

using cmos::height;
using cmos::width;

void
DecodeFluorescence::defineAutofocusTask() {
    autofocus = taskflow.emplace([&]() {
        file.getDataSet("autofocus_plane")
            .select({0, 0}, {n_wells, 1})
            .read(autofocus_plane.data());
    });

    autofocus.name("autofocus");
}

void
DecodeFluorescence::definePipeflow() {
    auto readFocalPlanes = [&](tf::Pipeflow& pf) {
        const auto job_id = pf.token();
        if (job_id >= image_list.size()) {
            pf.stop();
            return;
        }

        const auto well_id = image_list[job_id].well_id;

        arma::Col<size_t>::fixed<2> plane_id;
        plane_id(0) = autofocus_plane[well_id];

        constexpr auto focal_shift = 2;
        plane_id(1) = std::min(plane_id(0) + focal_shift, zsize - 1);

        using storage::readSlice;
        auto egfp_plane = readSlice(dataset, well_id, plane_id(0), width, height);
        auto txred_plane = readSlice(dataset, well_id, plane_id(1), width, height);

        const auto line_id = pf.line();
        buffer[line_id] = input_t{std::move(egfp_plane), std::move(txred_plane)};
    };

    auto decode = [&](const tf::Pipeflow& pf) {
        using storage::slice_t;

        const auto line_id = pf.line();
        auto& image_pair = std::get<input_t>(buffer[line_id]);

        using Halide::Runtime::Buffer;
        output_t output(width, height, 3);

        // TODO(Antony): Output the interleaved image, not planar.
        const auto error = raw2bgr(image_pair.egfp, image_pair.txred, output);
        assert(!error && "Halide error.");

        buffer[line_id] = std::move(output);
    };

    auto writeImage = [&](const tf::Pipeflow& pf) {
        const auto line_id = pf.line();
        auto normalized_image = std::move(std::get<output_t>(buffer[line_id]));

        const auto job_id = pf.token();
        const int well_id = image_list[job_id].well_id;
        const auto format = image_list[job_id].format;

        const auto writeSlice = [&](int idx, const std::string& path) {
            auto slice = normalized_image.sliced(2, idx);
            using f = storage::format_t;
            switch (format) {
                case f::TIF:
                    [[fallthrough]];
                case f::PNG:
                    Halide::Tools::convert_and_save_image(slice, path);
                    break;
                case f::XML:
                    storage::saveXML(
                        storage::span_t<uint8_t>{slice.begin(), slice.number_of_elements()}, path);
                    break;
                case f::UNKNOWN:
                    // Not implemented
                    break;
            }
            std::cout << "Well[" << well_id << "] -> " << path << std::endl;
        };

        const std::string* egfp_path = image_list[job_id].egfp_path;
        if (egfp_path != nullptr) {
            writeSlice(1, *egfp_path);
        }

        const auto* txred_path = image_list[job_id].txred_path;
        if (txred_path != nullptr) {
            writeSlice(0, *txred_path);
        }
    };

    using p = tf::PipeType;
    using tf::Pipe;
    auto* pipeline = new tf::Pipeline{n_lines,  //
                                      Pipe{p::SERIAL, std::move(readFocalPlanes)},
                                      Pipe{p::PARALLEL, std::move(decode)},
                                      Pipe{p::PARALLEL, std::move(writeImage)}};

    convert = taskflow.composed_of(*pipeline);
    cleanup = taskflow.emplace([=]() { delete pipeline; });

    convert.name("decode_fluorescence");
    cleanup.name("cleanup");
}

DecodeFluorescence::DecodeFluorescence(const HighFive::File& f, const image_list_t& list)
    : file(f), dataset(f.getDataSet("fluorescence")) {
    std::map<uint8_t, job_t> aggregated;

    // Manual implementation of SQL query:
    // SELECT well_id, channel, path FORM list WHERE channel = EGFP OR channel = TXRED;
    for (const auto& [path, image_param] : list) {
        using storage::EGFP;
        using storage::TXRED;

        const auto ch = image_param.channel;
        if (ch != EGFP && ch != TXRED) {
            continue;
        }

        // Create entry if not exist
        const uint8_t well_id = image_param.well_id;
        auto& entry = aggregated[well_id];

        entry.well_id = well_id;
        entry.format = image_param.format;
        if (ch == EGFP) {
            entry.egfp_path = &path;
        } else if (ch == TXRED) {
            entry.txred_path = &path;
        }
    }

    image_list.resize(aggregated.size());
    std::transform(aggregated.begin(), aggregated.end(), image_list.begin(),
                   [](const auto& p) -> job_t { return p.second; });
}

void
DecodeFluorescence::emplace() {
    defineAutofocusTask();
    definePipeflow();
}

void
DecodeFluorescence::schedule() {
    autofocus.precede(convert);
    convert.precede(cleanup);
}