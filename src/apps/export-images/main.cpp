#include <cxxopts.hpp>

#include "decode-fluorescence.h"
#include "decode-phase.h"
#include "metadata-parser.h"

namespace {

using HighFive::File;

struct params_t {
    bool quit_now{true};
    std::string config_path{};
    std::string raw_data_path{};
};

params_t
parseArg(int argc, const char* const* argv) {
    using str = std::string;
    cxxopts::Options options{argv[0], "Export 96-well images as raw"};
    options.positional_help("[optional args]").show_positional_help();

    options.add_options()("h,help", "Print help")(
        "c,config", "Configuration definition file",
        cxxopts::value<str>())("i,input", "Input HDF5 file", cxxopts::value<str>());

    auto result = options.parse(argc, argv);

    if (result.count("help") || !result.count("config") || !result.count("input")) {
        std::cerr << options.help({""}) << std::endl;
        return {true, {}, {}};
    }

    return {false, std::move(result["config"].as<str>()), std::move(result["input"].as<str>())};
}

class TaskflowTimeProfiler {
    std::shared_ptr<tf::ChromeObserver> observer;

   public:
    /** Create the default observer */
    TaskflowTimeProfiler(tf::Executor* executor)
        : observer((executor) ? executor->make_observer<tf::ChromeObserver>() : nullptr) {}

    ~TaskflowTimeProfiler() {
        if (observer == nullptr) {
            return;
        }

        std::ofstream tracing_file("/tmp/tracing.json", std::ofstream::trunc);

        // dump the execution timeline to json (view at chrome://tracing)
        observer->dump(tracing_file);

        tracing_file.close();
    }
};

}  // namespace

int
main(int argc, char** argv) {
    const auto params = parseArg(argc, argv);
    if (params.quit_now) {
        return 0;
    }

    const auto image_list = [&]() {
        const auto parser = storage::MetadataParser{params.config_path.c_str()};

        assert(parser.isParseSuccess() && "XML config should be well formed");

        return parser.getImageURL();
    }();

    auto file = File(params.raw_data_path, File::ReadOnly);

    ////////////////////////////////////////////////////////////////////////////////
    DecodeFluorescence decode_fluorescence{file, image_list};
    DecodePhase decode_phase{file, image_list};

    decode_fluorescence.emplace();
    decode_phase.emplace();

    ////////////////////////////////////////////////////////////////////////////////
    decode_fluorescence.schedule();
    decode_phase.schedule();

    tf::Taskflow taskflow;
    auto decode_fluor_module = taskflow.composed_of(decode_fluorescence.taskflow);
    auto decode_phase_module = taskflow.composed_of(decode_phase.taskflow);
    decode_fluor_module.precede(decode_phase_module);

    // Now execute the multithreaded tasks
    tf::Executor executor;
    TaskflowTimeProfiler profiler(&executor);

    // Run all the tasks
    executor.run(std::move(taskflow)).wait();

    return 0;
}
