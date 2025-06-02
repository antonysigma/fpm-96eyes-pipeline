#pragma once
#include <cstdint>
#include <string>

namespace storage {

template <typename T>
struct span_t {
    T* data{nullptr};
    size_t size{};

    span_t<const char> as_bytes() const {
        return {reinterpret_cast<const char*>(data), size * sizeof(T)};
    }
};

void saveXML(const span_t<uint8_t> buffer, const std::string& filename);

}  // namespace storage
