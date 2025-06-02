#include "FPM_datafile.h"

using namespace HighFive;

template <class Integer, class Float>
FPM_datafile<Integer, Float>::FPM_datafile(const char filename[], MPI_Comm& comm, MPI_Info& info)
    : File(filename, File::ReadWrite, MPIOFileDriver(comm, info)),
      imseqlow(getDataSet("imlow")),
      himr(getDataSet("himr")) {
    // for (int i = 0; i < FPM_HDF5_NDIMS; i++)
    //  {
    //    count[i] = 1;
    //    stride[i] = 1;
    //  }
    count.fill(1);
}

template <class Integer, class Float>
FPM_datafile<Integer, Float>::~FPM_datafile() {}

template <class Integer, class Float>
std::vector<Integer>&
FPM_datafile<Integer, Float>::get_read_buffer() {
    return imseqlow_buffer;
}

template <class Integer, class Float>
std::vector<std::complex<Float>>&
FPM_datafile<Integer, Float>::get_write_buffer() {
    return himr_buffer;
}

template <class Integer, class Float>
void
FPM_datafile<Integer, Float>::set_block_dims(size_t number_of_incidences, size_t height,
                                             size_t width, size_t wells) {
    block = {number_of_incidences, wells, height, width};

    const size_t length = block[0] * block[1] * block[2] * block[3];
    imseqlow_buffer.resize(length);
    himr_buffer.resize(length / number_of_incidences);
}

template <class Integer, class Float>
void
FPM_datafile<Integer, Float>::read_pupil() {
    getDataSet("initial_pupil")
        .select({0, 0, 0}, {block[1], block[2], block[3]})
        .read(himr_buffer.data());
}

template <class Integer, class Float>
void
FPM_datafile<Integer, Float>::write_pupil() {
    getDataSet("initial_pupil")
        .select({0, 0, 0}, {block[1], block[2], block[3]})
        .write_raw(himr_buffer.data());
}

template <class Integer, class Float>
template <typename T>
void
FPM_datafile<Integer, Float>::read_array(const std::string& dset, std::vector<T>& vector) {
    getDataSet(dset).select({0, 0}, {1, vector.size()}).read(vector.data());
    // array->access(data, offset, stride+2, count+2, _block , 'r');
}

template <class Integer, class Float>
template <typename T>
void
FPM_datafile<Integer, Float>::read_ndarray(const std::string& dset, T ptr) {
    getDataSet(dset).read(ptr);
}

template <class Integer, class Float>
template <typename T>
T
FPM_datafile<Integer, Float>::get_attribute(const std::string& name) const {
    T value;
    getAttribute(name).read(value);
    return value;
}

template <class Integer, class Float>
void
FPM_datafile<Integer, Float>::read_image(size_t row, size_t col) {
    ////TODO: Allow tile overlapping
    const std::vector<size_t> offset{0, 0, row, col};
    const std::vector<size_t> _block(block.begin(), block.end());
    // offset[0] = 0;
    // offset[1] = 0;

    // offset[2] = block[2] * row_id;
    // offset[3] = block[3] * col_id;

    // imseqlow->access (imseqlow_buffer, offset, stride, count, block, 'r', imseqlow_memspace);
    imseqlow.select(offset, _block).read(imseqlow_buffer.data());
}

template <class Integer, class Float>
void
FPM_datafile<Integer, Float>::read_phase_image(size_t row, size_t col,
                                               std::vector<std::complex<Float>>& buffer) {
    const std::vector<size_t> offset{0, 0, row, col};
    const std::vector<size_t> _block{4, block[1], block[2], block[3]};

    himr.select(offset, _block).read(buffer.data());
}

template <class Integer, class Float>
void
FPM_datafile<Integer, Float>::write_phase_image(size_t layer, size_t row, size_t col) {
    const std::vector<size_t> offset{layer, 0, row, col};
    const std::vector<size_t> _block{1, block[1], block[2], block[3]};

    himr.select(offset, _block).write_raw(himr_buffer.data());
}

template <class Integer, class Float>
void
FPM_datafile<Integer, Float>::write_local_pupil(size_t layer, size_t row, size_t col) {
    const std::vector<size_t> offset{layer, 0, row, col};
    const std::vector<size_t> _block{1, block[1], block[2], block[3]};

    getDataSet("corrected_pupil").select(offset, _block).write_raw(himr_buffer.data());
}

// template class FPM_datafile<uint16_t>;
