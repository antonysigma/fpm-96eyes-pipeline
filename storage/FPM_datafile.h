#ifndef __FPM_DATAFILE__
#define __FPM_DATAFILE__

#include <mpi.h>

#include <array>
#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "highfive/H5DataSet.hpp"
#include "highfive/H5DataSpace.hpp"
#include "highfive/H5File.hpp"

// Patch to encode std::complex<float> in HDF5
#include "complex_float_support.hpp"

const int FPM_HDF5_NDIMS = 4;

template <class Integer, class Float>
class FPM_datafile final : private HighFive::File {
    std::array<size_t, FPM_HDF5_NDIMS> block;
    std::array<size_t, FPM_HDF5_NDIMS> count;

    std::vector<Integer> imseqlow_buffer;
    std::vector<std::complex<Float>> himr_buffer;

    HighFive::DataSet imseqlow, himr;

   public:
    FPM_datafile(const char filename[], MPI_Comm& comm, MPI_Info& info);
    ~FPM_datafile();

    std::vector<Integer>& get_read_buffer();

    std::vector<std::complex<Float>>& get_write_buffer();

    void set_block_dims(size_t number_of_incidences, size_t height, size_t width, size_t wells);

    void read_pupil();
    void write_pupil();

    template <typename T>
    void read_array(const std::string& dset, std::vector<T>& data);

    template <typename T>
    void read_ndarray(const std::string& dset, T data);

    template <typename T>
    T get_attribute(const std::string& name) const;

    void read_image(size_t row, size_t col);
    void read_phase_image(size_t row, size_t col, std::vector<std::complex<Float>>& buffer);
    void write_phase_image(size_t layer, size_t row, size_t col);
    void write_local_pupil(size_t layer, size_t row, size_t col);
};

#include "FPM_datafile_impl.hpp"
#endif
