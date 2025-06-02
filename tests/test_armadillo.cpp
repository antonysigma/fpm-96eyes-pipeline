#include <iostream>

#include "wavevector_utility.hpp"

using namespace std;

int
main(int argc, char *argv[]) {
    //// Initialize MPI
    // MPI_Comm comm = MPI_COMM_WORLD;
    // MPI_Info info = MPI_INFO_NULL;

    // MPI_Init (&argc, &argv);
    // int mpi_size, mpi_rank;
    // MPI_Comm_size (comm, &mpi_size);
    // MPI_Comm_rank (comm, &mpi_rank);

    // FPM_datafile < unsigned short, float > h5file ("/dev/shm/test.hdf5", comm, info);

    constexpr auto n_leds = 49;
    wavevector_class wavevector_engine(n_leds);
    wavevector_engine.led_position = {
        {0., 0.},         {0., -0.003},     {0.003, 0.},      {0., 0.003},      {-0.003, 0.},
        {-0.003, -0.003}, {0.003, -0.003},  {0.003, 0.003},   {-0.003, 0.003},  {0., -0.006},
        {0.006, 0.},      {0., 0.006},      {-0.006, 0.},     {-0.006, -0.003}, {-0.003, -0.006},
        {0.003, -0.006},  {0.006, -0.003},  {0.006, 0.003},   {0.003, 0.006},   {-0.003, 0.006},
        {-0.006, 0.003},  {-0.006, -0.006}, {0.006, -0.006},  {0.006, 0.006},   {-0.006, 0.006},
        {0., -0.009},     {0.009, 0.},      {0., 0.009},      {-0.009, 0.},     {-0.009, -0.003},
        {-0.003, -0.009}, {0.003, -0.009},  {0.009, -0.003},  {0.009, 0.003},   {0.003, 0.009},
        {-0.003, 0.009},  {-0.009, 0.003},  {-0.009, -0.006}, {-0.006, -0.009}, {0.006, -0.009},
        {0.009, -0.006},  {0.009, 0.006},   {0.006, 0.009},   {-0.006, 0.009},  {-0.009, 0.006},
        {-0.009, -0.009}, {0.009, -0.009},  {0.009, 0.009},   {-0.009, 0.009}};
    // h5file.read_ndarray ("led_position", (double**) led_position.memptr());
    // wavevector_engine.led_position.print("led_position =");

    arma::cx_double tile_position(0.0e-3, 0.0);

    // h5file.get_attribute("led_height", led_height);
    // h5file.get_attribute("medium_height", medium_height);
    wavevector_engine.led_height = 33e-3;
    wavevector_engine.medium_height = 3e-3;
    wavevector_engine.medium_refractive_index = 1.33;
    wavevector_engine.numerical_aperture = 0.23;

    wavevector_engine.solve(tile_position);
    wavevector_engine.solution.print("wavevector =");
    // cout << "is_brightfield[0] = " << wavevector_engine.is_brightfield(0) << endl;

    // wavevector_engine.imseq.print("imseq =");
    // k.save("wavevectors.h5", arma::hdf5_binary);

    // wavevector_engine.imseq.save("imseq.txt", arma::raw_ascii);
    // wavevector_engine.solution.save("wavevector.txt", arma::raw_ascii);

    wavevector_engine.tile_width = 256;
    wavevector_engine.pixel_size = 0.4375e-6;
    wavevector_engine.wavelength = 533e-9;
    wavevector_engine.zeropad_factor = 2;

    const auto offset = wavevector_engine.get_offset(1);
    offset.print("offset[0] =");

    arma::umat all_offset(2, n_leds);
    for (auto i = 0; i < n_leds; i++) {
        all_offset.col(i) = wavevector_engine.get_offset(i);
    }
    all_offset.t().print("offset[:] =");

    return 0;
}
