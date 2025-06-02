#include <armadillo>
#include <cassert>

/** Iterative root finding method to estimate wavevector.
 * Guaranteed to converge with h/x < 1 (need proof).
 * @param[in] x lateral position of the i-th LED
 * @param[in] h vertical distance from the sample to the LED
 * @param[in] t vertical distance from the sample to the surface of the mounting liquid
 * @param[in] n refractive index of the mounting liquid
 * @param[out] wavevectors of the i-th LED
 * @return true if solution exists
 */
static bool
find_k(const arma::vec& x, float h, float t, float n, arma::vec& k) {
    // Naive iteratiion method: guarantee to converge when h/x < 1 (need proof)
    float hmt2 = (h - t);
    hmt2 *= hmt2;

    arma::vec delta = x * (t / h);

    arma::vec new_delta, error;
    int i;
    const int N = 20;
    for (i = 0; i < N; i++) {
        new_delta = arma::sqrt(arma::square(x - delta) / (arma::square(x - delta) + hmt2) %
                               (t * t + arma::square(delta))) /
                    n;
        error = arma::abs(new_delta - delta) / (new_delta + arma::datum::eps);
        if ((bool)arma::all(error < 5e-6f)) break;
        delta = new_delta;
    }

    // Return values
    k = delta / arma::sqrt(square(delta) + t * t) * n;
    if (i == N - 1) return false;
    // throw "Warning: algorithm does not converge.";
    return true;
}

class wavevector_class {
    arma::vec k;
    arma::uvec sort_radius;

   public:
    arma::cx_vec led_position, solution;

    arma::uvec imseq;

    // arma::vec<bool> is_brightfield;
    // arma::vec<unsigned> update_seq;

    // Variables for estimating the wave vector
    const double meniscus_factor{5};
    double led_height, medium_height, medium_refractive_index, numerical_aperture;

    // Variables for mapping wave vector to pixels in FFT space
    double tile_width, pixel_size, wavelength, zeropad_factor;

    wavevector_class(unsigned number_of_led);

    /** Estimate wave vectors for a given tile position.
     * @param[in] tile_position xy coordinates of the tile_position from the center of field-of-view
     * @return true if solution exists.
     */
    bool solve(const arma::cx_double& tile_position);
    // TODO: convert to pixels

    /** Determine if the i-th low resolution image is brightfield
     * @param[in] i the i-th low resolution image
     * @return true if the image is brightfield
     */
    bool is_brightfield(unsigned short i) const;

    /** Obtain the pixel offset of the i-th low resolution image
     * @param[in] i the i-th low resolution image
     * @param[out] out the pixel offset
     */
    arma::uvec::fixed<2> get_offset(unsigned short i) const;

    /** Obtain the mask for the pupil function
     * @param[out] out the pixel offset
     */
    arma::cx_fmat get_pupil() const;
};

/********************************************************************************/

wavevector_class::wavevector_class(unsigned number_of_led) : led_position(number_of_led) {}

bool
wavevector_class::is_brightfield(unsigned short i) const {
    return arma::as_scalar(arma::abs(solution.row(i))) < numerical_aperture * 0.9;
}

arma::uvec::fixed<2>
wavevector_class::get_offset(unsigned short i) const {
    const arma::cx_double k_offset =
        arma::as_scalar(solution.row(i)) * tile_width * pixel_size / wavelength +
        arma::cx_double(tile_width, tile_width) * (zeropad_factor - 1) * 0.5;

    arma::uvec::fixed<2> out{k_offset.real(), k_offset.imag()};

    // assert(k_offset.real() >= 0.0);
    // assert(k_offset.imag() >= 0.0);
    // assert(arma::all(out <= tile_width));

    return out;
}

bool
wavevector_class::solve(const arma::cx_double& tile_position) {
    // Alternative way to construct cx_vec from vec
    // arma::cx_vec relative_position(led_position.row(0), led_position.row(1));
    // relative_position -= arma::cx_double( tile_position(0), tile_position(1) );

    arma::cx_vec relative_position = led_position - tile_position * meniscus_factor;

    bool success =
        find_k(arma::abs(relative_position), led_height, medium_height, medium_refractive_index, k);

    // Azimuth not required for cx_vec
    // arma::vec azimuth = arma::arg(relative_position);

    solution = relative_position / (arma::abs(relative_position) + arma::datum::eps) % k;

    // Map to pixels in FFT space

    // Sort by magnitude and phase angle
    sort_radius = arma::sort_index(k);
    imseq = arma::sort_index(arma::round(k / k(sort_radius(1))) +
                             arma::arg(solution) / arma::datum::pi / 2);

    return success;
}

arma::cx_fmat
wavevector_class::get_pupil() const {
    const auto x = arma::linspace<arma::vec>(-1.0, 1.0, tile_width);
    arma::mat xx(tile_width, tile_width), yy(tile_width, tile_width);

    xx.each_row() = x.t();
    yy.each_col() = x;

    const double pupil_radius = numerical_aperture * pixel_size / wavelength * 2;

    arma::cx_fmat out = arma::zeros<arma::cx_fmat>(tile_width, tile_width);
    out.elem(find(xx % xx + yy % yy < pupil_radius * pupil_radius)).ones();
    // Warning: matrix follows Fortran index order

    return out;
}
