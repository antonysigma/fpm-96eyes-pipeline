#include <armadillo>

/** Compute the Fourier-domain offsets of each low-resolution images due to
 * oblique illumination. */
class WavevectorOverMeniscus {
    /** Wavevector. */
    arma::vec k;

    /** Sort the wavevector by increasing radius. */
    arma::uvec sort_radius;

   public:
    arma::cx_vec led_position, solution;

    arma::uvec imseq;

    // Variables for estimating the wave vector
    const double meniscus_factor{5};
    double led_height, medium_height, medium_refractive_index, numerical_aperture;

    // Variables for mapping wave vector to pixels in FFT space
    double tile_width, pixel_size, wavelength, zeropad_factor;

    WavevectorOverMeniscus(unsigned number_of_led);

    /** Estimate wave vectors for a given tile position.
     * @param[in] tile_position xy coordinates of the tile_position from the center of field-of-view
     * @return true if solution exists.
     */
    bool solve(const arma::cx_double tile_position);
    // TODO: convert to pixels

    /** Determine if the i-th low resolution image is brightfield
     * @param[in] i the i-th low resolution image
     * @return true if the image is brightfield
     */
    bool isBrightfield(unsigned short i) const;

    /** Obtain the pixel offset of the i-th low resolution image
     * @param[in] i the i-th low resolution image
     * @param[out] out the pixel offset
     */
    arma::uvec::fixed<2> getOffset(unsigned short i) const;

    /** Obtain the mask for the pupil function
     * @param[out] out the pixel offset
     */
    arma::cx_fmat getPupil() const;
};