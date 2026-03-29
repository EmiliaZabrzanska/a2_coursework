import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_wavelet, estimate_sigma
import cv2

# ---------------------------------------------------------------------------
# Exercise 2.1a: loading data
# ---------------------------------------------------------------------------

def load_kspace(path):

    """Load complex k-space data and identify the coil dimension"""

    # load using np.load routine as required
    kspace = np.load(path)

    # print properties
    print(f"Loaded k-space: dtype={kspace.dtype}, shape={kspace.shape}")

    # The coil dimension is the axis with size 6 (number of receiver coils)
    n_coils = 6

    # Find the coil axis
    coil_axis = kspace.shape.index(n_coils)
    print(f"Coil axis: {coil_axis}  (size={kspace.shape[coil_axis]})")

    return kspace, coil_axis

# ---------------------------------------------------------------------------
# Exercise 2.1b: visualising k-space
# ---------------------------------------------------------------------------

def plot_kspace_magnitude(kspace, coil_axis, title='K-space magnitude'):

    """Plot the log-scaled magnitude of k-space for each coil. """

    # determine number of coils and set up subplot grid dimensions
    n_coils = kspace.shape[coil_axis]
    n_cols = 3
    n_rows = n_coils // n_cols  

    # move the coil axis to the front for easy iteration: shape -> (n_coils, H, W)
    kspace_coils_first = np.moveaxis(kspace, coil_axis, 0)    

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 9))
    fig.suptitle(f'{title}', fontsize=16, y=1.02)

    # iterate over coils and flatten axes for easy indexing
    for coil_idx, ax in enumerate(axes.flat):

        # compute magnitude then compress range with log1p so zeros map to 0 with no warning
        kspace_mag = np.log1p(np.abs(kspace_coils_first[coil_idx]))

        # display as grayscale image, setting vmin/vmax to exclude extreme values
        im = ax.imshow(kspace_mag, cmap='gray', vmin=np.percentile(kspace_mag, 1), vmax=np.percentile(kspace_mag, 99))

        # set title and axes labels
        ax.set_title(f'Coil {coil_idx + 1}', fontsize=12)
        ax.set_xlabel('Pixel index', fontsize=10)
        ax.set_ylabel('Pixel index', fontsize=10)

        # add a colourbar beside each subplot conrolling its size and distance from image
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cbar.set_label('Signal intensity', fontsize=10, labelpad=10)

    # added spacing due to coloubars overlapping with titles 
    plt.tight_layout(h_pad=3, w_pad=2)

# ---------------------------------------------------------------------------
# Exercise 2.1c: transposing using fourier transform
# ---------------------------------------------------------------------------

def kspace_to_image(kspace, coil_axis):

    """Transform all coils from k-space into image space using the 2D inverse FFT. """

    # move coil axis to front for easy iteration (n_coils, H, W)
    kspace_coils_first = np.moveaxis(kspace, coil_axis, 0)

    # apply 2D inverse FFT to each coil
    images = np.fft.ifft2(kspace_coils_first)

    return images


def plot_magnitude_and_phase(images, coil_idx=0, title='Magnitude and Phase'):

    """Plot the magnitude and phase image from a single coil."""

    # extract the image for chosen coil
    image = images[coil_idx]

    # compute magnitude 
    magnitude = np.abs(image)

    # compute phase in radians
    phase = np.angle(image)    

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{title}', fontsize=16, y=1.02)

    # magnitude image, rotated for correct orientation and range limited to exclude extremes
    im0 = axes[0].imshow(np.rot90(magnitude), cmap='gray', vmin=np.percentile(magnitude, 1), vmax=np.percentile(magnitude, 99))

    # set title and turn off axes for readability
    axes[0].set_title('Magnitude', fontsize=12)
    axes[0].set_xlabel('Pixel index', fontsize=10)
    axes[0].set_ylabel('Pixel index', fontsize=10)

    # add a colourbar with label
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.05, pad=0.03)
    cbar0.set_label('Signal intensity', fontsize=10, labelpad=10)

    # phase image with vmin/vmax set to show full range of [-pi, pi]
    im1 = axes[1].imshow(np.rot90(phase), vmin=-np.pi, vmax=np.pi)

    # set title and axes labels
    axes[1].set_title('Phase (radians)', fontsize=12)
    axes[1].set_xlabel('Pixel index', fontsize=10)
    axes[1].set_ylabel('Pixel index', fontsize=10)

    # add a colourbar with ticks at key phase values, lavelling in terms of pi
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.05, pad=0.03)
    cbar1.set_label('Phase (rad)', fontsize=10, labelpad=10)
    cbar1.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar1.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

    plt.tight_layout()

# ---------------------------------------------------------------------------
# Exercise 2.1d: Plotting magnitude
# ---------------------------------------------------------------------------

def plot_magnitude(images, title='Magnitude Images'):

    """Plot the magnitude image for each coil in image space"""

    # determine number of coils and set up subplot grid dimensions
    n_coils = images.shape[0]
    n_cols  = 3
    n_rows  = n_coils // n_cols  

    # create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 9))
    fig.suptitle(f'{title}', fontsize=16, y=1.02)

    # iterate over coils and flatten axes for easy indexing
    for coil_idx, ax in enumerate(axes.flat):

        # compute magnitude from the complex image, rotated for correct orientation
        magnitude = np.rot90(np.abs(images[coil_idx]))

        # display as grayscale image, setting vmin/vmax to exclude extreme values
        im = ax.imshow(magnitude, cmap='gray', vmin=np.percentile(magnitude, 1), vmax=np.percentile(magnitude, 99))

        # set title and axes labels
        ax.set_title(f'Coil {coil_idx + 1}', fontsize=12)
        ax.set_xlabel('Pixel index', fontsize=10)
        ax.set_ylabel('Pixel index', fontsize=10)

        # add a colourbar beside each subplot
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cbar.set_label('Signal intensity', fontsize=10, labelpad=10)

    # added spacing due to coloubars overlapping with titles 
    plt.tight_layout(h_pad=3, w_pad=2)

# ---------------------------------------------------------------------------
# Exercise 2.1e: Combining coils
# ---------------------------------------------------------------------------

def root_sum_of_squares(images):

    """Combine images from all coils using root-sum-of-squares. """

    # compute the magnitude of each coil image
    magnitude = np.abs(images)

    # square the magnitudes
    squared_magnitudes = magnitude ** 2

    # sum across coils, setting axis=0 since coil dimension is at the front
    sums = np.sum(squared_magnitudes, axis=0)

    # square root to get root-sum-of-squares image
    rsos = np.sqrt(sums)

    return rsos


def plot_rsos(rsos, title='Root-sum-of-squares coil combination'):

    """Plot the root-sum-of-squares combined image """

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f'{title}', fontsize=16, y=1.02)

    # display 2D array as an image with a grayscale colormap, setting vmin/vmax to exclude extreme values
    im = ax.imshow(np.rot90(rsos), cmap='gray', vmin=np.percentile(rsos, 1), vmax=np.percentile(rsos, 99))

    # set axes labels
    ax.set_xlabel('Pixel index', fontsize=10)
    ax.set_ylabel('Pixel index', fontsize=10)

    # add a colorbar to the right of the image, controlling its size and distance from image
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label('Signal intensity', fontsize=10, labelpad=10)

    plt.tight_layout()

# ---------------------------------------------------------------------------
# Exercise 2.2a: Denoising with filters
# ---------------------------------------------------------------------------

# all denoising functions are implemented as shown in the in Lecture 9 23/02/2026

def gaussian_filtering(images, sigma=1):

    """Denoise all coil images using a Gaussian filter."""

    # create empty array to hold denoised images
    denoised = np.zeros_like(images)

    # loop over all coils and apply filter to each
    for i in range(images.shape[0]):
        denoised[i] = gaussian_filter(images[i], sigma=sigma)

    return denoised


def bilateral_filtering(images, d=10, sigma_color=0.2, sigma_space=8):

    """Denoise all coil images using a bilateral filter."""

    # create empty array to hold denoised images
    denoised = np.zeros_like(images)

    # loop over all coils and apply filter to each
    for i in range(images.shape[0]):
        
        # cv2 requires float32
        denoised[i] = cv2.bilateralFilter(images[i].astype(np.float32), d, sigma_color, sigma_space)

    return denoised


def wavelet_filtering(images):

    """Denoise all coil images using wavelet thresholding (BayesShrink) """

    # create empty array to hold denoised images
    denoised = np.zeros_like(images)

    # loop over all coils and apply filter to each
    for i in range(images.shape[0]):
        denoised[i] = denoise_wavelet(
            images[i],
            method='BayesShrink',    # automatically estimate noise level per subband
            mode='soft',             # soft thresholding shrinks coefficients smoothly
            channel_axis=None,       # grayscale image has no channel axis
            rescale_sigma=True       # rescale noise estimate to image scale
        )

    return denoised

def plot_denoising_comparison(original, denoised_gaussian, denoised_bilateral, denoised_wavelet, title='Denoising comparison for all coils'):

    """Plot all denoising results in a single figure."""

    # determine number of coils and set up subplot grid dimensions
    n_coils = original.shape[0]
    methods = [original, denoised_gaussian, denoised_bilateral, denoised_wavelet]
    row_titles = ['Original', 'Gaussian', 'Bilateral', 'Wavelet']                  

    # create subplots
    fig, axes = plt.subplots(4, n_coils, figsize=(3 * n_coils, 12))
    fig.suptitle(f'{title}', fontsize=16, y=1.02)

    for row, method_data in enumerate(methods):

        for col in range(n_coils):

            ax = axes[row, col]

            # use the same vmin/vmax for both images for easy comaprisons
            vmin = np.percentile(original[col], 1)
            vmax = np.percentile(original[col], 99)

            # display as rotated grayscale image with consistent scaling across methods
            ax.imshow(np.rot90(method_data[col]), cmap='gray', vmin=vmin, vmax=vmax)

            # set axes labels
            ax.set_xlabel('Pixel index', fontsize=10)

            # only add titles on the first column
            if col == 0:
                ax.set_ylabel(f'{row_titles[row]}\nPixel index', fontsize=10)
            else:
                ax.set_ylabel('Pixel index', fontsize=10)

            # only add method titles on the first row
            if row == 0:
                ax.set_title(f'Coil {col + 1}', fontsize=12)

    plt.tight_layout()

# ---------------------------------------------------------------------------
# Exercise 2.2b: Butterworth filter
# ---------------------------------------------------------------------------

# used butterworth filter code as given in coursework instructions

def butterworth_lowpass_filter(shape, D0=100, n=2):

    """Create a 2D Butterworth low-pass filter centred at the array centre."""

    # extract dimensions of the image
    P, Q = shape[0], shape[1]

    # create coordinate arrays centred at (0, 0)
    u = np.arange(P) - P // 2
    v = np.arange(Q) - Q // 2

    # create 2D grids of u and v coordinates
    U, V = np.meshgrid(u, v, indexing='ij')

    # compute distance from centre for each point
    D = np.sqrt(U**2 + V**2)

    # filter smoothly attenuates frequencies beyond cutoff D0, n controls the steepness
    H = 1 / (1 + (D / D0) ** (2 * n))

    return H


def apply_butterworth(kspace, coil_axis, coil_idx=0, D0=100, n=2):  

    """Apply a Butterworth low-pass filter in k-space for a single coil."""

    # move coil axis to front and extract the chosen coil: shape (H, W)
    kspace_coils_first = np.moveaxis(kspace, coil_axis, 0)
    kspace_coil = kspace_coils_first[coil_idx]

    # build the Butterworth filter matching the k-space shape
    H = butterworth_lowpass_filter(kspace_coil.shape, D0=D0, n=n)

    # apply filter
    kspace_filtered = kspace_coil * H

    # transform filtered k-space back to image space
    image_filtered = np.fft.ifft2(kspace_filtered)

    return image_filtered

# plot using the magnitude and phase plot defined in Ex 2.1

# ---------------------------------------------------------------------------
# Exercise 2.2c: Denoised combined image
# ---------------------------------------------------------------------------

def plot_rsos_denoised(rsos, rsos_denoised, title='Denoised combined coil image'):

    """Plot original and denoised rSoS images side by side for comparison."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=16, y=1.02)

    # iterate over both images, rotating axis for correct arientation
    for ax, image, subtitle in zip(axes, [np.rot90(rsos), np.rot90(rsos_denoised)], ['Original', 'Wavelet Denoised']):

        # plot image with range limited to exclude extremes
        im = ax.imshow(image, cmap='gray', vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
        
        # set title and axes labels
        ax.set_title(subtitle, fontsize=12)
        ax.set_xlabel('Pixel index', fontsize=10)
        ax.set_ylabel('Pixel index', fontsize=10)

        # add a colour bar to the right of the image, controlling its size and distance from image
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cbar.set_label('Signal intensity', fontsize=10, labelpad=10)

        # set colour bar ticks because they were being formatted differently across each subplot
        cbar.set_ticks([0.5, 1.0, 1.5, 2.0, 2.5])

    plt.tight_layout()



