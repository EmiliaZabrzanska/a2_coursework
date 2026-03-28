import numpy as np
import matplotlib.pyplot as plt

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

def plot_kspace_magnitude(kspace, coil_axis):

    """Plot the log-scaled magnitude of k-space for each coil. """

    # determine number of coils and set up subplot grid dimensions
    n_coils = kspace.shape[coil_axis]
    n_cols = 3
    n_rows = n_coils // n_cols  

    # move the coil axis to the front for easy iteration: shape -> (n_coils, H, W)
    kspace_coils_first = np.moveaxis(kspace, coil_axis, 0)    

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 9))
    fig.suptitle(f'K-space magnitude (log1p scaled)', fontsize=16, y=1.02)

    # iterate over coils and flatten axes for easy indexing
    for coil_idx, ax in enumerate(axes.flat):

        # compute magnitude then compress range with log1p so zeros map to 0 with no warning
        kspace_mag = np.log1p(np.abs(kspace_coils_first[coil_idx]))

        # display as grayscale image
        im = ax.imshow(kspace_mag, cmap='gray')

        # set title and axes labels
        ax.set_title(f'Coil {coil_idx + 1}', fontsize=12)
        ax.set_xlabel('Pixel index', fontsize=10)
        ax.set_ylabel('Pixel index', fontsize=10)

        # add a colourbar beside each subplot
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cbar.set_label('Signal intensity', fontsize=10, labelpad=15)

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


def plot_magnitude_and_phase(images, coil_idx=0):

    """Plot the magnitude and phase image from a single coil."""

    # extract the image for chosen coil
    image = images[coil_idx]

    # compute magnitude 
    magnitude = np.abs(image)

    # compute phase in radians
    phase = np.angle(image)    

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Image space (Coil {coil_idx + 1})', fontsize=16, y=1.02)

    # magnitude image
    im0 = axes[0].imshow(magnitude, cmap='gray')

    # set title and turn off axes for readability
    axes[0].set_title('Magnitude', fontsize=12)
    axes[0].set_xlabel('Pixel index', fontsize=10)
    axes[0].set_ylabel('Pixel index', fontsize=10)

    # add a colourbar with label
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.05, pad=0.03)
    cbar0.set_label('Signal intensity', fontsize=10, labelpad=15)

    # phase image with vmin/vmax set to show full range of [-pi, pi]
    im1 = axes[1].imshow(phase, vmin=-np.pi, vmax=np.pi)

    # set title and axes labels
    axes[1].set_title('Phase (radians)', fontsize=12)
    axes[1].set_xlabel('Pixel index', fontsize=10)
    axes[1].set_ylabel('Pixel index', fontsize=10)

    # add a colourbar with ticks at key phase values, lavelling in terms of pi
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.05, pad=0.03)
    cbar1.set_label('Phase (rad)', fontsize=10, labelpad=15)
    cbar1.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar1.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

    plt.tight_layout()

# ---------------------------------------------------------------------------
# Exercise 2.1d: Plotting magnitude
# ---------------------------------------------------------------------------

def plot_magnitude(images):

    """Plot the magnitude image for each coil in image space"""

    # determine number of coils and set up subplot grid dimensions
    n_coils = images.shape[0]
    n_cols  = 3
    n_rows  = n_coils // n_cols  

    # create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 9))
    fig.suptitle('Image space magnitudes', fontsize=16, y=1.02)

    # iterate over coils and flatten axes for easy indexing
    for coil_idx, ax in enumerate(axes.flat):

        # compute magnitude from the complex image
        magnitude = np.abs(images[coil_idx])

        # display as grayscale image
        im = ax.imshow(magnitude, cmap='gray')

        # set title and axes labels
        ax.set_title(f'Coil {coil_idx + 1}', fontsize=12)
        ax.set_xlabel('Pixel index', fontsize=10)
        ax.set_ylabel('Pixel index', fontsize=10)

        # add a colourbar beside each subplot
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cbar.set_label('Signal intensity', fontsize=10, labelpad=15)

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


def plot_rsos(rsos):

    """Plot the root-sum-of-squares combined image """

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Root-sum-of-squares coil combination', fontsize=16, y=1.02)

    # display 2D array as an image with a grayscale colormap,
    im = ax.imshow(rsos, cmap='gray')

    # set axes labels
    ax.set_xlabel('Pixel index', fontsize=10)
    ax.set_ylabel('Pixel index', fontsize=10)

    # add a colorbar to the right of the image, controlling its size and distance from image
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label('Signal intensity', fontsize=10, labelpad=15)

    plt.tight_layout()