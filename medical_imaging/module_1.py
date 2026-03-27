import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line
from skimage.io import imread
from skimage.transform import radon, iradon
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


# ---------------------------------------------------------------------------
# Exercise 1.1a: loading image
# ---------------------------------------------------------------------------

def load_image(path):

    """Load a CT image and normalise pixel values to attenuation units [0, 0.001]."""

    raw = imread(path)

    # we need a 2D image (greyscale) so need to make sure all images are converted
    if raw.ndim == 3:
        # slices array to keep first 3 channels and drop alpha, then convert to grayscale and normalise to [0, 1]
        raw = rgb2gray(raw[:, :, :3])  
    else:
        # if already 2D, just normalise to [0, 1]
        raw = raw.astype(np.float64) / raw.max()

    # rescale to match the typical range of attenuation coefficients (0, 0.001)
    return raw / 1000.0

def visualise_image(image, title='Image'):

    """Visualise a 2D greyscale image with attenuation scaling."""

    # using fig, ax seperately to add a colorbar to the side
    fig, ax = plt.subplots(figsize=(8, 6))

    # display 2D array as an image with a grayscale colormap, setting vmin/vmax for consistent scaling
    im = ax.imshow(image, cmap='gray', vmin=0, vmax=image.max())

    ax.set_title(title)
    ax.axis('off')                                                              # hide axes for cleaner look

    # add a colorbar to the right of the image, controlling its size and distance from image
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label('Attenuation coefficient $\\mu$ [cm$^{-1}$]', labelpad=15)   # add space from bar with padding
    
    plt.tight_layout()

# ---------------------------------------------------------------------------
# Exercise 1.1b: simulating noise and sinograms
# ---------------------------------------------------------------------------

def simulate_sinogram(image, theta, I0):

    """Forward projects to obtain a clean sinogram, then simulates noise"""

    # forward project image using radon transfrom to get a clean absorption sinogram
    sinogram_clean = radon(image, theta)

    # converts into Beer-Lambert transmisson form
    transmission   = I0 * np.exp(-sinogram_clean)

    # adds poisson noise, using np.clip to ensure there are no negative values and converts back to float
    transmission   = np.random.poisson(np.clip(transmission, 0, None)).astype(np.float64)

    # adds gaussian noise on top of poisson noise
    transmission   = transmission + np.random.normal(0, 0.05, transmission.shape)

    # converts back to absorption sinogram, using np,clip to avoid negative values and log(0)
    sinogram_noisy = np.clip(-np.log(np.clip(transmission, 1e-10, None) / I0), 0, None)

    return sinogram_clean, sinogram_noisy


def plot_sinograms(sinograms_noisy, row_values, angle_counts, title='Sinograms', use_angle_labels=False):
    
    """Plot a 3x3 grid of noisy sinograms across dose levels or angular ranges and angle counts."""

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(title, fontsize=16, y=1.02)

    # loop over all row values (dose levels or angular ranges) and angle counts
    for row, row_val in enumerate(row_values):

        for col, ang in enumerate(angle_counts):

            ax   = axes[row, col]

            # access the correct sinogram from dictionary
            sino = sinograms_noisy[(row_val, ang)]

            # display sinogram as grayscale image, setting vmin/vmax to percentiles to prevent extremes and show structure
            ax.imshow(sino, cmap='gray', aspect='auto',vmin=np.percentile(sino, 1), vmax=np.percentile(sino, 99))

            # format title based on whether we are showing dose or angle range
            if use_angle_labels:
                ax.set_title(f'{row_val}° range, {ang} projections', fontsize=12)
            else:
                ax.set_title(f'$I_0=10^{{{int(np.log10(row_val))}}}$, {ang} angles', fontsize=12)

            ax.set_xlabel('Angle index', fontsize=10)
            ax.set_ylabel('Detector bin', fontsize=10)

    plt.tight_layout()

# ---------------------------------------------------------------------------
# Exercise 1.1c: sinogram reconstructions and metrics
# ---------------------------------------------------------------------------

def reconstruct_fbp(sinogram_noisy, theta, filter_name='ramp'):

    """Reconstruct a CT image using Filtered Backprojection (FBP)."""

    # RBP with iradon
    reconstruction = iradon(sinogram_noisy, theta, filter_name=filter_name)

    # clip to remove any negative values from noise
    reconstruction = np.clip(reconstruction, 0, None)

    return reconstruction


def reconstruct_gd(sinogram_noisy, theta, n_iterations=200, gamma=0.001):

    """Reconstruct a CT image using Gradient Descent."""

    # initialise the estimate as all zeros
    gd = np.zeros((sinogram_noisy.shape[0], sinogram_noisy.shape[0]))

    # create list to store residual norm 
    residuals = []

    for _ in range(n_iterations):

        # compute forward projection and calculate residual
        residual = radon(gd, theta) - sinogram_noisy

        # compute and store residual norm
        residuals.append(np.linalg.norm(residual))

        # backproject residual to get gradient
        update = -gamma*iradon(residual, theta, filter_name=None)

        # update image estimate
        gd = gd + update

    # clip to remove any negative values from noise
    gd = np.clip(gd, 0, None)

    return gd, residuals


def plot_reconstructions(reconstructions, row_values, col_values, title='Reconstructions', use_angle_labels=False):

    """Plot a 3x3 grid of reconstructed images across dose levels and angle counts."""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(title, fontsize=16, y=1.02)

    # loop over all rows and columns
    for row, r_val in enumerate(row_values):

        for col, c_val in enumerate(col_values):

            ax    = axes[row, col]

            # access the correct reconstruction from dictionary
            recon = reconstructions[(r_val, c_val)]

            # display reconstruction as grayscale image, setting vmin/vmax to constants to allow comparison across images
            ax.imshow(recon, cmap='gray', vmin=0, vmax=0.001)
            
            # format title based on whether we are showing dose or angle range
            if use_angle_labels:
                ax.set_title(f'{r_val}° range, {c_val} projections', fontsize=12)
            else:
                ax.set_title(f'$I_0=10^{{{int(np.log10(r_val))}}}$, {c_val} angles', fontsize=12)

            ax.axis('off')

    plt.tight_layout()


def metrics_tables(image, recons_fbp, recons_gd, row_values, angle_counts, title='Image Quality Metrics', use_angle_labels=False):

    """Compute RMSE, PSNR and SSIM for all conditions and plot as tables."""

    # set image dimensions
    h, w    = image.shape

    # create empty lists to store results
    fbp_results = []
    gd_results  = []

    for ang in angle_counts:
        for row_val in row_values:

            # format row label based on whether we are showing dose or angle range
            row_label = f'{row_val}°' if use_angle_labels else f'$10^{{{int(np.log10(row_val))}}}$'

            # access reconstructions for both algorithms
            for results_list, recons in [(fbp_results, recons_fbp), (gd_results, recons_gd)]:

                # crop to match image size
                recon = recons[(row_val, ang)][:h, :w]

                # compute metrics, setting data range to max pixel value for scaling
                rmse = float(np.sqrt(np.mean((image - recon) ** 2)))
                psnr = float(peak_signal_noise_ratio(image, recon, data_range=image.max()))
                ssim = float(structural_similarity(image, recon, data_range=image.max()))

                # add results to list
                results_list.append({
                    'row'   : row_label,
                    'angles' : str(ang),
                    'rmse'   : f'{rmse:.4f}',
                    'psnr'   : f'{psnr:.4f}',
                    'ssim'   : f'{ssim:.4f}',
                })

    # build tables, using pale colors to differentiate algorithms
    col_labels = ['Range', 'Projections', 'RMSE', 'PSNR (dB)', 'SSIM'] if use_angle_labels else ['Dose', 'Projections', 'RMSE', 'PSNR (dB)', 'SSIM']
    fbp_cells = [[r['row'], r['angles'], r['rmse'], r['psnr'], r['ssim']] for r in fbp_results]
    gd_cells  = [[r['row'], r['angles'], r['rmse'], r['psnr'], r['ssim']] for r in gd_results]
    fbp_colors = [['#d4e6f1'] * 5] * len(fbp_results)
    gd_colors  = [['#fde8d0'] * 5] * len(gd_results)

    # plot tables alongside eachother
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, len(fbp_results) * 0.5))
    fig.suptitle(title, fontsize=16, y=1.02)

    # add titles and hide axes for both tables
    for ax, cells, colors, subtitle in [(ax1, fbp_cells, fbp_colors, 'Filtered Backprojection'), (ax2, gd_cells,  gd_colors,  'Gradient Descent (200 iterations)')]:

        ax.axis('off')
        ax.set_title(subtitle, fontsize=12, pad=10)
    
        # create tables
        table = ax.table(cellText=cells, colLabels=col_labels, cellColours=colors, cellLoc='center', loc='center')

        # adjust font size and widths for readability
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width([0, 1, 2, 3, 4])
        table.scale(1, 1.8)

    plt.tight_layout()


def error_maps(image, recons_fbp, recons_gd, row_values, angle_counts, title='Absolute Error Maps', use_angle_labels=False):

    """Plot two grids of absolute error maps for FBP and GD reconstructions."""

    h, w = image.shape

    # two 3x3 grids side by side
    fig, axes = plt.subplots(3, 6, figsize=(20, 10))
    fig.suptitle(title, fontsize=16, y=1.02)

    # add subtitles for each grid
    axes[0, 1].annotate('Filtered Backprojection', xy=(0.5, 1.15), xycoords='axes fraction', fontsize=12, ha='center')
    axes[0, 4].annotate('Gradient Descent (200 iterations)', xy=(0.5, 1.15), xycoords='axes fraction', fontsize=12, ha='center')

    for row, ang in enumerate(angle_counts):
        for col_val, row_val in enumerate(row_values):

            # format subplot title based on whether we are showing dose or angle range
            subplot_title = f'{row_val}° range, {ang} angles' if use_angle_labels else f'$I_0=10^{{{int(np.log10(row_val))}}}$, {ang} angles'

            # iterate over both algorithms to plot error maps
            for offset, recons in enumerate([recons_fbp, recons_gd]):

                # set axes for both grids
                ax  = axes[row, col_val + offset * 3]

                # access and crop reconstructions
                recon = recons[(row_val, ang)][:h, :w]

                # compute and plot absolute error map
                err = np.abs(image - recon)
                ax.imshow(err, vmin=0, vmax=image.max() * 0.3)

                ax.set_title(subplot_title, fontsize=10)
                ax.axis('off')

    plt.tight_layout()

    # add a vertical line to separate FBP and GD grids
    line = plt.Line2D([0.5, 0.5], [0.01, 0.97], transform=fig.transFigure, color='black', linewidth=1.5)
    fig.add_artist(line)

# ---------------------------------------------------------------------------
# Exercise 1.2a: limited angle sinograsm
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Exercise 1.2b: limited angle reconstructions and metrics
# ---------------------------------------------------------------------------

# I added use_angle_labels=False to all the functions above to make them reusable for both dose and angle range comparisons
