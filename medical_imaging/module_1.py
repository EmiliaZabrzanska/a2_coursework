import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import radon, iradon
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


# ---------------------------------------------------------------------------
# Exercise 1.1a: loading image
# ---------------------------------------------------------------------------

def load_image(path):

    """
    Load a CT image and normalise pixel values to attenuation units [0, 0.001].
 
    Parameters
    ----------
    path : str
        File path to the image.
 
    Returns
    -------
    image : np.ndarray
        2D array of attenuation coefficients in the range [0, 0.001].
    """

    raw = imread(path)

    # convert to 2D
    if raw.ndim == 3:

        # slice array to keep first 3 channels and drop alpha, then convert to grayscale and normalise to [0, 1]
        raw = rgb2gray(raw[:, :, :3])  

    else:

        # if already 2D, just normalise to [0, 1]
        raw = raw.astype(np.float64) / raw.max()

    # rescale to attenuation coefficients (0, 0.001)
    return raw / 1000.0

def visualise_image(image, title='Image'):

    """
    Display a 2D greyscale image with a colourbar labelled in attenuation units.
 
    Parameters
    ----------
    image : np.ndarray
        2D array of attenuation coefficients.

    title : str, optional
        Figure title. Default is 'Image'.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    # display 2D array as an image with a grayscale colormap, setting vmin/vmax for consistent scaling
    im = ax.imshow(image, cmap='gray', vmin=0, vmax=image.max())

    ax.set_title(title)

    # hide axes for cleaner look
    ax.axis('off')                                                              

    # add colorbar to the right, controlling size and distance from image
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)

    # add space from bar with padding
    cbar.set_label('Attenuation coefficient $\\mu$ [cm$^{-1}$]', labelpad=15)   
    
    plt.tight_layout()

# ---------------------------------------------------------------------------
# Exercise 1.1b: simulating noise and sinograms
# ---------------------------------------------------------------------------

def simulate_sinogram(image, theta, I0):

    """
    Forward-project an image to obtain a sinogram and simulate noisy acquisition.
 
    Applies the Radon transform to get a clean absorption sinogram, converts to Beer-Lambert transmission form, 
    adds Poisson noise and Gaussian noise, then converts back to an absorption sinogram.
 
    Parameters
    ----------
    image : np.ndarray
        2D attenuation image to project.

    theta : np.ndarray
        1D array of projection angles in degrees.

    I0 : float
        Incident photon intensity (higher = lower noise).
 
    Returns
    -------
    sinogram_clean : np.ndarray
        Noise-free absorption sinogram from the Radon transform.

    sinogram_noisy : np.ndarray
        Noisy absorption sinogram after Poisson and Gaussian corruption.
    """

    # forward project image using radon transfrom to get a clean absorption sinogram
    sinogram_clean = radon(image, theta)

    # convert to Beer-Lambert transmisson form
    transmission   = I0 * np.exp(-sinogram_clean)

    # add poisson noise, np.clip to ensure there are no negative values and convert back to float
    transmission   = np.random.poisson(np.clip(transmission, 0, None)).astype(np.float64)

    # add gaussian noise
    transmission   = transmission + np.random.normal(0, 0.05, transmission.shape)

    # convert back to absorption sinogram, np,clip to avoid negative values and log(0)
    sinogram_noisy = np.clip(-np.log(np.clip(transmission, 1e-10, None) / I0), 0, None)

    return sinogram_clean, sinogram_noisy


def plot_sinograms(sinograms_noisy, row_values, angle_counts, title='Sinograms', use_angle_labels=False):
    
    """
    Plot a 3x3 grid of noisy sinograms across varying acquisition parameters.
 
    Each row corresponds to a different dose level or angular range, and each column to a different number of projection angles.
 
    Parameters
    ----------
    sinograms_noisy : dict

    row_values : list

    angle_counts : list

    title : str, optional

    use_angle_labels : bool, optional
    """

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(title, fontsize=16, y=1.02)

    # loop over all row values and angle counts
    for row, row_val in enumerate(row_values):

        for col, ang in enumerate(angle_counts):

            ax   = axes[row, col]

            # access correct sinogram from dictionary
            sino = sinograms_noisy[(row_val, ang)]

            # display grayscale image, setting vmin/vmax to percentiles to prevent extremes and show structure
            ax.imshow(sino, cmap='gray', aspect='auto',vmin=np.percentile(sino, 1), vmax=np.percentile(sino, 99))

            # format title based on dose or angle range
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

    """
    Reconstruct a CT image from a sinogram using Filtered Back-Projection.
 
    Parameters
    ----------
    sinogram_noisy : np.ndarray

    theta : np.ndarray

    filter_name : str, optional
 
    Returns
    -------
    reconstruction : np.ndarray
        2D reconstructed image, clipped to non-negative values.
    """

    # RBP with iradon
    reconstruction = iradon(sinogram_noisy, theta, filter_name=filter_name)

    # clip to remove any negative values from noise
    reconstruction = np.clip(reconstruction, 0, None)

    return reconstruction


def reconstruct_gd(sinogram_noisy, theta, n_iterations=200, gamma=0.001):

    """
    Reconstruct a CT image from a sinogram using gradient descent (SIRT).
 
    Parameters
    ----------
    sinogram_noisy : np.ndarray

    theta : np.ndarray

    n_iterations : int, optional

    gamma : float, optional
 
    Returns
    -------
    gd : np.ndarray
        2D reconstructed image, clipped to non-negative values.

    residuals : list of float
        Residual norm at each iteration.
    """

    # initialise estimate as all zeros
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

    """
    Plot a 3x3 grid of reconstructed images across varying acquisition parameters.
 
    Parameters
    ----------
    reconstructions : dict

    row_values : list

    col_values : list

    title : str, optional

    use_angle_labels : bool, optional
    """
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(title, fontsize=16, y=1.02)

    # loop over all rows and columns
    for row, r_val in enumerate(row_values):

        for col, c_val in enumerate(col_values):

            ax    = axes[row, col]

            # access correct reconstruction from dictionary
            recon = reconstructions[(r_val, c_val)]

            # display as grayscale image, setting vmin/vmax to constants to allow comparison across images
            ax.imshow(recon, cmap='gray', vmin=0, vmax=0.001)
            
            # format title based on dose or angle range
            if use_angle_labels:
                ax.set_title(f'{r_val}° range, {c_val} projections', fontsize=12)
            else:
                ax.set_title(f'$I_0=10^{{{int(np.log10(r_val))}}}$, {c_val} angles', fontsize=12)

            ax.axis('off')

    plt.tight_layout()


def metrics_tables(image, recons_fbp, recons_gd, row_values, angle_counts, title='Image Quality Metrics', use_angle_labels=False):

    """
    Compute RMSE, PSNR, and SSIM for FBP and GD reconstructions and display as tables.
 
    Generates two side-by-side matplotlib tables (one per algorithm) showing
    image quality metrics for every combination of acquisition parameters.
 
    Parameters
    ----------
    image : np.ndarray

    recons_fbp : dict

    recons_gd : dict

    row_values : list

    angle_counts : list

    title : str, optional

    use_angle_labels : bool, optional
    """

    # set image dimensions
    h, w    = image.shape

    # create empty lists to store results
    fbp_results = []
    gd_results  = []

    for ang in angle_counts:
        for row_val in row_values:

            # format row label based on dose or angle range
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
                    'rmse'   : f'{rmse:.5f}',
                    'psnr'   : f'{psnr:.2f}',
                    'ssim'   : f'{ssim:.5f}',
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

    """
    Plot absolute error maps for FBP and GD reconstructions side by side.
 
    Parameters
    ----------
    image : np.ndarray

    recons_fbp : dict

    recons_gd : dict

    row_values : list

    angle_counts : list

    title : str, optional

    use_angle_labels : bool, optional
    """

    # set image dimensions
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

# ---------------------------------------------------------------------------
# Exercise 1.3a: FBP filters
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Exercise 1.3b: Filter comparison
# ---------------------------------------------------------------------------

def plot_comparisons(image, recons, title='Algorithm Comparison'):

    """
    Plot the ground-truth image alongside reconstructions with quality metrics.
 
    Parameters
    ----------
    image : np.ndarray

    recons : dict

    title : str, optional
    """

    # number of recons determines subplots
    n = len(recons)

    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 5))
    fig.suptitle(title, fontsize=16, y=1.02)

    # # display original image with a grayscale colormap, setting vmin/vmax for consistent scaling
    axes[0].imshow(image, cmap='gray', vmin=0, vmax=image.max())
    axes[0].set_title('Ground Truth', fontsize=10)
    axes[0].axis('off')                                

    # plot each reconstruction in remaining subplots
    for ax, (name, recon) in zip(axes[1:], recons.items()):

        # set image dimensions
        h, w = image.shape

        # compute metrics, setting data range to max pixel value for scaling
        rmse = float(np.sqrt(np.mean((image - recon[:h, :w]) ** 2)))
        psnr = float(peak_signal_noise_ratio(image, recon[:h, :w], data_range=image.max()))
        ssim = float(structural_similarity(image, recon[:h, :w], data_range=image.max()))

        # display reconstruction with a grayscale colormap, setting vmin/vmax for consistent scaling, 
        ax.imshow(recon[:h, :w], cmap='gray', vmin=0, vmax=image.max())

         # format title with filter name bold and metrics on one line below
        ax.set_title(f'{name}\nRMSE={rmse:.5f} | PSNR={psnr:.2f} dB | SSIM={ssim:.5f}', fontsize=10)

        ax.axis('off')

    plt.tight_layout()

# ---------------------------------------------------------------------------
# Exercise 1.3c: OS-SART comparison
# ---------------------------------------------------------------------------

def reconstruct_os(sinogram_noisy, theta, n_subsets=10, n_iterations=20, gamma=0.001):

    """
    Reconstruct a CT image from a sinogram using Ordered Subsets SART (OS-SART).
 
    Parameters
    ----------
    sinogram_noisy : np.ndarray

    theta : np.ndarray

    n_subsets : int, optional

    n_iterations : int, optional

    gamma : float, optional
 
    Returns
    -------
    os : np.ndarray
        2D reconstructed image, clipped to non-negative values.

    residuals : list of float
        Residual norm at the end of each outer iteration.
    """

    # initialise the estimate as all zeros
    os = np.zeros((sinogram_noisy.shape[0], sinogram_noisy.shape[0]))

    # create lists to store residual norm
    residuals = []
    n_angles  = len(theta)

    # split angles into subsets, spreading out the angles in each set to cover full range
    subset_indices = [np.arange(s, n_angles, n_subsets) for s in range(n_subsets)]

    for _ in range(n_iterations):

        for idx in subset_indices:

            # extract subset sinogram and angles
            sinogram_subset  = sinogram_noisy[:, idx]
            theta_subset = theta[idx]

            # compute forward projection and calculate residual for subset
            residual = radon(os, theta=theta_subset) - sinogram_subset

            # backproject residual to get gradient
            update = - gamma * iradon(residual, theta=theta_subset, filter_name=None)

            # update image estimate
            os = os + update

        # compute and store residual norm at end of each iteration
        full_residual = radon(os, theta=theta) - sinogram_noisy
        residuals.append(np.linalg.norm(full_residual))

    # clip to remove any negative values from noise
    os = np.clip(os, 0, None)

    return os, residuals

# reused plot_comparisons function from above to compare SIRT and OS-SART reconstructions

def plot_convergence(res_sirt, res_ossart, title='SIRT vs OS-SART Convergence'):

    """
    Plot convergence curves comparing SIRT and OS-SART on a shared axis.
 
    Parameters
    ----------
    res_sirt : list of float
        Residual norms from SIRT at each iteration.

    res_ossart : list of float
        Residual norms from OS-SART at each outer iteration.

    title : str, optional
    """

    # added to help format tick labels
    from matplotlib.ticker import ScalarFormatter

    _, ax = plt.subplots(figsize=(9, 6))

    # plot SIRT residuals
    ax.plot(res_sirt, label='SIRT', color='C0', linewidth=1.5)

    # create x positions so we can plot on same axis as SIRT
    ossart_x = np.linspace(0, len(res_sirt), len(res_ossart))

    # plot OS-SART residuals
    ax.plot(ossart_x, res_ossart, label='OS-SART', color='C1', linewidth=1.5)

    # set labels and title
    ax.set_xlabel('Equivalent full iterations')
    ax.set_ylabel('Residual norm $\\|Ax^k - b\\|$')
    ax.set_title(title)
    
    # add legend
    ax.legend()

    # Add a faint grid for better readability
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()

# ---------------------------------------------------------------------------
# End of Module 1 code
# ---------------------------------------------------------------------------