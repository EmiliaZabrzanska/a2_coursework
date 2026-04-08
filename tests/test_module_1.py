"""Tests for medical_imaging.module_1 (CT reconstruction)."""

import numpy as np
import pytest
import medical_imaging.module_1 as ct


# ---------------------------------------------------------------------------
# Fixtures: reusable synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def phantom():

    """Create a simple 64x64 phantom image with known attenuation values."""

    image = np.zeros((64, 64), dtype=np.float64)

    # set x and y coordinates
    y, x = np.ogrid[-32:32, -32:32]

    # disc with attenuation 0.0008 and radius 15 pixels
    mask = x**2 + y**2 <= 15**2
    image[mask] = 0.0008

    return image


@pytest.fixture
def theta_full():

    """Full 180-degree angular range with 90 projections."""

    return np.linspace(0, 180, 90, endpoint=False)


@pytest.fixture
def clean_sinogram(phantom, theta_full):

    """Clean sinogram from the phantom (no noise)."""

    from skimage.transform import radon
    return radon(phantom, theta_full)


# ---------------------------------------------------------------------------
# Tests for Exercise 1.1a: load_image
# ---------------------------------------------------------------------------

def test_load_image_creates_greyscale(tmp_path):

    """load_image should return a 2D array scaled to [0, 0.001]."""

    from skimage.io import imsave

    # select random RBG values
    rgb = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

    # save to temporary path
    path = tmp_path / "test.png"
    imsave(str(path), rgb)

    # load the image 
    result = ct.load_image(str(path))

    # check the output is 2D and scaled correctly
    assert result.ndim == 2
    assert result.max() <= 0.001 + 1e-10
    assert result.min() >= 0.0


def test_load_image_greyscale_input(tmp_path):

    """load_image should handle a greyscale image without error."""

    from skimage.io import imsave

    # create a random 32x32 greyscale image
    grey = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    path = tmp_path / "test_grey.png"
    imsave(str(path), grey)

    # load the image 
    result = ct.load_image(str(path))

    # check the output is 2D and scaled correctly
    assert result.ndim == 2
    assert result.max() <= 0.001 + 1e-10


# ---------------------------------------------------------------------------
# Tests for Exercise 1.1b: simulate_sinogram
# ---------------------------------------------------------------------------

def test_simulate_sinogram_shapes(phantom, theta_full):

    """Both sinograms should have the same shape."""

    # simulate both clean and noisy sinograms
    clean, noisy = ct.simulate_sinogram(phantom, theta_full, I0=1e5)

    # check they have the same shape and correct number of angles
    assert clean.shape == noisy.shape
    assert clean.shape[1] == len(theta_full)


def test_simulate_sinogram_clean_nonnegative(phantom, theta_full):

    """Clean sinogram should be non-negative (absorption values)."""

    # simulate clean sinogram
    clean, _ = ct.simulate_sinogram(phantom, theta_full, I0=1e5)

    # check all values are non-negative
    assert np.all(clean >= 0)


def test_simulate_sinogram_noisy_differs(phantom, theta_full):

    """Noisy sinogram should differ from clean due to added noise."""

    # simulate both clean and noisy sinograms
    clean, noisy = ct.simulate_sinogram(phantom, theta_full, I0=1e5)

    # check they differ
    assert not np.allclose(clean, noisy)


# ---------------------------------------------------------------------------
# Tests for Exercise 1.1c: reconstruct_fbp
# ---------------------------------------------------------------------------

def test_reconstruct_fbp_nonnegative(clean_sinogram, theta_full):

    """FBP reconstruction should be non-negative after clipping."""

    # reconstruct image from clean sinogram
    recon = ct.reconstruct_fbp(clean_sinogram, theta_full)

    # check all values are non-negative
    assert np.all(recon >= 0)


def test_reconstruct_fbp_shape(clean_sinogram, theta_full):

    """FBP output should be a square image."""

    # reconstruct image from clean sinogram
    recon = ct.reconstruct_fbp(clean_sinogram, theta_full)

    # check output is 2D and square
    assert recon.ndim == 2
    assert recon.shape[0] == recon.shape[1]

# ---------------------------------------------------------------------------
# Tests for Exercise 1.1c: reconstruct_gd
# ---------------------------------------------------------------------------

def test_reconstruct_gd_nonnegative(clean_sinogram, theta_full):

    """GD reconstruction should be non-negative after clipping."""

    # reconstruct image from clean sinogram
    recon, _ = ct.reconstruct_gd(clean_sinogram, theta_full, n_iterations=10, gamma=0.001)

    # check all values are non-negative
    assert np.all(recon >= 0)

# ---------------------------------------------------------------------------
# Tests for Exercise 1.3c: reconstruct_os (OS-SART)
# ---------------------------------------------------------------------------

def test_reconstruct_os_nonnegative(clean_sinogram, theta_full):

    """OS-SART reconstruction should be non-negative after clipping."""

    # reconstruct image from clean sinogram
    recon, _ = ct.reconstruct_os(clean_sinogram, theta_full, n_subsets=5, n_iterations=5, gamma=0.001)

    # check all values are non-negative
    assert np.all(recon >= 0)
