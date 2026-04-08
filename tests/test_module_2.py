"""Tests for medical_imaging.module_2 (MRI denoising)."""

import numpy as np
import pytest
import medical_imaging.module_2 as mri


# ---------------------------------------------------------------------------
# Fixtures: reusable synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_kspace(tmp_path):
    
    """Create and save a synthetic complex k-space array with 6 coils."""

    n_coils, h, w = 6, 64, 64

    # complex Gaussian noise for k-space
    kspace = np.random.randn(n_coils, h, w) + 1j * np.random.randn(n_coils, h, w)

    # save to temporary path
    path = tmp_path / "kspace.npy"
    np.save(str(path), kspace)

    return str(path), kspace


@pytest.fixture
def coil_images():

    """Create synthetic complex coil images with shape (6, 64, 64)."""

    n_coils, h, w = 6, 64, 64
    images = np.random.randn(n_coils, h, w) + 1j * np.random.randn(n_coils, h, w)
    return images


@pytest.fixture
def magnitude_images(coil_images):

    """Real-valued magnitude images derived from the complex coil images."""

    return np.abs(coil_images)


@pytest.fixture
def rsos_image(coil_images):

    """Root-sum-of-squares combined image from the coil images."""

    return mri.root_sum_of_squares(coil_images)


# ---------------------------------------------------------------------------
# Tests for Exercise 2.1a: load_kspace
# ---------------------------------------------------------------------------

def test_load_kspace_shape(synthetic_kspace):

    """load_kspace should return the array with the correct shape."""

    # load k-space from the synthetic fixture
    path, original = synthetic_kspace

    # load using the function
    kspace, _ = mri.load_kspace(path)

    # check shape matches the original
    np.testing.assert_array_equal(kspace, original)


def test_load_kspace_coil_axis(synthetic_kspace):

    """load_kspace should correctly identify the coil axis."""

    path, _ = synthetic_kspace

    # load k-space and coil axis
    _, coil_axis = mri.load_kspace(path)

    # check if coil axis is 0
    assert coil_axis == 0

# ---------------------------------------------------------------------------
# Tests for Exercise 2.1c: kspace_to_image
# ---------------------------------------------------------------------------

def test_kspace_to_image_shape(synthetic_kspace):

    """kspace_to_image should preserve spatial dimensions and number of coils."""


    path, _ = synthetic_kspace

    # load k-space and coil axis
    kspace, coil_axis = mri.load_kspace(path)

    # convert to image space
    images = mri.kspace_to_image(kspace, coil_axis)

    # check the output shape
    assert images.shape == (6, 64, 64)


def test_kspace_to_image_invertible(synthetic_kspace):

    """Applying FFT to the image-space result should recover the original k-space."""


    path, original = synthetic_kspace

    # load k-space and coil axis
    kspace, coil_axis = mri.load_kspace(path)

    # convert to image space
    images = mri.kspace_to_image(kspace, coil_axis)

    # move coil axis to front
    kspace_front = np.moveaxis(original, coil_axis, 0)

    # forward FFT should recover k-space
    recovered = np.fft.fft2(images)
    np.testing.assert_allclose(recovered, kspace_front, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests for Exercise 2.1e: root_sum_of_squares
# ---------------------------------------------------------------------------

def test_rsos_shape(coil_images):

    """rSoS output should be 2D with the spatial dimensions of the input."""

    # compute rSoS image
    rsos = mri.root_sum_of_squares(coil_images)

    assert rsos.shape == (64, 64)


def test_rsos_nonnegative(coil_images):

    """rSoS output should be non-negative everywhere."""

    # compute rSoS image
    rsos = mri.root_sum_of_squares(coil_images)

    assert np.all(rsos >= 0)

# ---------------------------------------------------------------------------
# Tests for Exercise 2.2a: denoising filters
# ---------------------------------------------------------------------------

def test_gaussian_filtering_shape(magnitude_images):

    """Gaussian filtering should preserve array shape."""

    # apply Gaussian filtering
    denoised = mri.gaussian_filtering(magnitude_images, sigma=1)

    # check the output shape matches the input
    assert denoised.shape == magnitude_images.shape


def test_bilateral_filtering_shape(magnitude_images):
    
    """Bilateral filtering should preserve array shape."""

    # apply bilateral filtering
    denoised = mri.bilateral_filtering(magnitude_images, d=5, sigma_color=0.2, sigma_space=5)

    assert denoised.shape == magnitude_images.shape


def test_wavelet_filtering_shape(magnitude_images):

    """Wavelet filtering should preserve array shape."""

    # apply wavelet filtering
    denoised = mri.wavelet_filtering(magnitude_images)

    assert denoised.shape == magnitude_images.shape


# ---------------------------------------------------------------------------
# Tests for Exercise 2.2b: Butterworth filter
# ---------------------------------------------------------------------------

def test_butterworth_filter_shape():

    """Butterworth filter should match the requested shape."""

    # create a Butterworth filter with specified shape
    H = mri.butterworth_lowpass_filter((64, 64), D0=30, n=2)

    assert H.shape == (64, 64)


def test_apply_butterworth_shape(synthetic_kspace):

    """Filtered image should have the same spatial dimensions as the k-space."""

    path, _ = synthetic_kspace

    # load k-space and coil axis
    kspace, coil_axis = mri.load_kspace(path)

    # apply Butterworth filter to the first coil
    filtered = mri.apply_butterworth(kspace, coil_axis, coil_idx=0, D0=20, n=2)

    assert filtered.shape == (64, 64)