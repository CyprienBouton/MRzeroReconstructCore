"""Tests of the shared k-space utilities on synthetic data.

Identical in MRzeroReconstructCore and MRTwixReconstructCore except for the
package import below. Run with ``python -m pytest tests``.
"""

import torch

from MRzeroReconstructCoreLib import reco_utils
from phantom import make_phantom, relative_error


def test_centered_fft_roundtrip_is_unitary():
    data = torch.randn(8, 6, 5, 2, dtype=torch.complex64)
    kspace = reco_utils.centered_fft(data, dims=(0, 1, 2))
    assert torch.allclose(reco_utils.centered_ifft(kspace, dims=(0, 1, 2)), data, atol=1e-5)
    assert abs(float(torch.linalg.norm(kspace) / torch.linalg.norm(data)) - 1) < 1e-5


def test_centered_ifft_puts_dc_at_n_over_2():
    kspace = torch.zeros(8, 7, 1, 1, dtype=torch.complex64)
    kspace[4, 3, 0, 0] = 1  # k = 0 at n // 2 -> constant image
    image = reco_utils.centered_ifft(kspace, dims=(0, 1, 2))
    assert torch.allclose(image.imag, torch.zeros_like(image.imag), atol=1e-6)
    assert float(image.real.std()) < 1e-6


def test_centering_padding():
    padding, center = reco_utils.get_centering_padding((64, 40, 1, 8), (32, 10, 0))
    assert padding[0] == (0, 0) and center[0] == 32          # already centred
    assert padding[2] == (0, 0) and center[2] == 0           # single partition
    before, after = padding[1]
    assert (40 + before + after) // 2 == center[1] == 30     # asymmetric: padded, even size
    assert padding[3] == (0, 0)


def test_center_kspace_keeps_every_sample():
    kspace = torch.randn(10, 12, 1, 2, dtype=torch.complex64)
    centered, center = reco_utils.center_kspace(kspace, (5, 3, 0))
    assert center == (5, 9, 0)
    assert torch.equal(centered[:, 6:18], kspace)
    assert int((centered != 0).sum()) == int((kspace != 0).sum())  # only zeros added


def test_acs_and_acceleration_detection():
    _, kspace = make_phantom()
    undersampled = kspace.clone()
    undersampled[:, 1::2] = 0
    undersampled[:, 12:20] = kspace[:, 12:20]
    bounds = reco_utils.get_acs_bounds(undersampled)
    # line 20 is also acquired (even), line 21 is not
    assert bounds == ((0, 32), (12, 21), (0, 4))
    assert reco_utils.estimate_acceleration(undersampled) == (2, 1)


def test_grappa_recovers_undersampled_kspace():
    images, kspace = make_phantom()
    undersampled = kspace.clone()
    undersampled[:, 1::2] = 0
    undersampled[:, 10:22] = kspace[:, 10:22]
    acs = reco_utils.get_acs_from_kspace(undersampled)
    filled = reco_utils.grappa_reconstruct(undersampled, acs, kernel_size=(4, 1, 5),
                                           cuda=False)
    assert filled.shape == kspace.shape
    error_grappa = relative_error(filled, kspace)
    error_zero = relative_error(undersampled, kspace)
    assert error_grappa < 0.5 * error_zero, (error_grappa, error_zero)


def test_pocs_improves_partial_fourier():
    images, kspace = make_phantom(shape=(32, 32, 1))
    partial = kspace.clone()
    partial[:, :10] = 0  # 22/32 lines acquired, k = 0 at index 16
    recon = reco_utils.pocs_partial_fourier(partial, dim_pf=1, number_of_iterations=20,
                                            gaussian_std=4.0, center_index=16)
    assert torch.allclose(recon[:, 10:], partial[:, 10:], atol=1e-4)
    error_pocs = relative_error(recon, kspace)
    error_zero = relative_error(partial, kspace)
    assert error_pocs < error_zero, (error_pocs, error_zero)


def test_sitk_conversion():
    _, kspace = make_phantom(shape=(16, 12, 3))
    image = reco_utils.kspace_to_sitk(kspace)
    assert image.GetSize() == (16, 12, 3)
    volume = reco_utils.rss(kspace).permute(2, 1, 0)
    image = reco_utils.make_sitk_image(volume, spacing=(2, 1, 3), center_volume=True)
    assert image.GetSpacing() == (2.0, 1.0, 3.0)
    assert abs(image.GetOrigin()[0] + 15.0) < 1e-9  # -(16 * 2 / 2 - 2 / 2)
