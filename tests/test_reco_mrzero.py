"""Tests of RecoMRzero gridding on a fake sequence (MRzeroCore not required)."""

import torch

from MRzeroReconstructCoreLib import RecoMRzero, reco_utils
from phantom import make_phantom, relative_error

DK = 1 / 0.2  # k-space step [1/m] -> 200 mm field of view


class FakeSequence(list):
    """Minimal stand-in for MRzeroCore.Sequence: only get_kspace() is used."""

    def __init__(self, kspace_coords):
        super().__init__()
        self._coords = kspace_coords

    def get_kspace(self):
        return self._coords


def _cartesian_coords(iy_range, nx=32, nz=4):
    """ADC sample coordinates [n, 4] in 1/m, readout fastest, k=0 at n // 2."""
    ix, iy, iz = torch.meshgrid(torch.arange(nx), torch.tensor(list(iy_range)),
                                torch.arange(nz), indexing="ij")
    coords = torch.stack([(ix - nx // 2) * DK, (iy - 16) * DK, (iz - nz // 2) * DK,
                          torch.zeros_like(ix, dtype=torch.float32)], dim=-1)
    return coords.reshape(-1, 4).float(), (ix.reshape(-1), iy.reshape(-1), iz.reshape(-1))


def test_grid_and_fft_reproduce_phantom():
    images, kspace = make_phantom()
    coords, (ix, iy, iz) = _cartesian_coords(range(32))
    signal = kspace[ix, iy, iz]                      # [n_samples, ncoils]
    reco = RecoMRzero(FakeSequence(coords))
    grid, *_ = reco.cartesian_kspace_grid(signal)
    assert torch.allclose(grid, kspace)
    centered, center = reco.center_kspace(grid)
    assert center == (16, 16, 2)
    assert relative_error(reco.fft_reconstruct(centered), images) < 1e-5
    spacing = reco.get_spacing(centered.shape)
    assert all(abs(s - expected) < 1e-3 for s, expected in zip(spacing, (6.25, 6.25, 50.0)))


def test_partial_fourier_grid_is_padded_not_cropped():
    _, kspace = make_phantom()
    coords, (ix, iy, iz) = _cartesian_coords(range(10, 32))   # lines 10..31 acquired
    reco = RecoMRzero(FakeSequence(coords))
    grid, *_ = reco.cartesian_kspace_grid(kspace[ix, iy, iz])
    assert grid.shape[1] == 22 and reco.kspace_center[1] == 6
    centered, center = reco.center_kspace(grid)
    assert centered.shape[1] == 32 and center[1] == 16
    assert torch.allclose(centered[:, 10:], kspace[:, 10:])   # no sample lost


def test_navigator_samples_are_dropped_from_signal():
    _, kspace = make_phantom()
    coords, (ix, iy, iz) = _cartesian_coords(range(32))
    signal = kspace[ix, iy, iz]
    extra = torch.ones(5, signal.shape[1], dtype=signal.dtype)
    reco = RecoMRzero(FakeSequence(coords))
    reco.signal_keep_mask = torch.cat([torch.ones(len(signal), dtype=torch.bool),
                                       torch.zeros(5, dtype=torch.bool)])
    grid, *_ = reco.cartesian_kspace_grid(torch.cat([signal, extra]))
    assert torch.allclose(grid, kspace)


def test_grappa_through_reco_class():
    images, kspace = make_phantom()
    lines = [y for y in range(32) if y % 2 == 0 or 10 <= y < 22]
    coords, (ix, iy, iz) = _cartesian_coords(lines)
    reco = RecoMRzero(FakeSequence(coords))
    grid, *_ = reco.cartesian_kspace_grid(kspace[ix, iy, iz])
    centered, center = reco.center_kspace(grid)
    acs = reco.get_acs_from_kspace(centered, center)
    filled = reco.grappa_reconstruct(centered, acs, kernel_size=(4, 1, 5), cuda=False)
    error = relative_error(reco_utils.rss(reco.fft_reconstruct(filled)),
                           reco_utils.rss(images))
    assert error < 0.1, error


def test_zero_filled_cs_rejects_irregular_recovery_line(monkeypatch):
    # Odd readout and a single partition also exercise Haar boundary handling.
    nx, ny, nz = 9, 32, 1
    coords, _ = _cartesian_coords(range(ny), nx=nx, nz=nz)
    kspace = torch.zeros((nx, ny, nz, 2), dtype=torch.complex64)
    kspace[nx // 2, ny // 2, 0] = torch.tensor([1, 1j])
    kspace[:, 2, 0] = 100
    durations = torch.ones((ny, nz))
    durations[2] = 2

    reco = RecoMRzero(FakeSequence(coords))
    grid, *_ = reco.cartesian_kspace_grid(kspace.reshape(-1, 2))
    monkeypatch.setattr(reco, "get_inter_scan_duration", lambda: durations)

    image = reco.zero_filled_cs(grid, regularization=0)
    regularized = reco.zero_filled_cs(grid, regularization=0.1)
    expected = torch.full_like(image.abs(), (2 / (nx * ny * nz)) ** 0.5)
    assert image.shape == (nx, ny, nz)
    torch.testing.assert_close(image.abs(), expected, atol=1e-6, rtol=1e-5)
    assert 0 < regularized.norm() < image.norm()


def test_radial_nufft_through_slicer_entry_point(tmp_path, monkeypatch):
    import importlib
    import numpy as np
    import SimpleITK as sitk
    from torchkbnufft import KbNufft
    from MRzeroReconstructCoreLib import MRzeroReconstructLogic

    angles = torch.arange(48) * torch.pi / 48
    radii = torch.linspace(-0.49, 0.49, 48)
    trajectory = torch.stack((torch.cos(angles)[:, None] * radii,
                              torch.sin(angles)[:, None] * radii)).reshape(2, -1)
    shape = (32, 32, 1)
    fov = (0.2, 0.2, 0.005)
    coords = torch.zeros(trajectory.shape[1], 4)
    coords[:, :2] = trajectory.T * 32 / 0.2
    image = torch.zeros(1, 1, 32, 32, dtype=torch.complex64)
    image[:, :, 12:20, 12:20] = 1
    signal = KbNufft(im_size=(32, 32))(image, 2 * torch.pi * trajectory, norm="ortho")
    path = tmp_path / "signal.npy"
    np.save(path, signal[0].T.numpy())
    module = importlib.import_module("MRzeroReconstructCoreLib.MRzeroReconstructCore")
    monkeypatch.setattr(module, "load_sequence", lambda _: FakeSequence(coords))

    def fail_gridding(*args, **kwargs):
        raise AssertionError("NUFFT must bypass Cartesian gridding")

    monkeypatch.setattr(RecoMRzero, "cartesian_kspace_grid", fail_gridding)
    volumes = MRzeroReconstructLogic().runReconstruction(
        path, "radial.seq", "NUFFT", doRemoveNavigator=False,
        nufft_shape=shape, nufft_fov=fov)
    reconstructed = torch.tensor(sitk.GetArrayFromImage(volumes[0])[0])
    target = image[0, 0].abs().T
    reconstructed *= (target * reconstructed).sum() / reconstructed.square().sum()
    assert (target - reconstructed).norm() / target.norm() < 0.2
    assert all(v.GetSize() == shape for v in volumes)
    assert volumes[0].GetSpacing() == (6.25, 6.25, 5.0)


def test_cartesian_grid_rejects_huge_inferred_grid():
    import pytest
    coords = torch.tensor([[0., 0., 0., 0.], [0.05, 0.05, 0.05, 0.],
                           [1000., 1000., 1000., 0.]])
    reco = RecoMRzero(FakeSequence(coords))
    with pytest.raises(ValueError, match="select NUFFT"):
        reco.cartesian_kspace_grid(torch.ones(3, 1, dtype=torch.complex64))


def test_automatic_radial_fov_and_manual_equivalence():
    from types import SimpleNamespace
    import pytest

    angles = torch.arange(16) * torch.pi / 16
    radii = torch.arange(-16, 16) * 5.0  # 5 cycles/m -> 200 mm encoded FOV
    coords = torch.zeros(16, 32, 4)
    coords[..., 0] = torch.cos(angles)[:, None] * radii
    coords[..., 1] = torch.sin(angles)[:, None] * radii
    seq = FakeSequence(coords.reshape(-1, 4))
    seq.extend(SimpleNamespace(adc_usage=torch.ones(32)) for _ in angles)
    reco = RecoMRzero(seq)
    assert reco.estimate_radial_fov((32, 32, 1)) == pytest.approx((0.2, 0.2, 0.005))
    signal = torch.ones(16 * 32, 1, dtype=torch.complex64)
    automatic = reco.nufft_reconstruct(signal, (32, 32, 1))
    manual = reco.nufft_reconstruct(signal, (32, 32, 1), (0.2, 0.2, 0.005))
    # Float trajectory roundoff can change NUFFT interpolation table bins.
    assert (automatic - manual).norm() / manual.norm() < 1e-3
    # Variable spacing requires an explicit field of view.
    seq._coords[4, 0] += 1
    with pytest.raises(ValueError, match="spacing varies"):
        reco.estimate_radial_fov()
