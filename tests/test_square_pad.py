import pytest
import torch
from model.utils.nn import square_pad, pad_inputs

def test_3d_input_padding():
    xs = [torch.ones(1, 2, 2), torch.ones(1, 3, 3)]
    out = square_pad(xs)  # replace with your method
    assert out.shape == (2, 3, 3, 1)
    assert torch.all(out[0, :2, :2, 0] == 1)
    assert torch.count_nonzero(out[0, 2:, :, :]) == 0

def test_4d_input_padding():
    xs = [torch.ones(1, 2, 2, 4), torch.ones(1, 3, 3, 4)]
    out = square_pad(xs)
    assert out.shape == (2, 3, 3, 4)
    assert torch.all(out[0, :2, :2, :] == 1)
    assert torch.count_nonzero(out[0, 2:, :, :]) == 0

def test_single_tensor_no_padding():
    x = torch.randn(1, 4, 4, 2)
    out = square_pad([x])
    assert out.shape == (1, 4, 4, 2)
    assert torch.allclose(out[0], x[0])

def test_device_dtype_preserved():
    x = torch.zeros(1, 2, 2, 1, device="cpu", dtype=torch.float64)
    out = square_pad([x])
    assert out.device == x.device
    assert out.dtype == x.dtype

def test_mismatched_channels_fails():
    x1 = torch.zeros(1, 2, 2, 1)
    x2 = torch.zeros(1, 2, 2, 2)
    with pytest.raises(AssertionError):
        square_pad([x1, x2])

def test_invalid_rank_fails():
    bad = torch.zeros(2, 2)  # 2D tensor
    with pytest.raises(ValueError):
        square_pad([bad])
