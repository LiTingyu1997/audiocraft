# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import typing as tp
import warnings

# import torch
# from torch.nn.utils import spectral_norm, weight_norm
# a = torch.nn.Linear(20, 40)
# m = spectral_norm(torch.nn.Linear(20, 40))


CONV_NORMALIZATIONS = frozenset(['none', 'weight_norm', 'spectral_norm',
                                 'time_group_norm'])

import numpy as np
import mindspore
from mindspore import nn, ops, Tensor, Parameter
import mindspore.numpy as mnp
from mindspore.common.initializer import initializer, Normal

#context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
mindspore.set_seed(123)
np.random.seed(123)


class _SpectralNorm(nn.Cell):
    def __init__(
            self,
            weight,
            n_power_iterations: int = 1,
            dim: int = 0,
            eps: float = 1e-12
    ) -> None:
        super(_SpectralNorm, self).__init__()
        self.dim = dim
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.expand_dims = ops.ExpandDims()
        self.l2_normalize = ops.L2Normalize(epsilon=self.eps)

        weight_mat = self._reshape_weight_to_matrix(weight)

        h, w = weight_mat.shape
        init_u = initializer(Normal(1.0, 0), [h], mindspore.float32).init_data()
        init_v = initializer(Normal(1.0, 0), [w], mindspore.float32).init_data()
        self.u = Parameter(self.l2_normalize(init_u), requires_grad=False)
        self.v = Parameter(self.l2_normalize(init_v), requires_grad=False)
        self._update_vectors(weight_mat, init=True)

    def _reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dim != 0:
            list_dim = list(range(1, self.dim + 1))
            list_dim.append(0)
            for i in range(self.dim + 1, weight_mat.ndim):
                list_dim.append(i)

            weight_mat = mnp.moveaxis(weight_mat, list(range(weight_mat.ndim)), list_dim)
        height = weight_mat.shape[0]
        return weight_mat.reshape((height, -1))

    def _update_vectors(self, weight_mat, init=False) -> None:
        for _ in range(self.n_power_iterations):
            self.u = self.l2_normalize(mnp.multi_dot([weight_mat, self.expand_dims(self.v, -1)]).flatten())
            self.v = self.l2_normalize(mnp.multi_dot([weight_mat.T, self.expand_dims(self.u, -1)]).flatten())

    def construct(self, weight):
        weight_mat = self._reshape_weight_to_matrix(weight)
        #print(weight_mat)
        self._update_vectors(weight_mat)
        sigma = ops.tensor_dot(self.u, mnp.multi_dot([weight_mat, self.expand_dims(self.v, -1)]), 1)
        return weight / sigma


# spe = _SpectralNorm()
# res = spe(Tensor([[1, 2], [3, 4]], mindspore.float32))
# print("res: ", res)
from typing import Any, TypeVar


def norm_except_dim(v, pow, dim):
    if dim == -1:
        return mnp.norm(v, pow)
    elif dim == 0:
        output_size = (v.shape[0],) + (1,) * (v.ndim - 1)
        return mnp.norm(v.view((v.shape[0], -1)), pow, 1).view(output_size)
    elif dim == (v.ndim - 1):
        output_size = (1,) * (v.ndim - 1) + (v.shape[v.ndim - 1])
        return mnp.norm(v.view((-1, v.shape[v.ndim - 1])), pow, 0).view(output_size)
    else:
        return norm_except_dim(v.swapaxes(0, dim), pow, dim).swapaxes(0, dim)


def _weight_norm(v, g, dim):
    return v * (g / norm_except_dim(v, 2, dim))

class WeightNorm(nn.Cell):
    def __init__(self, module, dim: int = 0):
        super().__init__()

        if dim is None:
            dim = -1

        self.dim = dim
        self.module = module

        self.assign = ops.Assign()
        # add g and v as new parameters and express w as g/||v|| * v
        self.param_g = Parameter(Tensor(norm_except_dim(self.module.weight, 2, dim)))
        self.param_v = Parameter(Tensor(self.module.weight.data))
        self.module.weight.set_data(_weight_norm(self.param_v, self.param_g, self.dim))

        self.use_weight_norm = True
        self.kernel_size = module.kernel_size
        self.stride = module.stride
        self.dilation = module.dilation

    def construct(self, *inputs, **kwargs):
        if not self.use_weight_norm:
            return self.module(*inputs, **kwargs)

        # if isinstance(self.module, nn.Conv1dTranspose):
        #     test_a = self.module.weight
        #     print(test_a.asnumpy())
        #     return self.module(*inputs, **kwargs)

        self.assign(self.module.weight, _weight_norm(self.param_v, self.param_g, self.dim))
        return self.module(*inputs, **kwargs)

    def remove_weight_norm(self):
        self.assign(self.module.weight, _weight_norm(self.param_v, self.param_g, self.dim))
        self.use_weight_norm = False


def apply_parametrization_norm(module: nn.Cell, norm: str = 'none'):
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return WeightNorm(module)
    elif norm == 'spectral_norm':
        return _SpectralNorm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module
#apply_parametrization_norm(nn.Conv1d(120, 240, 4, has_bias=False, weight_init='normal'), norm= 'weight_norm')


def get_norm_module(module: nn.Cell, causal: bool = False, norm: str = 'none', **norm_kwargs):
    """Return the proper normalization module. If causal is True, this will ensure the returned
    module is causal, or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


def get_extra_padding_for_conv1d(x: Tensor, kernel_size: int, stride: int,
                                 padding_total: int = 0) -> int:
    """See `pad_for_conv1d`."""
    length = ops.shape(x)[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(x: Tensor, kernel_size: int, stride: int, padding_total: int = 0):
    """Pad for a convolution to make sure that the last window is full.
    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.
    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution, last 0 is never used)
        0 0 1 2 3 4 5 0     # (output of tr. conv., but pos. 5 is going to get removed as padding)
            1 2 3 4         # once you removed padding, we are missing one time step !
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return ops.pad(x, (0, extra_padding))


def pad1d(x: Tensor, paddings: tp.Tuple[int, int], mode: str = 'constant', value: float = 0.):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = ops.shape(x)[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = ops.pad(x, (0, extra_pad))
        #padded = ops.pad(x, paddings, mode, value)
        padded = ops.pad(x, paddings, mode)
        end = ops.shape(padded)[-1] - extra_pad
        return padded[..., :end]
    else:
        return ops.pad(x, paddings, mode, value)


def unpad1d(x: Tensor, paddings: tp.Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= ops.shape(x)[-1]
    end = ops.shape(x)[-1] - padding_right
    return x[..., padding_left: end]


class NormConv1d(nn.Cell):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, pad_mode='valid', **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def construct(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConv2d(nn.Cell):
    """Wrapper around Conv2d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal=False, norm=norm, **norm_kwargs)
        self.norm_type = norm

    def construct(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConvTranspose1d(nn.Cell):
    """Wrapper around ConvTranspose1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.Conv1dTranspose(*args, pad_mode='valid', has_bias=True, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def construct(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x


class NormConvTranspose2d(nn.Cell):
    """Wrapper around ConvTranspose2d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal=False, norm=norm, **norm_kwargs)

    def construct(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x


class StreamableConv1d(nn.Cell):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, dilation: int = 1,
                 groups: int = 1, bias: bool = True, causal: bool = False,
                 norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {},
                 pad_mode: str = 'reflect'):
        super().__init__()
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            warnings.warn("StreamableConv1d has been initialized with stride > 1 and dilation > 1"
                          f" (kernel_size={kernel_size} stride={stride}, dilation={dilation}).")
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride,
                               dilation=dilation, group=groups, has_bias=bias, causal=causal,
                               norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode

    def construct(self, x):
        B, C, T = x.shape
        kernel_size = self.conv.conv.kernel_size[1]
        stride = self.conv.conv.stride[1]
        dilation = self.conv.conv.dilation[1]
        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        if self.causal:
            # Left padding for causal
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)

        return self.conv(x)


class StreamableConvTranspose1d(nn.Cell):
    """ConvTranspose1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, causal: bool = False,
                 norm: str = 'none', trim_right_ratio: float = 1.,
                 norm_kwargs: tp.Dict[str, tp.Any] = {}):
        super().__init__()
        self.convtr = NormConvTranspose1d(in_channels, out_channels, kernel_size, stride,
                                          causal=causal, norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1., \
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0. and self.trim_right_ratio <= 1.

    def construct(self, x):
        kernel_size = self.convtr.convtr.kernel_size[1]
        stride = self.convtr.convtr.stride[1]
        padding_total = kernel_size - stride

        y = self.convtr(x)

        # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
        # removed at the very end, when keeping only the right length for the output,
        # as removing it here would require also passing the length at the matching layer
        # in the encoder.
        if self.causal:
            # Trim the padding on the right according to the specified ratio
            # if trim_right_ratio = 1.0, trim everything from right
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y
