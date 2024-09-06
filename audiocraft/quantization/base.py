# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Base class for all quantizers.
"""

from dataclasses import dataclass, field
import typing as tp

# import torch
# from torch import nn
from mindspore import nn, ops, Tensor


@dataclass
class QuantizedResult:
    x: Tensor
    codes: Tensor
    bandwidth: Tensor  # bandwidth in kb/s used, per batch item.
    penalty: tp.Optional[Tensor] = None
    metrics: dict = field(default_factory=dict)


class BaseQuantizer(nn.Cell):
    """Base class for quantizers.
    """

    def forward(self, x: Tensor, frame_rate: int) -> QuantizedResult:
        """
        Given input tensor x, returns first the quantized (or approximately quantized)
        representation along with quantized codes, bandwidth, and any penalty term for the loss.
        Finally, this returns a dict of metrics to update logging etc.
        Frame rate must be passed so that the bandwidth is properly computed.
        """
        raise NotImplementedError()

    def encode(self, x: Tensor) -> Tensor:
        """Encode a given input tensor with the specified sample rate at the given bandwidth."""
        raise NotImplementedError()

    def decode(self, codes: Tensor) -> Tensor:
        """Decode the given codes to the quantized representation."""
        raise NotImplementedError()

    @property
    def total_codebooks(self):
        """Total number of codebooks."""
        raise NotImplementedError()

    @property
    def num_codebooks(self):
        """Number of active codebooks."""
        raise NotImplementedError()

    def set_num_codebooks(self, n: int):
        """Set the number of active codebooks."""
        raise NotImplementedError()


class DummyQuantizer(BaseQuantizer):
    """Fake quantizer that actually does not perform any quantization.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, frame_rate: int):
        q = ops.unsqueeze(x, 1)
        return QuantizedResult(x, q, Tensor(q.numel() * 32 * frame_rate / 1000 / len(x), x.dtype))

    def encode(self, x: Tensor) -> Tensor:
        """Encode a given input tensor with the specified sample rate at the given bandwidth.
        In the case of the DummyQuantizer, the codes are actually identical
        to the input and resulting quantized representation as no quantization is done.
        """
        return ops.unsqueeze(x, 1)

    def decode(self, codes: Tensor) -> Tensor:
        """Decode the given codes to the quantized representation.
        In the case of the DummyQuantizer, the codes are actually identical
        to the input and resulting quantized representation as no quantization is done.
        """
        return ops.squeeze(codes, 1)

    @property
    def total_codebooks(self):
        """Total number of codebooks."""
        return 1

    @property
    def num_codebooks(self):
        """Total number of codebooks."""
        return self.total_codebooks

    def set_num_codebooks(self, n: int):
        """Set the number of active codebooks."""
        raise AttributeError("Cannot override the number of codebooks for the dummy quantizer")
