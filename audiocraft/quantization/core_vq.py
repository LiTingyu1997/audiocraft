# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import mindspore
from einops import rearrange, repeat
# import flashy
# import torch
# from torch import nn, einsum
# import torch.nn.functional as F

from mindspore import nn, ops, Tensor, Parameter
import mindspore.common.initializer as init


def exists(val: tp.Optional[tp.Any]) -> bool:
    return val is not None


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if exists(val) else d


def l2norm(t):
    return ops.L2Normalize(axis=-1)(t)


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))
    #todo


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (ops.sum(x) + n_categories * epsilon)


def uniform_init(*shape: int):
    t = ops.empty(shape)
    init.HeUniform(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = ops.randperm(num_samples)[:num]
    else:
        indices = ops.randint(0, num_samples, (num,))

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        # diffs = rearrange(samples, "n d -> n () d") - rearrange(
        #     means, "c d -> () c d"
        # )
        diffs = ops.unsqueeze(samples, dim=1) - ops(
            means, dim=0
        )
        dists = ops.sum(-(diffs ** 2), dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = ops.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = ops.masked_fill(bins, zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        #new_means = ops.tensor_scatter_elements(axis=0, input_x=new_means, index=repeat(buckets, "n -> n d", d=dim), src=samples)
        new_means = ops.tensor_scatter_elements(axis=0, input_x=new_means, index=buckets.repeat(dim).reshape(-1, dim),
                                                src=samples)

        new_means = new_means / bins_min_clamped[..., None]

        means = ops.where(zero_mask[..., None], means, new_means)

    return means, bins


def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    n = t.shape[0]
    normed_codes = l2norm(t)
    identity = ops.eye(n)
    cosine_sim = ops.einsum("i d, j d -> i j", normed_codes, normed_codes)
    return ((cosine_sim - identity) ** 2).sum() / (n ** 2)


class EuclideanCodebook(nn.Cell):
    """Codebook with Euclidean distance.

    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.8,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        #init_fn: tp.Union[tp.Callable[..., Tensor], tp.Any] = uniform_init if not kmeans_init else ops.zeros
        #embed = init_fn((codebook_size, dim))
        embed = ops.zeros((codebook_size, dim))
        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        # self.register_buffer("inited", ops.Tensor([not kmeans_init]))
        # self.register_buffer("cluster_size", ops.zeros(codebook_size))
        # self.register_buffer("embed", embed)
        # self.register_buffer("embed_avg", embed.clone())
        self.inited = Parameter(ops.Tensor([float(not kmeans_init)], dtype=mindspore.float32), name='inited', requires_grad=False)
        self.cluster_size = Parameter(ops.zeros(codebook_size), name='cluster_size', requires_grad=False)
        self.embed = Parameter(embed, name='embed', requires_grad=False)
        #self.embed_avg = Parameter(embed.clone(), name='embed_avg')
        self.embed_avg = self.embed.clone()

    #@torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        # flashy.distrib.broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        modified_codebook = ops.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not ops.any(expired_codes):
            return

        #batch_samples = rearrange(batch_samples, "... d -> (...) d")
        batch_samples = batch_samples.reshape(-1, batch_samples.shape[-1])
        self.replace_(batch_samples, mask=expired_codes)
        # flashy.distrib.broadcast_tensors(self.buffers())

    def preprocess(self, x):
        #x = rearrange(x, "... d -> (...) d")
        return x.reshape(-1, x.shape[-1])

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        # import torch
        # import torch.nn.functional as F
        # embed_ind = torch.tensor(embed_ind.asnumpy())
        # embed_torch = torch.tensor(self.embed.asnumpy())
        # quantize = F.embedding(embed_ind, embed_torch)
        # quantize = Tensor(quantize.detach().numpy())

        cast = ops.Cast()
        embed_ind = cast(embed_ind, mindspore.int32)
        embed = cast(self.embed, mindspore.int32)
        #quantize = ops.EmbeddingLookup()(embed_ind, embed, 0)
        vocab_size, embedding_size = embed.shape
        embedding = nn.Embedding(vocab_size, embedding_size, embedding_table=embed)
        quantize = embedding(embed_ind)
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def construct(self, x):
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)
        self.init_embed_(x)

        embed_ind = self.quantize(x)
        embed_onehot = ops.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)

        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            self.expire_codes_(x)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind


class VectorQuantization(nn.Cell):
    """Vector quantization implementation.
    Currently supports only euclidean distance.

    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int):
        channels_last (bool): Channels are the last dimension in the input tensors.
        commitment_weight (float): Weight for commitment loss.
        orthogonal_reg_weight (float): Orthogonal regularization weights.
        orthogonal_reg_active_codes_only (bool): Apply orthogonal regularization only on active codes.
        orthogonal_reg_max_codes (optional int): Maximum number of codes to consider
            for orthogonal regularization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.8,
        epsilon: float = 1e-5,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: int = 2,
        channels_last: bool = False,
        commitment_weight: float = 1.,
        orthogonal_reg_weight: float = 0.0,
        orthogonal_reg_active_codes_only: bool = False,
        orthogonal_reg_max_codes: tp.Optional[int] = None,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        self._codebook = EuclideanCodebook(dim=_codebook_dim, codebook_size=codebook_size,
                                           kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                                           decay=decay, epsilon=epsilon,
                                           threshold_ema_dead_code=threshold_ema_dead_code)
        self.codebook_size = codebook_size

        self.channels_last = channels_last

    @property
    def codebook(self):
        return self._codebook.embed

    @property
    def inited(self):
        return self._codebook.inited

    def _preprocess(self, x):
        if not self.channels_last:
            b, d, n = x.shape
            x = x.reshape(b, n, d)
            #x = rearrange(x, "b d n -> b n d")
        return x

    def _postprocess(self, quantize):
        if not self.channels_last:
            # b, n, d = quantize.shape
            # quantize = quantize.reshape(b, d, n)
            quantize = quantize.asnumpy()
            quantize = rearrange(quantize, "b n d -> b d n")
            quantize = Tensor(quantize)
        return quantize

    def encode(self, x):
        x = self._preprocess(x)
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = self._postprocess(quantize)
        return quantize

    def construct(self, x):
        device = x.device
        x = self._preprocess(x)

        x = self.project_in(x)
        quantize, embed_ind = self._codebook(x)

        if self.training:
            quantize = x + (quantize - x).detach()

        loss = Tensor([0.0], requires_grad=self.training)

        if self.training:
            if self.commitment_weight > 0:
                commit_loss = ops.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

            if self.orthogonal_reg_weight > 0:
                codebook = self.codebook

                if self.orthogonal_reg_active_codes_only:
                    # only calculate orthogonal loss for the activated codes for this batch
                    unique_code_ids = ops.unique(embed_ind)
                    codebook = codebook[unique_code_ids]

                num_codes = codebook.shape[0]
                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = ops.randperm(num_codes, device=device)[:self.orthogonal_reg_max_codes]
                    codebook = codebook[rand_ids]

                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        quantize = self.project_out(quantize)
        quantize = self._postprocess(quantize)

        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Cell):
    """Residual vector quantization implementation.

    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.CellList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def construct(self, x, n_q: tp.Optional[int] = None):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        n_q = n_q or len(self.layers)

        for i, layer in enumerate(self.layers[:n_q]):
            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            all_losses.append(loss)

        out_losses, out_indices = map(ops.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses

    def encode(self, x: Tensor, n_q: tp.Optional[int] = None) -> Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = ops.stack(all_indices)
        return out_indices

    def decode(self, q_indices: Tensor) -> Tensor:
        quantized_out = Tensor(0.0)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out
