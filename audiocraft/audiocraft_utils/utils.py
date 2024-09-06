# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import wraps, lru_cache
import hashlib
import json
import logging
from pathlib import Path
import typing as tp

# import flashy
# import flashy.distrib
import omegaconf
#import torch
#from torch.nn.utils.rnn import pad_sequence

import mindspore
from mindspore import ops, Tensor, nn
import numpy as np


logger = logging.getLogger(__name__)

def pad_sequence(
    sequences: tp.List[np.ndarray],
    batch_first=True,
    padding_value: int = 0,
    padding_max_len: int = None,
    atype=np.int32,
) -> np.ndarray:
    """[summary]

    Args:
        sequences (List[np.ndarray]): [description]
        batch_first (bool, optional): [description]. Defaults to True.
        padding_value (int, optional): [description]. Defaults to 0.
        padding_max_len (int, optional): [description]. Defaults to None.
        atype ([type], optional): [description]. Defaults to np.int32.

    Returns:
        np.ndarray: [description]
    """
    max_size = sequences[0].shape
    trailing_dims = max_size[1:]

    if padding_max_len is not None:
        max_len = padding_max_len
    else:
        max_len = max([s.shape[0] for s in sequences])

    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_sequences = np.full(out_dims, fill_value=padding_value).astype(atype)

    for i, seq in enumerate(sequences):
        length = seq.shape[0] if seq.shape[0] <= max_len else max_len
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_sequences[i, :length, ...] = seq[:length]
        else:
            out_sequences[:length, i, ...] = seq[:length]

    return out_sequences


def model_hash(model: nn.Cell) -> str:
    """Return a model hash. This should allow us to track regressions in model init
    from the logs of past experiments.
    """
    hasher = hashlib.sha1()
    for p in model.parameters():
        hasher.update(p.data.cpu().numpy().tobytes())
    return hasher.hexdigest()


def dict_from_config(cfg: omegaconf.DictConfig) -> dict:
    """Convenience function to map an omegaconf configuration to a dictionary.

    Args:
        cfg (omegaconf.DictConfig): Original configuration to map to dict.
    Returns:
        dict: Config as dictionary object.
    """
    dct = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(dct, dict)
    return dct


# def random_subset(dataset, max_samples: int, seed: int = 42) -> torch.utils.data.Subset:
#     if max_samples >= len(dataset):
#         return dataset

#     generator = torch.Generator().manual_seed(seed)
#     perm = torch.randperm(len(dataset), generator=generator)
#     return torch.utils.data.Subset(dataset, perm[:max_samples].tolist())


# def get_loader(dataset, num_samples: tp.Optional[int], batch_size: int,
#                num_workers: int, seed: int, **kwargs) -> torch.utils.data.DataLoader:
#     """Convenience function to load dataset into a dataloader with optional subset sampling.

#     Args:
#         dataset: Dataset to load.
#         num_samples (Optional[int]): Number of samples to limit subset size.
#         batch_size (int): Batch size.
#         num_workers (int): Number of workers for data loading.
#         seed (int): Random seed.
#     """
#     if num_samples is not None:
#         dataset = random_subset(dataset, num_samples, seed)

#     dataloader = flashy.distrib.loader(
#         dataset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         **kwargs
#     )
#     return dataloader


def get_dataset_from_loader(dataloader):
    dataset = dataloader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        return dataset.dataset
    else:
        return dataset


def multinomial(input: Tensor, num_samples: int, replacement=False, *, seed=None):
    """torch.multinomial with arbitrary number of dimensions, and number of candidates on the last dimension.

    Args:
        input (Tensor): The input tensor containing probabilities.
        num_samples (int): Number of samples to draw.
        replacement (bool): Whether to draw with replacement or not.
    Keywords args:
        seed (int): A pseudorandom number generator for sampling.
    Returns:
        torch.Tensor: Last dimension contains num_samples indices
            sampled from the multinomial probability distribution
            located in the last dimension of tensor input.
    """
    input_ = ops.reshape(input, (-1, input.shape[-1]))
    cast_method = ops.Cast()
    input_ = cast_method(input_, mindspore.float16)
    input_ = cast_method(input_, mindspore.float32)
    # import torch
    # input_ = torch.tensor(input_.asnumpy())
    # rng = torch.Generator()
    # #rng.manual_seed(seed)
    # if seed:
    #     output_ = torch.multinomial(input_, num_samples=num_samples, replacement=replacement, generator=rng.manual_seed(seed))
    # else:
    #     output_ = torch.multinomial(input_, num_samples=num_samples, replacement=replacement)
    # output_ = Tensor(output_.detach().numpy())
    output_ = ops.multinomial(input_, num_samples=num_samples, replacement=replacement, seed=seed)
    output = ops.reshape(output_, ((*list(input.shape[:-1]), -1)))
    return output

#todo
def sample_top_k(probs: Tensor, k: int) -> Tensor:
    """Sample next token from top K values along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        k (int): The k in “top-k”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    top_k_value, _ = ops.topk(probs, k, dim=-1)
    min_value_top_k = top_k_value[..., [-1]]
    probs *= (probs >= min_value_top_k).float()
    probs = ops.div(probs, ops.sum(probs, dim=-1, keepdim=True))
    next_token = multinomial(probs, num_samples=1)
    return next_token

# x = Tensor(np.array([[8, 2, 1], [5, 9, 3], [4, 6, 7]]))
# test_x = sample_top_k(x, 0.5)


#todo
def sample_top_p(probs: Tensor, p: float) -> Tensor:
    """Sample next token from top P probabilities along the last dimension of the input probs tensor.

    Args:
        probs (Tensor): Input probabilities with token candidates on the last dimension.
        p (int): The p in “top-p”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    probs_sort, probs_idx = ops.sort(probs, axis=-1, descending=True)
    probs_sum = ops.cumsum(probs_sort, axis=-1)
    mask = probs_sum - probs_sort > p
    probs_sort *= (~mask).float()
    #_fuzhi
    probs_sort = ops.div(probs_sort, ops.sum(probs_sort, dim=-1, keepdim=True))

    next_token = multinomial(probs_sort, num_samples=1)
    next_token = ops.gather_elements(probs_idx, -1, next_token)
    return next_token

def test_sample_top_p():
    x = Tensor(np.array([[8, 2, 1], [5, 9, 3], [4, 6, 7]]))
    test_x = sample_top_p(x, 0.5)


class DummyPoolExecutor:
    """Dummy pool executor to use when we actually have only 1 worker.
    (e.g. instead of ProcessPoolExecutor).
    """
    class DummyResult:
        def __init__(self, func, *args, **kwargs):
            self.func = func
            self.args = args
            self.kwargs = kwargs

        def result(self):
            return self.func(*self.args, **self.kwargs)

    def __init__(self, workers, mp_context=None):
        pass

    def submit(self, func, *args, **kwargs):
        return DummyPoolExecutor.DummyResult(func, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return


def get_pool_executor(num_workers: int, mp_context=None):
    return ProcessPoolExecutor(num_workers, mp_context) if num_workers > 1 else DummyPoolExecutor(1)


def length_to_mask(lengths: Tensor, max_len: tp.Optional[int] = None) -> Tensor:
    """Utility function to convert a tensor of sequence lengths to a mask (useful when working on padded sequences).
    For example: [3, 5] => [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]

    Args:
        lengths (torch.Tensor): tensor with lengths
        max_len (int): can set the max length manually. Defaults to None.
    Returns:
        torch.Tensor: mask with 0s where there is pad tokens else 1s
    """
    assert len(lengths.shape) == 1, "Length shape should be 1 dimensional."
    final_length = lengths.max().item() if not max_len else max_len
    final_length = max(final_length, 1)  # if all seqs are of len zero we don't want a zero-size tensor
    return ops.arange(final_length)[None, :].to(lengths.device) < lengths[:, None]


def hash_trick(word: str, vocab_size: int) -> int:
    """Hash trick to pair each word with an index

    Args:
        word (str): word we wish to convert to an index
        vocab_size (int): size of the vocabulary
    Returns:
        int: index of the word in the embedding LUT
    """
    hash = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
    return hash % vocab_size


# def with_rank_rng(base_seed: int = 1234):
#     """Decorator for a function so that the function will use a Random Number Generator
#     whose state depend on the GPU rank. The original RNG state is restored upon returning.

#     Args:
#         base_seed (int): Random seed.
#     """
#     def _decorator(fun: tp.Callable):
#         @wraps(fun)
#         def _decorated(*args, **kwargs):
#             state = torch.get_rng_state()
#             seed = base_seed ^ flashy.distrib.rank()
#             torch.manual_seed(seed)
#             logger.debug('Rank dependent seed set to %d', seed)
#             try:
#                 return fun(*args, **kwargs)
#             finally:
#                 torch.set_rng_state(state)
#                 logger.debug('RNG state restored.')
#         return _decorated
#     return _decorator


def collate(tensors: tp.List[Tensor], dim: int = 0) -> tp.Tuple[Tensor, Tensor]:
    """Get a list of tensors and collate them to a single tensor. according to the following logic:
    - `dim` specifies the time dimension which will be stacked and padded.
    - The output will contain 1 new dimension (dimension index 0) which will be the size of
    of the original list.

    Args:
        tensors (tp.List[torch.Tensor]): List of tensors to collate.
        dim (int): Dimension which will be stacked and padded.
    Returns:
        tp.Tuple[torch.Tensor, torch.Tensor]:
            torch.Tensor: Stacked and padded tensor. The output will contain 1 new dimension
                (dimension index 0) which will be the size of the original list.
            torch.Tensor: Tensor containing length of original tensor sizes (without padding).
    """
    tensors = [x.transpose(0, dim) for x in tensors]
    lens = mindspore.LongTensor([len(x) for x in tensors])
    padded_tensors = pad_sequence(tensors)
    padded_tensors = padded_tensors.transpose(0, 1)
    padded_tensors = padded_tensors.transpose(1, dim + 1)
    return padded_tensors, lens


# TODO: Move to flashy?
# def copy_state(state: tp.Any, device: tp.Union[torch.device, str] = 'cpu',
#                dtype: tp.Optional[torch.dtype] = None) -> tp.Any:
#     if isinstance(state, torch.Tensor):
#         if dtype is None or not state.is_floating_point():
#             dtype = state.dtype
#         return state.detach().to(device=device, dtype=dtype, copy=True)
#     elif isinstance(state, dict):
#         return {k: copy_state(v, device, dtype) for k, v in state.items()}
#     elif isinstance(state, list):
#         return [copy_state(v, device, dtype) for v in state]


# # TODO: Move to flashy?
# @contextmanager
# def swap_state(model, state, **kwargs):
#     old_state = copy_state(model.state_dict())
#     model.load_state_dict(state, **kwargs)
#     try:
#         yield
#     finally:
#         model.load_state_dict(old_state)


@lru_cache(None)
def warn_once(logger, msg):
    """Warn about a given message only once."""
    logger.warning(msg)


def is_jsonable(x: tp.Any):
    """Check if an object can be serialized into a json:"""
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def load_clap_state_dict(clap_model, path: tp.Union[str, Path]):
    """Wrapper around state dict loading of CLAP model
    addressing compatibility issues between CLAP and AudioCraft
    HuggingFace transformer version.
    See: https://github.com/LAION-AI/CLAP/issues/118
    """
    from clap_module.factory import load_state_dict  # type: ignore
    pkg = load_state_dict(path)
    pkg.pop('text_branch.embeddings.position_ids', None)
    clap_model.model.load_state_dict(pkg)
