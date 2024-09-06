# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions to load from the checkpoints.
Each checkpoint is a torch.saved dict with the following keys:
- 'xp.cfg': the hydra config as dumped during training. This should be used
    to rebuild the object using the audiocraft.models.builders functions,
- 'model_best_state': a readily loadable best state for the model, including
    the conditioner. The model obtained from `xp.cfg` should be compatible
    with this state dict. In the case of a LM, the encodec model would not be
    bundled along but instead provided separately.

Those functions also support loading from a remote location with the Torch Hub API.
They also support overriding some parameters, in particular the device and dtype
of the returned model.
"""

from pathlib import Path

import mindspore
from huggingface_hub import hf_hub_download
import typing as tp
import os

from omegaconf import OmegaConf, DictConfig
import torch

import builders
from encodec import CompressionModel

from mindspore import Tensor


def get_audiocraft_cache_dir() -> tp.Optional[str]:
    return os.environ.get('AUDIOCRAFT_CACHE_DIR', None)


def _get_state_dict(
    file_or_url_or_id: tp.Union[Path, str],
    filename: tp.Optional[str] = None,
    device='cpu',
    cache_dir: tp.Optional[str] = None,
):
    if cache_dir is None:
        cache_dir = get_audiocraft_cache_dir()
    # Return the state dict either from a file or url
    file_or_url_or_id = str(file_or_url_or_id)
    assert isinstance(file_or_url_or_id, str)

    if os.path.isfile(file_or_url_or_id):
        return torch.load(file_or_url_or_id, map_location=device)

    if os.path.isdir(file_or_url_or_id):
        file = f"{file_or_url_or_id}/{filename}"
        return torch.load(file, map_location=device)

    elif file_or_url_or_id.startswith('https://'):
        return torch.hub.load_state_dict_from_url(file_or_url_or_id, map_location=device, check_hash=True)

    else:
        assert filename is not None, "filename needs to be defined if using HF checkpoints"

        file = hf_hub_download(repo_id=file_or_url_or_id, filename=filename, cache_dir=cache_dir)
        return torch.load(file, map_location=device)


def load_compression_model_ckpt(file_or_url_or_id: tp.Union[Path, str], cache_dir: tp.Optional[str] = None):
    return _get_state_dict(file_or_url_or_id, filename="compression_state_dict.bin", cache_dir=cache_dir)


def load_compression_model(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = load_compression_model_ckpt(file_or_url_or_id, cache_dir=cache_dir)
    if 'pretrained' in pkg:
        return CompressionModel.get_pretrained(pkg['pretrained'], device=device)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    model = builders.get_compression_model(cfg)
    ckpt_pre = pkg['best_state']
    param_dict = []
    for k in ckpt_pre.keys():
        # if ckpt_pre[k].dtype == torch.float16:
        #     ms_type = mindspore.float16
        # else:
        # if "weight_g" in k:
        #     print(k)
        #     k_new = k.split("weight_g")[0] + "param_g"
        # elif "weight_v" in k:
        #     k_new = k.split("weight_v")[0] + "param_v"
        # else:
        #     k_new = k
        ms_type = mindspore.float32
        param_np = ckpt_pre[k].numpy()
        param_ms = Tensor(param_np, dtype=ms_type)
        param_dict.append({"name": k, "data": param_ms})
    #mindspore.save_checkpoint(param_dict, '/home/litingyu/08c/compressin.ckpt')
    #print(model)
    param_dict = mindspore.load_checkpoint('/disk1/lty/small_config/compressin.ckpt')
    #print(param_dict)
    #model.load_state_dict(pkg['best_state'])
    mindspore.load_param_into_net(model, param_dict)
    #model.eval()
    model.set_train(False)
    return model


def load_lm_model_ckpt(file_or_url_or_id: tp.Union[Path, str], cache_dir: tp.Optional[str] = None):
    return _get_state_dict(file_or_url_or_id, filename="state_dict.bin", cache_dir=cache_dir)


def _delete_param(cfg: DictConfig, full_name: str):
    parts = full_name.split('.')
    for part in parts[:-1]:
        if part in cfg:
            cfg = cfg[part]
        else:
            return
    OmegaConf.set_struct(cfg, False)
    if parts[-1] in cfg:
        del cfg[parts[-1]]
    OmegaConf.set_struct(cfg, True)


def load_lm_model(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    #pkg = load_lm_model_ckpt(file_or_url_or_id, cache_dir=cache_dir)
    # ckpt_pre = pkg['best_state']
    # param_dict = []
    # for k in ckpt_pre.keys():
    #     # if ckpt_pre[k].dtype == torch.float16:
    #     #     ms_type = mindspore.float16
    #     # else:
    #     if "norm" in k:
    #         if "weight" in k:
    #             k_new = k.split(".weight")[0] + ".gamma"
    #         if "bias" in k:
    #             k_new = k.split(".bias")[0] + ".beta"
    #     else:
    #         k_new = k
    #     ms_type = mindspore.float32
    #     param_np = ckpt_pre[k].numpy()
    #     param_ms = Tensor(param_np, dtype=ms_type)
    #     param_dict.append({"name": k_new, "data": param_ms})
    # mindspore.save_checkpoint(param_dict, '/home/litingyu/08c/test.ckpt')
    #print(pkg['xp.cfg'])
    #cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg = OmegaConf.load("/disk1/lty/small_config/config.yaml")
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    _delete_param(cfg, 'conditioners.self_wav.chroma_stem.cache_path')
    _delete_param(cfg, 'conditioners.args.merge_text_conditions_p')
    _delete_param(cfg, 'conditioners.args.drop_desc_p')

    model = builders.get_lm_model(cfg)
    # ms_params = {}
    # for param in model.get_parameters():
    #     name = param.name
    #     value = param.data.asnumpy()
    #     print(name, value.shape)
    #     ms_params[name] = value
    #model.load_state_dict(pkg['best_state'])


    param_dict = mindspore.load_checkpoint('/disk1/lty/small_config/test.ckpt')
    #print(param_dict)
    mindspore.load_param_into_net(model, param_dict)

    model.set_train(False)
    #model.cfg = cfg
    return model


def load_mbd_ckpt(file_or_url_or_id: tp.Union[Path, str], cache_dir: tp.Optional[str] = None):
    return _get_state_dict(file_or_url_or_id, filename="all_in_one.pt", cache_dir=cache_dir)


def load_diffusion_models(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = load_mbd_ckpt(file_or_url_or_id, cache_dir=cache_dir)
    models = []
    processors = []
    cfgs = []
    sample_rate = pkg['sample_rate']
    for i in range(pkg['n_bands']):
        cfg = pkg[i]['cfg']
        model = builders.get_diffusion_model(cfg)
        model_dict = pkg[i]['model_state']
        model.load_state_dict(model_dict)
        model.to(device)
        processor = builders.get_processor(cfg=cfg.processor, sample_rate=sample_rate)
        processor_dict = pkg[i]['processor_state']
        processor.load_state_dict(processor_dict)
        processor.to(device)
        models.append(model)
        processors.append(processor)
        cfgs.append(cfg)
    return models, processors, cfgs
