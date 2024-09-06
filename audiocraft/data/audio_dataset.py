from dataclasses import dataclass, fields

import logging

import typing as tp

from .zip import PathInZip

@dataclass(order=True)
class BaseInfo:

    @classmethod
    def _dict2fields(cls, dictionary: dict):
        return {
            field.name: dictionary[field.name]
            for field in fields(cls) if field.name in dictionary
        }

    @classmethod
    def from_dict(cls, dictionary: dict):
        _dictionary = cls._dict2fields(dictionary)
        return cls(**_dictionary)

    def to_dict(self):
        return {
            field.name: self.__getattribute__(field.name)
            for field in fields(self)
            }


@dataclass(order=True)
class AudioMeta(BaseInfo):
    path: str
    duration: float
    sample_rate: int
    amplitude: tp.Optional[float] = None
    weight: tp.Optional[float] = None
    # info_path is used to load additional information about the audio file that is stored in zip files.
    info_path: tp.Optional[PathInZip] = None

    @classmethod
    def from_dict(cls, dictionary: dict):
        base = cls._dict2fields(dictionary)
        if 'info_path' in base and base['info_path'] is not None:
            base['info_path'] = PathInZip(base['info_path'])
        return cls(**base)

    def to_dict(self):
        d = super().to_dict()
        if d['info_path'] is not None:
            d['info_path'] = str(d['info_path'])
        return d


@dataclass(order=True)
class SegmentInfo(BaseInfo):
    meta: AudioMeta
    seek_time: float
    # The following values are given once the audio is processed, e.g.
    # at the target sample rate and target number of channels.
    n_frames: int      # actual number of frames without padding
    total_frames: int  # total number of frames, padding included
    sample_rate: int   # actual sample rate
    channels: int      # number of audio channels.


DEFAULT_EXTS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

logger = logging.getLogger(__name__)