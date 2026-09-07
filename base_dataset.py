"""Legacy import compatibility.

TMFNet++ applies synchronized tensor augmentations directly in dataset.py.
"""
from typing import Dict, Tuple


def get_params(size: Tuple[int, int], **_: object) -> Dict[str, object]:
    return {"size": size}


def get_transform(*_: object, **__: object):
    raise RuntimeError("Use the TMFNet++ dataset loaders in dataset.py instead.")
