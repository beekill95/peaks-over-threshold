from __future__ import annotations

import os
import pickle

from ucimlrepo import fetch_ucirepo

CACHE_DIR = ".cache"


def _ensure_cache_dir_exists():
    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)


def _patch_dotdict():
    from ucimlrepo.dotdict import dotdict

    dotdict.__getstate__ = lambda self: self.__dict__
    dotdict.__setstate__ = lambda self, d: self.__dict__.update(d)


def fetch_ds(id: int):
    _ensure_cache_dir_exists()
    _patch_dotdict()

    cached_file = os.path.join(CACHE_DIR, f"ds_{id}.pkl")
    try:
        with open(cached_file, "rb") as infile:
            return pickle.load(infile)
    except FileNotFoundError:
        ds = fetch_ucirepo(id=id)

        with open(cached_file, "wb") as outfile:
            pickle.dump(ds, outfile)

        return ds
