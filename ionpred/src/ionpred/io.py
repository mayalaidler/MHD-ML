"""Reading FLASH HDF5 checkpoints and discovering species fields."""

from __future__ import annotations

import re

import h5py
import numpy as np

# FLASH stores each variable as a 4-character dataset name.  Species follow
# "<element><ionization>" where ionization is '' (neutral), 'p' (singly
# ionized), '2p', '3p', ... .  Everything is padded to 4 chars with spaces.
_ELEMENTS = {
    "h": "H", "he": "He", "c": "C", "n": "N", "o": "O", "ne": "Ne",
    "na": "Na", "mg": "Mg", "si": "Si", "s": "S", "ca": "Ca", "fe": "Fe",
}
_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
          "XI", "XII"]

# Non-species 4-char FLASH variables we should never offer as targets.
_NOT_SPECIES = {
    "dens", "temp", "pres", "velx", "vely", "velz", "magx", "magy", "magz",
    "magp", "eint", "ener", "gamc", "game", "accx", "accy", "accz",
    "divb", "shok", "elec", "metl", "oden", "otmp", "chdt", "jtdt",
    "cjto", "mode", "np  ", "sp  ",
}

_SPECIES_RE = re.compile(r"^([a-z]{1,2})(|p|[1-9]p)\s*$")


def species_label(field: str) -> str | None:
    """'si  ' -> 'Si I', 'sip ' -> 'Si II', 'si3p' -> 'Si IV'; None if not
    a recognized species field."""
    if field in _NOT_SPECIES:
        return None
    m = _SPECIES_RE.match(field)
    if not m:
        return None
    elem, ion = m.groups()
    if elem not in _ELEMENTS:
        return None
    if ion == "":
        stage = 0          # neutral: 'si  ' -> Si I
    elif ion == "p":
        stage = 1          # singly ionized: 'sip ' -> Si II
    else:
        stage = int(ion[:-1])  # 'si2p' -> Si III, 'si3p' -> Si IV
    if stage >= len(_ROMAN):
        return None
    return f"{_ELEMENTS[elem]} {_ROMAN[stage]}"


def discover_species(path: str) -> dict[str, str]:
    """Map of species field name -> human label for one checkpoint file.

    Skips the log-abundance duplicates ('lsi ', 'lo  ', ...) that some
    setups write alongside the linear fields.
    """
    out: dict[str, str] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            if len(key) != 4 or key.startswith("l"):
                continue
            label = species_label(key)
            if label is not None and isinstance(f[key], h5py.Dataset):
                out[key] = label
    return dict(sorted(out.items(), key=lambda kv: kv[1]))


def read_fields(path: str, names: list[str]) -> dict[str, np.ndarray]:
    """Read the named FLASH variables as flat float64 arrays.

    All arrays come from the same file in the same cell order, so rows
    correspond to the same cells across fields — the invariant every
    downstream step relies on.
    """
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for name in names:
            key = _resolve_key(f, name)
            out[name] = f[key][()].astype(np.float64).ravel()
    n = min(len(v) for v in out.values())
    return {k: v[:n] for k, v in out.items()}


def _resolve_key(f: h5py.File, name: str) -> str:
    """FLASH names may be stored with or without trailing spaces."""
    for key in (name, name.rstrip(), name.rstrip().ljust(4)):
        if key in f:
            return key
    raise KeyError(
        f"Field {name!r} not found. Available: {sorted(f.keys())[:40]}...")
