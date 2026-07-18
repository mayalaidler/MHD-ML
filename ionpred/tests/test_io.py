import h5py
import numpy as np
import pytest

from ionpred.io import discover_species, read_fields, species_label


def test_species_label_parsing():
    assert species_label("si  ") == "Si I"
    assert species_label("sip ") == "Si II"
    assert species_label("si2p") == "Si III"
    assert species_label("si3p") == "Si IV"
    assert species_label("o5p ") == "O VI"
    assert species_label("dens") is None
    assert species_label("velx") is None
    assert species_label("shok") is None


@pytest.fixture
def tiny_checkpoint(tmp_path):
    path = str(tmp_path / "chk_0001.h5")
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for name in ["dens", "temp", "velx", "vely", "velz",
                     "magx", "magy", "magz", "si  ", "sip ", "lsi "]:
            f.create_dataset(name, data=rng.random((4, 2, 2, 2)))
    return path


def test_discover_species(tiny_checkpoint):
    found = discover_species(tiny_checkpoint)
    assert found == {"si  ": "Si I", "sip ": "Si II"}


def test_read_fields_same_length_and_order(tiny_checkpoint):
    fields = read_fields(tiny_checkpoint, ["dens", "si  ", "sip"])
    lengths = {len(v) for v in fields.values()}
    assert lengths == {32}
    # trailing-space resolution: 'sip' resolves to 'sip '
    assert "sip" in fields
