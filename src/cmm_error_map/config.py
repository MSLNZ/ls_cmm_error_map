from pathlib import Path

import tomllib as toml

import cmm_error_map.data_cmpts as dc

config_folder = Path(__file__).parent.parent.parent / "config"
machines_toml = config_folder / "machines.toml"
artefacts_toml = config_folder / "artefacts.toml"


def read_toml(fn, input_type):
    with open(fn, "rb") as f:
        toml_in = toml.load(f)
    toml_in = {key: input_type(**value) for key, value in toml_in.items()}
    return toml_in


cmm_models = read_toml(machines_toml, dc.MachineType)
artefact_models = read_toml(artefacts_toml, dc.ArtefactType)
