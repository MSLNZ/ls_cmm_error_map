import os, sys
from pathlib import Path
import logging
import tomllib as toml

import cmm_error_map.data_cmpts as dc


logger = logging.getLogger(__name__)

base_folder = Path(
    getattr(sys, "_MEIPASS", Path(__file__).parent.parent.parent.resolve())
)

logger.debug(f"{base_folder=}")


config_folder = base_folder / "config"
machines_toml = config_folder / "machines.toml"
artefacts_toml = config_folder / "artefacts.toml"
# TODO change for pyinstaller
log_folder = base_folder / "logs"


def read_toml(fn, input_type):
    with open(fn, "rb") as f:
        toml_in = toml.load(f)
    toml_in = {key: input_type(**value) for key, value in toml_in.items()}
    return toml_in


cmm_models = read_toml(machines_toml, dc.MachineType)
artefact_models = read_toml(artefacts_toml, dc.ArtefactType)
