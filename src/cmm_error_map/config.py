import sys
from pathlib import Path

import tomllib as toml

import cmm_error_map.data_cmpts as dc


# logger = logging.getLogger(__name__)

# this is for pyinstaller paths
base_folder = Path(
    getattr(sys, "_MEIPASS", Path(__file__).parent.parent.parent.resolve())
)

# logger.debug(f"{base_folder=}")

static_path = base_folder / "static"
fn_icon = static_path / "icon.ico"
validation_path = base_folder / "tests" / "validation_data"
test_configs_path = base_folder / "tests" / "gui_configs"

# look for toml config files in
# base_folder/config - ls_cmm_error_map/config or for pyinstaller  _internal/config
# base_folder
# base_folder.parent - pyinstaller alongside exe

machines_tomls = []
artefacts_tomls = []
for folder in [base_folder / "config", base_folder, base_folder.parent]:
    if (folder / "machines.toml").exists():
        machines_tomls.append(folder / "machines.toml")
    if (folder / "artefacts.toml").exists():
        artefacts_tomls.append(folder / "artefacts.toml")


# log_folder = base_folder / "logs"


def read_toml(fn, input_type):
    with open(fn, "rb") as f:
        toml_in = toml.load(f)
    toml_in = {key: input_type(**value) for key, value in toml_in.items()}
    return toml_in


# combine all found models
# if duplaicte entries the last one found takes precedence
# that is a model in base_folder.parent will replace a model in base_folder/config

cmm_models = {}
for ftoml in machines_tomls:
    models = read_toml(ftoml, dc.MachineType)
    # | is dict union operator
    cmm_models = cmm_models | models

artefact_models = {}
for ftoml in artefacts_tomls:
    models = read_toml(ftoml, dc.ArtefactType)
    artefact_models = artefact_models | models
