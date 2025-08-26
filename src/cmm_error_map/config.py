import sys
from pathlib import Path

import tomllib as toml

import cmm_error_map.data_cmpts as dc

# import logging
# logger = logging.getLogger(__name__)

# this is for pyinstaller paths using a one folder set up with an `_internal` folder alongside the exe
# the gui_configs  folder and `artefacts.toml` and `machines.toml` should also be copied into this folder
# before zipping for release
# if they're not there the versions in "_internal/src/cmm_error_map/config" will be used instead


pyinstaller_base =  getattr(sys, '_MEIPASS', False)
if pyinstaller_base:
    # run from pyinstaller exe
    # internal will be `_internal` folder in the unzipped release
    internal = Path(pyinstaller_base)
    toml_folders = [internal.parent.resolve(), internal / "src" / "cmm_error_map" / "config"]
    default_config_folders = [internal / "gui_configs", internal / "src" / "cmm_error_map" / "config" / "gui_configs"]
    static_path = internal / "static" 
        
else:
    # local run from python
    code_folder =  Path(__file__).parent.resolve() # folder of this file
    base_folder = code_folder.parent.parent.resolve() # project folder
    toml_folders = [code_folder  / "config"]
    default_config_folders = [code_folder  / "config" / "gui_configs"]

    static_path = base_folder / "static"
    validation_path = base_folder / "tests" / "validation_data"
    test_configs_path = base_folder / "tests" / "gui_configs"

fn_icon = static_path / "icon.ico"

machines_tomls = []
artefacts_tomls = []
for folder in toml_folders:
    if (folder / "machines.toml").exists():
        machines_tomls.append(folder / "machines.toml")
    if (folder / "artefacts.toml").exists():
        artefacts_tomls.append(folder / "artefacts.toml")


# log_folder = base_folder / "logs"

_config_fn = "default_config.pkl"


for folder in default_config_folders:
    fn = folder / _config_fn
    if fn.exists():
        default_config_fn = fn
        break


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

print(f"{toml_folders=}")
print(f"{default_config_folders=}")

print(f"{machines_tomls=}")
print(f"{artefacts_tomls=}")

print(f"{len(cmm_models)=}")
print(f"{len(artefact_models)=}")
