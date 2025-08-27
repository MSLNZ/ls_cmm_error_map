import sys
from pathlib import Path
import time

import logging
from logging.handlers import TimedRotatingFileHandler
import tempfile


import tomllib as toml

import cmm_error_map.data_cmpts as dc

# import logging
# logger = logging.getLogger(__name__)

# this is for pyinstaller paths using a one folder set up with an `_internal` folder alongside the exe
# the gui_configs  folder and `artefacts.toml` and `machines.toml` should also be copied into this folder (for user editing)
# before zipping for release
# if they're not there the versions in "_internal/src/cmm_error_map/config" will be used instead

log_folder = tempfile.gettempdir()
pyinstaller_base =  getattr(sys, '_MEIPASS', False)
if pyinstaller_base:
    # run from pyinstaller exe
    # internal will be `_internal` folder in the unzipped release
    internal = Path(pyinstaller_base)
    toml_folders = [internal.parent.resolve(), internal / "src" / "cmm_error_map" / "config"]
    default_config_folders = [internal / "gui_configs", internal / "src" / "cmm_error_map" / "config" / "gui_configs"]
    static_path = internal / "static" 
    if (internal.parent / "logs").exists():
        # if a "logs" folder exists alongside exe use it
        log_folder = (internal.parent / "logs").resolve()
        
else:
    # local run from python
    code_folder =  Path(__file__).parent.resolve() # folder of this file
    base_folder = code_folder.parent.parent.resolve() # project folder
    toml_folders = [code_folder  / "config"]
    default_config_folders = [code_folder  / "config" / "gui_configs"]
    static_path = base_folder / "static"
    # only needed for tests
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




# logging
def config_log(path):
    """
    """
    logger = logging.getLogger("cmm_error_map")
    logger.setLevel(logging.DEBUG)

    # Create a console handler and a (rotating) file handler
    console_handler = logging.StreamHandler()
    file_handler = TimedRotatingFileHandler(path,
                                       when="midnight",
                                       utc=True,
                                       backupCount=7)
    # Set level for handlers
    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)

    # create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s %(levelname)8s: %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logging.Formatter.converter = time.gmtime

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = config_log(Path(log_folder) / "cmm_error_map.log")

def exception_hook(exc_type, exc_value, exc_traceback):
    """
    custom exeption hook to send tracebacks to file
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = exception_hook

logger.info(f"{toml_folders=}")
logger.info(f"{default_config_folders=}")

logger.info(f"{machines_tomls=}")
logger.info(f"{artefacts_tomls=}")

logger.info(f"{len(cmm_models)=}")
logger.info(f"{len(artefact_models)=}")