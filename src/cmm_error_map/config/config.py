from pathlib import Path
import tomllib as toml
import cmm_error_map.data_cmpts as dc

config_folder = Path(__file__).parent.resolve()
machines_toml = config_folder / "machines.toml"


def get_cmm_models():
    with open(machines_toml, "rb") as f:
        machines = toml.load(f)
    machines = {key: dc.MachineType(**value) for key, value in machines.items()}
    return machines


cmm_models = get_cmm_models()
