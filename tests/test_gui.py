"""
some very simple gui tests via pytest-qt
"""

import pytest

import cmm_error_map.config as cf
from cmm_error_map.ls_cmm_error_map import MainWindow


@pytest.fixture
def app(qtbot):
    widget = MainWindow()
    qtbot.addWidget(widget)
    return widget


def test_app_start(qtbot, app):
    assert app.machine.cmm_model.title == "PMM866"


def test_restore_state(qtbot, app):
    filename = cf.base_folder / "config" / "gui_configs" / "Legex-3-axis-6-prbs.pkl"
    app.restore_filename(filename)
    assert app.machine.cmm_model.title == "Legex574"


@pytest.fixture
def simple_config(qtbot, app):
    filename = cf.base_folder / "config" / "gui_configs" / "Legex-XY-2-prbs.pkl"
    app.restore_filename(filename)
    return app


# def test_save_snapshot(qtbot, simple_config, tmp_path_factory):
#     assert simple_config.machine.cmm_model.title == "Legex574"
#     snapshot_folder = tmp_path_factory.mktemp("snapshot")
