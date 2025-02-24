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


def test_restore_state_plate(qtbot, app):
    filename = cf.test_configs_path / "Legex-3-axis-6-prbs.pkl"
    app.restore_filename(filename)
    assert app.machine.cmm_model.title == "Legex574"


def test_restore_state_bar(qtbot, app):
    filename = cf.test_configs_path / "stepgauge600.pkl"
    app.restore_filename(filename)
    balls, lines = app.plot_docks["bar1"].plot_data["mmt_control_grp0"]
    assert lines.opts["pen"].color().getRgb() == (255, 0, 255, 255)
    assert balls.opts["symbolPen"].color().getRgb() == (255, 0, 255, 255)


@pytest.fixture
def simple_config(qtbot, app):
    filename = cf.test_configs_path / "Legex-XY-2-prbs.pkl"
    app.restore_filename(filename)
    return app


# def test_save_snapshot(qtbot, simple_config, tmp_path_factory):
#     assert simple_config.machine.cmm_model.title == "Legex574"
#     snapshot_folder = tmp_path_factory.mktemp("snapshot")
