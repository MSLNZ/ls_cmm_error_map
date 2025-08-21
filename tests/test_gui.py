"""
some very simple gui tests via pytest-qt
"""

import pickle

import numpy as np
import numpy.testing as npt
import pytest

import cmm_error_map.config as cf
import cmm_error_map.gui_cmpts as gc
from cmm_error_map.ls_cmm_error_map import MainWindow


@pytest.fixture
def app(qtbot):
    widget = MainWindow()
    qtbot.addWidget(widget)
    filename = cf.test_configs_path / "default_config.pkl"
    widget.restore_state_from_file(filename)
    return widget


def test_app_start(qtbot, app):
    assert app.machine.cmm_model.title == "Legex574"


def test_restore_state_plate(qtbot, app):
    filename = cf.test_configs_path / "Legex-3-axis-6-prbs.pkl"
    app.restore_state_from_file(filename)
    assert app.machine.cmm_model.title == "Legex574"


def test_restore_state_bar(qtbot, app):
    filename = cf.test_configs_path / "stepgauge600.pkl"
    app.restore_state_from_file(filename)
    balls, lines = app.plot_docks["bar1"].plot_data["mmt_control_grp0"]
    assert lines.opts["pen"].color().getRgb() == (255, 0, 255, 255)
    assert balls.opts["symbolPen"].color().getRgb() == (255, 0, 255, 255)


def test_save_state(qtbot, app, tmp_path):
    file_in = cf.test_configs_path / "Legex-XY-2-prbs.pkl"
    app.restore_state_from_file(file_in)
    file_out = tmp_path / "test.pkl"
    app.save_state_to_file(file_out)
    with open(file_out, "rb") as fp:
        state_dict = pickle.load(fp)
    assert state_dict["counts"] == {"probes": 2, "simulations": 2, "snapshots": 0}


def test_set_model_slider(qtbot, app):
    # check the points are nominal before setting the slider
    # this just has the box points
    assert app.machine.boxes["prb_control_grp0"].grid_dev.shape == (3, 240)
    box_dev0 = app.machine.boxes["prb_control_grp0"].grid_dev[1, -1]
    assert pytest.approx(box_dev0, 0.001) == 0.0
    # this has all the connecting lines for the deformed box
    assert app.plot_docks["3D Deformation"].box_lineplot.pos.shape == (1204, 3)
    plot_value0 = app.plot_docks["3D Deformation"].box_lineplot.pos[-1, 1]
    assert pytest.approx(plot_value0, 0.001) == 700.0

    # set the slider
    app.slider_group.child("y_axis", "Ryz").slider.setValue(3.0)

    # check model set
    assert app.machine.model_params["Ryz"][1] == 3.0 * gc.slider_factors["Ryz"]
    # check data for 3d plot has changed
    box_dev1 = app.machine.boxes["prb_control_grp0"].grid_dev[1, -1]
    assert box_dev1 != box_dev0
    # check 3d plot deformed
    plot_value1 = app.plot_docks["3D Deformation"].box_lineplot.pos[-1, 1]
    assert plot_value1 != plot_value0


def test_add_delete_mmt_group(qtbot, app):
    nmmts = len(app.machine.measurements)
    app.add_new_mmt_group(app.mmt_group)
    assert len(app.machine.measurements) == nmmts + 1
    assert len(app.mmt_group.children()) == nmmts + 1
    grp = app.mmt_group.children()[0]
    change = "Delete"
    app.delete_mmt(grp, change)
    assert len(app.machine.measurements) == nmmts
    assert len(app.mmt_group.children()) == nmmts


def test_add_delete_prb_group(qtbot, app):
    nprbs = len(app.machine.probes)
    app.add_new_probe_group(app.prb_group)
    assert len(app.machine.probes) == nprbs + 1
    assert len(app.prb_group.children()) == nprbs + 1
    grp = app.prb_group.children()[1]
    change = "Delete"
    app.delete_prb(grp, change)
    assert len(app.machine.probes) == nprbs
    assert len(app.prb_group.children()) == nprbs


def test_centre_on_cmm(qtbot, app):
    item = app.mmt_group.child("mmt_control_grp0", "grp_location", "centre")
    app.centre_on_cmm(item)
    new_loc = app.machine.measurements["mmt_control_grp0"].transform_mat[:-1, -1]
    npt.assert_allclose(np.array([84.0, 184.0, 200.0]), new_loc, atol=1e-12)


def test_save_snapshot(qtbot, app, tmp_path):
    """
    the actual save stuff has been covered in test_snapshots.py
    """
    file_in = cf.test_configs_path / "Legex-XY-2-prbs.pkl"
    app.restore_state_from_file(file_in)
    app.slider_group.child("y_axis", "Ryz").setValue(3.0)
    mmt = app.machine.measurements["mmt_control_grp0"]

    files = gc.FileSaveTree(
        root_folder=tmp_path,
        folder_prefix="simulation_",
        filenames={
            "snapshot": "snapshot.csv",
            "fulldata": "fulldata.csv",
            "metadata": "metadata.csv",
            "readme": "readme.txt",
        },
    )

    filenames = files.get_filenames()

    now = "2025-02-19 14:23:58.890420"
    readme_text = "I'm a test save of a snapshot"
    app.save_snapshot_to_folder(mmt, filenames, now, readme_text)
    for fn in filenames.values():
        assert fn.exists()


def test_load_snapshot(qtbot, app):
    filename = cf.validation_path / "plate" / "snapshot.csv"
    app.load_snapshot_from_file(filename)
    assert len(app.snapshot_group.children()) == 1
    # check we can plot snapshot in 3D
    app.plot_docks["3D Deformation"].mmts_to_plot.setValue(["Simulation 0", "Plate XZ"])
    balls, lines = app.plot_docks["3D Deformation"].plot_data["snapshot_grp0"]
    npt.assert_allclose(balls.pos[-1, :], np.array([782, 316, 582]), atol=1e-6)


def test_restore_state_with_snapshot(qtbot, app, tmp_path):
    # load a non-default state
    file_in = cf.test_configs_path / "Legex-XY-2-prbs.pkl"
    app.restore_state_from_file(file_in)
    # add a snaphot to it
    file_snap = cf.validation_path / "plate" / "snapshot.csv"
    app.load_snapshot_from_file(file_snap)
    # save this state
    file_out = tmp_path / "test_state_snapshot.pkl"
    app.save_state_to_file(file_out)
    # load the default state
    file_in = cf.test_configs_path / "default_config.pkl"
    app.restore_state_from_file(file_in)
    # load the state saved with a snapshot
    app.restore_state_from_file(file_out)
    # the snapshot is NOT loaded
    assert len(app.snapshot_group.children()) == 0


def test_plot_snapshot_plate(qtbot, app):
    filename = cf.validation_path / "plate" / "snapshot.csv"
    app.load_snapshot_from_file(filename)
    # check we can plot snapshot in 2D
    app.add_new_plot_plate_dock(None)
    app.plot_docks["plate0"].mmts_to_plot.setValue(["Simulation 0", "Plate XZ"])
    balls, lines = app.plot_docks["plate0"].plot_data["snapshot_grp0"]
    npt.assert_allclose(balls.yData[-1], 532.95, atol=1e-6)


def test_snapshot_fixed(qtbot, app):
    filename = cf.validation_path / "plate" / "snapshot.csv"
    app.load_snapshot_from_file(filename)
    app.add_new_plot_plate_dock(None)
    app.plot_docks["plate0"].mmts_to_plot.setValue(["Simulation 0", "Plate XZ"])
    # get the data plotted on the 2D dock
    snap_balls, lines = app.plot_docks["plate0"].plot_data["snapshot_grp0"]
    mmt_balls, lines = app.plot_docks["plate0"].plot_data["mmt_control_grp0"]
    snap_data0 = np.array([snap_balls.xData, snap_balls.yData])
    mmt_data0 = np.array([mmt_balls.xData, mmt_balls.yData])

    # change the model
    app.slider_group.child("y_axis", "Ryz").slider.setValue(3.0)

    snap_data1 = np.array([snap_balls.xData, snap_balls.yData])
    mmt_data1 = np.array([mmt_balls.xData, mmt_balls.yData])
    # check the snapshot didn't respond to model change
    npt.assert_allclose(snap_data0, snap_data1, atol=1e-6)
    # check the  simulation did change
    assert np.any(np.not_equal(mmt_data0, mmt_data1))


def test_simple_save_restore(qtbot, app, tmp_path):
    """
    this was giving errors when done manually - now passes
    """
    # set Probe0 z-length to -250
    app.prb_group.child("prb_control_grp0", "grp_probe_lengths", "Z").setValue(-250.0)
    # centre plate and set z position to 50 mm
    item = app.mmt_group.child("mmt_control_grp0", "grp_location", "centre")
    app.centre_on_cmm(item)
    app.mmt_group.child("mmt_control_grp0", "grp_location", "Z").setValue(50.0)
    # select Simulation 0 in 3D dock
    app.plot_docks["3D Deformation"].plot_controls.child("mmts_to_plot").setValue(
        "Simulation 0"
    )
    # add plate plot and select Simulation 0
    # name must begin with p
    app.add_new_plot_plate_dock(None, name="plate_test")
    app.plot_docks["plate_test"].plot_controls.child("mmts_to_plot").setValue(
        "Simulation 0"
    )
    # add some Rxz deformation
    app.slider_group.child("x_axis", "Rxz").setValue(3.0)
    assert app.mmt_group.child(
        "mmt_control_grp0", "pen", "color"
    ).value().toTuple() == (200, 200, 200, 255)
    # change the color
    app.mmt_group.child("mmt_control_grp0", "pen").setOpts(color="blue")
    assert app.mmt_group.child(
        "mmt_control_grp0", "pen", "color"
    ).value().toTuple() == (0, 0, 255, 255)

    # save config
    file_out = tmp_path / "test.pkl"
    app.save_state_to_file(file_out)
    # restore default
    app.restore_state_from_file(cf.test_configs_path / "default_config.pkl")
    assert app.mmt_group.child(
        "mmt_control_grp0", "pen", "color"
    ).value().toTuple() == (200, 200, 200, 255)

    # restore saved state
    app.restore_state_from_file(file_out)
