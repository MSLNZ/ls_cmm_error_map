"""
test snapshot loading and saving without gui
"""

import copy
from pathlib import Path
from dataclasses import replace

import numpy as np
import numpy.testing as npt
import pytest

import cmm_error_map.config as cf
import cmm_error_map.data_cmpts as dc


# all of these require a Measurement and a file path and a Machine


@pytest.fixture
def single_mmt():
    ballspacing = 133.0
    x0, y0, z0 = 250.0, 50.0, 50.0
    transform_mat_xz = dc.matrix_from_vectors(
        [x0, y0 + 2.0 * ballspacing, z0],
        [90.0, 0.0, 0.0],
    )
    xt_xz, yt_xz, zt_xz = 0.0, 130.0, -243.4852
    prb_xz = dc.Probe(title="P0", name="p0", length=np.array([xt_xz, yt_xz, zt_xz]))
    mmt = dc.Measurement(
        title="Plate XZ",
        name="mmt_01",
        artefact=cf.artefact_models["KOBA 0620"],
        transform_mat=transform_mat_xz,
        probe=prb_xz,
        cmm_nominal=None,
        cmm_dev=None,
        mmt_nominal=None,
        mmt_dev=None,
    )
    cmm = dc.pmm_866
    model_params = dc.model_parameters_test.copy()
    mmt.recalculate(model_params, cmm.cmm_model)
    return mmt


def test_short_header(single_mmt):
    now = "2025-02-19 14:23:58.890420"
    header_calc = dc.short_header(single_mmt, now)
    header = (
        "save time,2025-02-19 14:23:58.890420\n"
        "title,Plate XZ\n"
        "name,mmt_01\n"
        "artefact.title,KOBA 0620\n"
        "artefact.nballs,5,5 \n"
        "artefact.ball_spacing,133.0\n"
        "location,250.0, 316.0, 50.0\n"
        "rotation/deg,89.99999999999999, 0.0, 0.0\n"
    )
    assert header_calc == header


def test_mmt_snapshot_to_csv(single_mmt, tmp_path_factory):
    fp_test = tmp_path_factory.mktemp("snapshot") / "snapshot.csv"
    now = "2025-02-19 14:23:58.890420"
    dc.mmt_snapshot_to_csv(fp_test, single_mmt, now)

    assert fp_test.exists()

    with open(fp_test, "r") as f:
        actual_lines = f.readlines()
    with open(cf.validation_path / "snapshot.csv", "r") as f:
        expected_lines = f.readlines()
    assert actual_lines == expected_lines


def test_mmt_full_data_to_csv(single_mmt, tmp_path_factory):
    fp_test = tmp_path_factory.mktemp("snapshot") / "fulldata.csv"
    now = "2025-02-19 14:23:58.890420"
    dc.mmt_full_data_to_csv(fp_test, single_mmt, now)

    assert fp_test.exists()

    with open(fp_test, "r") as f:
        actual_lines = f.readlines()
    with open(cf.validation_path / "fulldata.csv", "r") as f:
        expected_lines = f.readlines()
    assert actual_lines == expected_lines


def test_mmt_metadata_to_csv(single_mmt, tmp_path_factory):
    fp_test = tmp_path_factory.mktemp("snapshot") / "metadata.csv"
    now = "2025-02-19 14:23:58.890420"
    cmm = dc.pmm_866
    dc.mmt_metadata_to_csv(fp_test, single_mmt, cmm, now)

    assert fp_test.exists()
    with open(fp_test, "r") as f:
        actual_lines = f.readlines()
    with open(cf.validation_path / "metadata.csv", "r") as f:
        expected_lines = f.readlines()
    assert actual_lines == expected_lines


def test_mmt_from_snapshot_csv(single_mmt, tmp_path_factory):
    fp_test = tmp_path_factory.mktemp("snapshot") / "snapshot.csv"
    now = "2025-02-19 14:23:58.890420"
    dc.mmt_snapshot_to_csv(fp_test, single_mmt, now)
    mmt_fp = dc.mmt_from_snapshot_csv(fp_test)
    mmt_cmp = copy.deepcopy(single_mmt)
    # set mmt_cmp probe and transform to that of mmt_fp as we don't save and load probe
    # as a snapshot is fixed
    mmt_cmp.probe = mmt_fp.probe
    mmt_cmp.transform_mat = np.identity(4)
    mmt_cmp.cmm_dev = mmt_fp.cmm_dev
    mmt_cmp.fixed = True

    assert mmt_fp == mmt_cmp

    """
    seeing we've written Measurement.__eq__
    we need to test it!
    """


def test_mmt_eq(single_mmt):
    mmt1 = single_mmt
    mmt2 = copy.deepcopy(single_mmt)
    assert mmt1 == mmt2


def test_mmt_not_eq_str(single_mmt):
    mmt1 = single_mmt
    mmt2 = copy.deepcopy(single_mmt)
    mmt2.title = "wrong"
    assert mmt1 != mmt2


def test_mmt_not_eq_array(single_mmt):
    mmt1 = single_mmt
    mmt2 = copy.deepcopy(single_mmt)
    mmt2.cmm_dev[1, 1] = 999
    assert mmt1 != mmt2


def test_mmt_not_eq_probe(single_mmt):
    mmt1 = single_mmt
    mmt2 = copy.deepcopy(single_mmt)
    mmt2.probe.length[0] = 999
    assert mmt1 != mmt2


def test_mmt_not_eq_artefact(single_mmt):
    mmt1 = single_mmt
    mmt2 = copy.deepcopy(single_mmt)
    mmt2.artefact = cf.artefact_models["KOBA 0320"]
    assert mmt1 != mmt2
