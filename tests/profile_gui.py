import cmm_error_map.config as cf
from cmm_error_map.ls_cmm_error_map import MainWindow


def app(qtbot):
    widget = MainWindow()
    qtbot.addWidget(widget)
    filename = cf.test_configs_path / "default_config.pkl"
    widget.restore_state_from_file(filename)
    return widget


def load_time_flamegraph(app):
    # complicated configs take a longish time to restore

    file_in = cf.test_configs_path / "Legex-3-axis-6-prbs.pkl"
    app.restore_state_from_file(file_in)


if __name__ == "__main__":
    print("hello")
    load_time_flamegraph()
    print("bye")
