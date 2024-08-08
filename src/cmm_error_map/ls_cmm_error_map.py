"""
main gui for cmm error map app
"""

import numpy as np
from pyqtgraph.Qt.QtCore import Qt as qtc

import pyqtgraph.Qt.QtWidgets as qtw
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
import pyqtgraph as pg

import qdarktheme

import cmm_error_map.design_matrix_linear_fixed as design

main_params = [
    {
        "title": "Basic parameter data types",
        "name": "p1",
        "type": "group",
        "children": [
            {
                "name": "Boolean",
                "type": "bool",
                "value": True,
                "tip": "This is a checkbox",
            },
            {
                "name": "str",
                "type": "str",
                "value": "hello",
            },
            {
                "name": "Color",
                "type": "color",
                "value": "#4c0027",
                "tip": "This is a color selector",
            },
            {
                "name": "float",
                "type": "float",
                "value": "1066.3",
                "decimals": 2,
                "limits": [0.01, None],
                "format": "{value:.1f}",
            },
            {
                "name": "list",
                "type": "list",
                "limits": ["A", "B", "C"],
            },
            {
                "name": "check list",
                "type": "checklist",
                "limits": ["A", "B", "C"],
            },
            {
                "name": "select file",
                "type": "file",
                "directory": "C:/",
            },
        ],
    }
]


class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setup_gui()
        self.add_parameter_tree()
        self.make_docks()
        self.add_summary()

    def setup_gui(self):
        self.dock_area = DockArea()

        self.tree = ParameterTree(showHeader=False)
        # self.tree.setContentsMargins(0, 0, 0, 0)
        # self.tree.header().setStretchLastSection(False)
        # self.tree.header().setMinimumSectionSize(170)

        self.summary = qtw.QTextEdit()

        v_split = qtw.QSplitter(qtc.Vertical)
        v_split.addWidget(self.tree)
        v_split.addWidget(self.summary)
        v_split.setSizes([200, 100])

        h_split = qtw.QSplitter(qtc.Horizontal)
        h_split.addWidget(v_split)
        h_split.addWidget(self.dock_area)
        h_split.setSizes([150, 600])

        layout1 = qtw.QHBoxLayout()
        layout1.addWidget(h_split)
        layout1.setContentsMargins(0, 0, 0, 0)
        layout1.setSpacing(20)

        widget = qtw.QWidget()
        widget.setLayout(layout1)
        self.setCentralWidget(widget)

    def add_parameter_tree(self):
        self.param = Parameter.create(name="params", type="group")
        # create sliders
        slider_group = self.param.addChild(dict(type="group", name="Linear Model"))
        for p in design.modelparameters:
            slider_group.addChild(
                dict(
                    type="slider",
                    name=p[0],
                    limits=[-5.0, 5.0],
                    step=0.1,
                    value=0,
                )
            )
        # spacers
        slider_group.insertChild(
            3, dict(type="str", name="spacer", title="", value="", readonly=True)
        )
        self.tree.setParameters(self.param, showTop=False)

    def make_docks(self):
        """
        add a table and a plot in separate docks
        """
        self.table_data, self.table = self.make_table()
        table_dock = Dock("Data Table")
        table_dock.addWidget(self.table)
        self.dock_area.addDock(table_dock)

        self.plot_data, self.plot = self.make_plot()
        plot_dock = Dock("Plot")
        plot_dock.addWidget(self.plot)
        self.dock_area.addDock(plot_dock)

    def make_table(self):
        table = pg.TableWidget(editable=False, sortable=True)
        data = np.array(
            [
                (1, 1.6, "x"),
                (3, 5.4, "y"),
                (8, 12.5, "z"),
                (443, 1e-12, "w"),
            ],
            dtype=[("Column 1", int), ("Column 2", float), ("Column 3", object)],
        )
        table.setData(data)
        return data, table

    def make_plot(self):
        plot_data = np.random.normal(size=100)
        plot = pg.plot(plot_data, title="Simplest possible plotting example")
        return plot_data, plot

    def add_summary(self):
        text = "Summary\n"
        text += "-------\n\n"
        text += "Use this area for summary or user info"
        self.summary.setPlainText(text)


def main():
    _app = pg.mkQApp("CMM Error Map App")
    qdarktheme.setup_theme(
        "dark",
        custom_colors={"primary": "#f07845", "list.alternateBackground": "#202124"},
    )

    w = MainWindow()
    # app.setStyleSheet("QCheckBox { padding: 5px }")
    w.showMaximized()
    w.show()

    pg.exec()


if __name__ == "__main__":
    main()
