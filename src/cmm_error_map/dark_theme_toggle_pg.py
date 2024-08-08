import pyqtgraph as pg
from pyqtgraph.Qt.QtWidgets import QMainWindow, QComboBox, QHBoxLayout, QWidget

import qdarktheme

app = pg.mkQApp()
# Apply dark theme.
qdarktheme.setup_theme("light")

main_win = QMainWindow()
combo_box = QComboBox()
combo_box.addItems(qdarktheme.get_themes())
combo_box.currentTextChanged.connect(qdarktheme.setup_theme)

layout = QHBoxLayout()
layout.addWidget(combo_box)

central_widget = QWidget()
central_widget.setLayout(layout)
main_win.setCentralWidget(central_widget)

main_win.show()

pg.exec()
