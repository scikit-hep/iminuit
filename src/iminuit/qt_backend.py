"""Qt Backend."""

from contextlib import contextmanager
from typing import Sequence
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


app = QApplication.instance()
if app is None:
    app = QApplication([])


class Slider(QSlider):
    MAX: int = 100_000_000

    def __init__(self, vmin: float, vmax: float, value: float, on_change: callable):
        super().__init__(Qt.Orientation.Horizontal)
        super().setMinimum(0)
        super().setMaximum(self.MAX)
        super().setValue(value)
        self._vmin = vmin
        self._vdel = vmax - vmin
        super().valueChanged.connect(lambda value: on_change(self._to_float(value)))

    def set_value(self, value: float):
        super().setValue(self._to_int(value))

    def _to_float(self, value: int) -> float:
        return self._vmin + (value / self.MAX) * (self._vdel)

    def _to_int(self, value: float) -> int:
        return int((value - self._vmin) / self._vdel * self.MAX)

    def set_enabled(self, on: bool):
        super().setEnabled(on)


class ComboBox(QComboBox):
    def __init__(self, choices: Sequence[str], value: str, on_change: callable):
        super().__init__()
        for c in choices:
            super().addItem(c)
        super().setCurrentText(value)
        super().currentTextChanged.connect(on_change)

    def text(self):
        return super().currentText()


class SpinBox(QDoubleSpinBox):
    def __init__(
        self,
        decimals: int,
        step: float,
        value: float,
        vmin: float,
        vmax: float,
        on_change: callable,
    ):
        super().__init__(
            value=value,
            decimals=decimals,
            singleStep=step,
            minimum=vmin,
            maximum=vmax,
        )
        super().valueChanged.connect(on_change)

    def set_value(self, value):
        super().setValue(value)

    def set_min(self, value):
        super().setMinimum(value)

    def set_max(self, value):
        super().setMaximum(value)


class Button(QPushButton):
    def __init__(self, label, on_click):
        super().__init__(label)
        super().clicked.connect(on_click)

    def set_style(self, style: str):
        super().setStyleSheet(style)


class ToggleButton(QPushButton):
    def __init__(self, label, on_click, checked=False):
        super().__init__(label)
        super().setCheckable(True)
        super().setChecked(checked)
        super().clicked.connect(lambda: on_click(self.isChecked()))

    def set_checked(self, checked: bool):
        super().setChecked(checked)

    def is_checked(self) -> bool:
        return super().isChecked()


class VLayout(QVBoxLayout):
    def __init__(self, *args):
        super().__init__()
        for arg in args:
            if isinstance(arg, (HLayout, VLayout)):
                super().addLayout(arg)
            else:
                super().addWidget(arg)


class HLayout(QHBoxLayout):
    def __init__(self, *args):
        super().__init__()
        for arg in args:
            if isinstance(arg, (HLayout, VLayout)):
                super().addLayout(arg)
            else:
                super().addWidget(arg)


class MainWidget(QWidget):
    def __init__(self, layout):
        super().__init__()
        font = QFont()
        font.setPointSize(11)
        self.setFont(font)
        self.setWindowTitle("iminuit")
        super().setLayout(layout)

    def render(self):
        super().show()
        app.exec()

    def make_plot_widget(self):
        fig = plt.figure()
        manager = plt.get_current_fig_manager()
        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas.manager = manager

        plot_group = GroupBox(VLayout(self.canvas))
        # plot_group.set_size_policy(
        #     "minimum_expanding",
        #     "minimum_expanding",
        # )
        return plot_group


class Label(QLabel):
    def __init__(self, text, min_width: int = 0):
        super().__init__(text)
        if min_width > 0:
            super().setMinimumWidth(min_width)

    def set_text(self, text):
        super().setText(text)


class ScrollArea(QScrollArea):
    def __init__(self, *args):
        super().__init__()
        super().setContentsMargins(0, 0, 0, 0)
        super().setFrameShape(QFrame.NoFrame)
        super().setWidgetResizable(True)
        child_widget = QWidget(self)
        super().setWidget(child_widget)
        # workaround for sizing issue
        super().setMinimumWidth(max(x.sizeHint().width() for x in args) + 30)

        layout = VLayout(*args)
        layout.setContentsMargins(0, 0, 0, 0)
        # if list is too short to fill full area,
        # add stretch at the end
        layout.addStretch()
        child_widget.setLayout(layout)


class HtmlView(QLabel):
    def set_html(self, html: str):
        return super().setText(html)


class GroupBox(QGroupBox):
    def __init__(self, layout, title: str = ""):
        super().__init__(title=title)
        super().setLayout(layout)


@contextmanager
def signal_block(*widgets):
    for w in widgets:
        w.blockSignals(True)
    yield
    for w in widgets:
        w.blockSignals(False)
