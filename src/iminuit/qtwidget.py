"""Interactive fitting widget using PySide6."""

from typing import Dict, Any, Callable, Sequence, Tuple
from . import widget

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib import pyplot as plt
except ModuleNotFoundError as e:
    e.msg += (
        "\n\nPlease install PySide6, and matplotlib to enable interactive "
        "outside of Jupyter notebooks."
    )
    raise


class Label(QtWidgets.QLabel, widget.Label):
    def __init__(self, text: str):
        super().__init__(text)

    def set_text(self, text):
        return super().setText(text)


class SizePolicyMixin(widget.SizePolicyMixin):
    @staticmethod
    def _to_policy(policy: str):
        if policy == "minimum_expanding":
            return QtWidgets.QSizePolicy.Policy.MinimumExpanding
        elif policy == "minimum":
            return QtWidgets.QSizePolicy.Policy.Minimum
        elif policy == "fixed":
            return QtWidgets.QSizePolicy.Policy.Fixed
        assert False

    def set_size_policy(self, hpolicy: str, vpolicy: str):
        self.setSizePolicy(self._to_policy(hpolicy), self._to_polciy(vpolicy))


class Container(widget.Container):
    def set_layout(self, layout):
        return self.setLayout(layout)


class Button(QtWidgets.QPushButton, widget.Button, SizePolicyMixin):
    def __init__(self, text: str):
        super().__init__(text)


class CheckButton(QtWidgets.QPushButton, widget.CheckButton, SizePolicyMixin):
    def __init__(self, text: str, checked: bool):
        super().__init__(text)
        super().setChecked(checked)

    def set_checked(self, checked):
        return super().setChecked(checked)


class ComboBox(QtWidgets.QComboBox, widget.ComboBox, SizePolicyMixin):
    def __init__(self, choices: Sequence[str]):
        super().__init__()
        for c in choices:
            super().addItem(c)

    def text(self):
        return super().currentText()


class Slider(QtWidgets.QSlider, widget.Slider, SizePolicyMixin):
    MAX: int = 100000000

    def __init__(self, vmin: float, vmax: float):
        super().__init__()
        super().setMinimum(0)
        super().setMaximum(self.MAX)

    def _int_to_float(self, value):
        return self.vmin + (value / self.MAX) * (self.vmax - self.vmin)

    def _float_to_int(self, value):
        return int((value - self.vmin) / (self.vmax - self.vmin) * self.MAX)

    def set_value(self, value: float):
        super().setValue(self._float_to_int(value))

    def connect(self, callback):
        self.valueChanged.connect(lambda value: callback(self._int_to_float(value)))


class DoubleSpinBox(QtWidgets.QDoubleSpinBox, widget.DoubleSpinBox, SizePolicyMixin):
    def __init__(
        self,
        step: float,
        decimals: int,
        range: Tuple[float, float],
    ):
        super().__init__(
            decimals=decimals, singleStep=step, minimum=range[0], maximum=range[1]
        )


class HLayout(QtWidgets.QHBoxLayout, widget.HLayout):
    def __init__(self, *args):
        for arg in args:
            if isinstance(arg, (widget.HLayout, widget.VLayout)):
                super().addLayout(arg)
            else:
                super().addWidget(arg)


class VLayout(QtWidgets.QVBoxLayout, widget.VLayout):
    def __init__(self, *args):
        for arg in args:
            if isinstance(arg, (widget.HLayout, widget.VLayout)):
                super().addLayout(arg)
            else:
                super().addWidget(arg)


class MainWidget(QtWidgets.QWidget, widget.MainWidget, Container):
    def set_window_title(self, text: str):
        super().setWindowTitle(text)

    def set_font_size(self, points: int):
        font = QtGui.QFont()
        font.setPointSize(points)
        super().setFont(font)


class HtmlView(QtWidgets.QTextEdit, widget.HtmlView, SizePolicyMixin):
    def set_html(self, text: str):
        return super().setHtml(text)


class GroupBox(QtWidgets.QGroupBox, widget.GroupBox, Container, SizePolicyMixin):
    def __init__(self, title: str = ""):
        super().__init__(self, title=title)


class ScrollArea(QtWidgets.QScrollArea, widget.ScrollArea, SizePolicyMixin):
    def __init__(self):
        super().__init__(self)
        self.child_widget = QtWidgets.QWidget()
        super().setWidget(self.child_widget)

    def set_layout(self, layout):
        return self.child_widget.setLayout(layout)


class Backend:
    Label: Label
    Button: Button
    CheckButton: CheckButton
    Slider: Slider
    DoubleSpinBox: DoubleSpinBox
    HLayout: HLayout
    VLayout: VLayout
    MainWidget: MainWidget
    HtmlView: HtmlView
    GroupBox: GroupBox
    ComboBox: ComboBox
    ScrollArea: ScrollArea


def make_widget(
    minuit: Any,
    plot: Callable[..., None],
    kwargs: Dict[str, Any],
    raise_on_exception: bool,
    run_event_loop: bool = True,
):
    """Make interactive fitting widget."""
    main_widget = widget.make_main_widget(
        backend, minuit, plot, kwargs, raise_on_exception
    )

    if run_event_loop:  # pragma: no cover, should not be executed in tests
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        main_widget.show()
        app.exec()  # this blocks the main thread
    else:
        return main_widget
