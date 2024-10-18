"""Interactive fitting widget using PyQt6."""

import warnings
import numpy as np
from typing import Dict, Any, Callable
import sys

with warnings.catch_warnings():
    # ipywidgets produces deprecation warnings through use of internal APIs :(
    warnings.simplefilter("ignore")
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib import pyplot as plt
    except ModuleNotFoundError as e:
        e.msg += (
            "\n\nPlease install PyQt6, and matplotlib to enable interactive "
            "outside of Jupyter notebooks."
        )
        raise


def make_widget(
    minuit: Any,
    plot: Callable[..., None],
    kwargs: Dict[str, Any],
    raise_on_exception: bool,
):
    """Make interactive fitting widget."""
    original_values = minuit.values[:]
    original_limits = minuit.limits[:]

    def plot_with_frame(from_fit, report_success):
        trans = plt.gca().transAxes
        try:
            with warnings.catch_warnings():
                minuit.visualize(plot, **kwargs)
        except Exception:
            if raise_on_exception:
                raise

            import traceback

            plt.figtext(
                0,
                0.5,
                traceback.format_exc(limit=-1),
                fontdict={"family": "monospace", "size": "x-small"},
                va="center",
                color="r",
                backgroundcolor="w",
                wrap=True,
            )
            return

        fval = minuit.fmin.fval if from_fit else minuit._fcn(minuit.values)
        plt.text(
            0.05,
            1.05,
            f"FCN = {fval:.3f}",
            transform=trans,
            fontsize="x-large",
        )
        if from_fit and report_success:
            plt.text(
                0.95,
                1.05,
                f"{'success' if minuit.valid and minuit.accurate else 'FAILURE'}",
                transform=trans,
                fontsize="x-large",
                ha="right",
            )

    def fit():
        if algo_choice.value == "Migrad":
            minuit.migrad()
        elif algo_choice.value == "Scipy":
            minuit.scipy()
        elif algo_choice.value == "Simplex":
            minuit.simplex()
            return False
        else:
            assert False  # pragma: no cover, should never happen
        return True

    def do_fit(change):
        report_success = fit()
        for i, x in enumerate(parameters):
            x.reset(minuit.values[i])
        if change is None:
            return report_success
        OnParameterChange()({"from_fit": True, "report_success": report_success})

    def on_update_button_clicked(change):
        for x in parameters:
            x.slider.continuous_update = not x.slider.continuous_update

    def on_reset_button_clicked(change):
        minuit.reset()
        minuit.values = original_values
        minuit.limits = original_limits
        for i, x in enumerate(parameters):
            x.reset(minuit.values[i], minuit.limits[i])
        OnParameterChange()()

    def on_parameter_change(value):
        pass


    class FloatSlider(QtWidgets.QSlider):
        floatValueChanged = QtCore.pyqtSignal(float)

        def __init__(self, label):
            super().__init__(QtCore.Qt.Orientation.Horizontal)
            super().setMinimum(0)
            super().setMaximum(1000)
            super().setValue(500)
            self._min = 0.0
            self._max = 1.0
            self._label = label
            self.valueChanged.connect(self._emit_float_value_changed)

        def _emit_float_value_changed(self, value):
            float_value = self._int_to_float(value)
            self._label.setText(str(float_value))
            self.floatValueChanged.emit(float_value)

        def _int_to_float(self, value):
            return self._min + (value / 1000) * (self._max - self._min)

        def _float_to_int(self, value):
            return int((value - self._min) / (self._max - self._min) * 1000)

        def setMinimum(self, min_value):
            self._min = min_value

        def setMaximum(self, max_value):
            self._max = max_value

        def setValue(self, value):
            super().setValue(self._float_to_int(value))

        def value(self):
            return self._int_to_float(super().value())

        def setSliderPosition(self, value):
            super().setSliderPosition(self._float_to_int(value))


    class Parameter(QtWidgets.QGroupBox):
        def __init__(self, minuit, par) -> None:
            super().__init__(par)
            self.par = par
            # Set up the Qt Widget
            layout = QtWidgets.QGridLayout()
            self.setLayout(layout)
            # Add line edit to display slider value
            self.value_label = QtWidgets.QLabel()
            # Add value slider
            self.slider = FloatSlider(line_edit=self.value_label)
            self.slider.floatValueChanged.connect()
            # Add line edit for changing the limits
            self.vmin = QtWidgets.QLineEdit()
            self.vmin.returnPressed.connect(self.on_limit_changed)
            self.vmax = QtWidgets.QLineEdit()
            self.vmax.returnPressed.connect(self.on_limit_changed)
            # Add buttons
            self.fix = QtWidgets.QPushButton("Fix")
            self.fix.setCheckable(True)
            self.fix.setChecked(minuit.fixed[par])
            self.fix.clicked.connect(self.on_fix_toggled)
            self.fit = QtWidgets.QPushButton("Fit")
            self.fit.setCheckable(True)
            self.fit.setChecked(False)
            self.fit.clicked.connect(self.on_fit_toggled)
            # Add widgets to the layout
            layout.addWidget(self.slider, 0, 0)
            layout.addWidget(self.value_label, 0, 1)
            layout.addWidget(self.vmin, 1, 0)
            layout.addWidget(self.vmax, 1, 1)
            layout.addWidget(self.fix, 2, 0)
            layout.addWidget(self.fit, 2, 1)
            # Add tooltips
            self.slider.setToolTip("Parameter Value")
            self.value_label.setToolTip("Parameter Value")
            self.vmin.setToolTip("Lower Limit")
            self.vmax.setToolTip("Upper Limit")
            self.fix.setToolTip("Fix Parameter")
            self.fit.setToolTip("Fit Parameter")
            # Set initial value and limits
            val = minuit.values[par]
            vmin, vmax = minuit.limits[par]
            step = _guess_initial_step(val, vmin, vmax)
            vmin2 = vmin if np.isfinite(vmin) else val - 100 * step
            vmax2 = vmax if np.isfinite(vmax) else val + 100 * step
            self.slider.setMinimum(vmin2)
            self.slider.setMaximum(vmax2)
            self.slider.setValue(val)
            self.value_label.setText(f"{val:.1g}")
            self.vmin.setText(f"{vmin2:.1g}")
            self.vmax.setText(f"{vmax2:.1g}")

        def on_val_changed(self, val):
            self.minuit.values[self.par] = val
            self.value_label.setText(f"{val:.1g}")
            on_parameter_change()

        def on_limit_changed(self):
            vmin = float(self.vmin.text())
            vmax = float(self.vmax.text())
            self.minuit.limits[self.par] = (vmin, vmax)
            self.slider.setMinimum(vmin)
            self.slider.setMaximum(vmax)
            # Update the slider position
            current_value = self.slider.value()
            if current_value < vmin:
                self.slider.setValue(vmin)
                self.vmin.setText(f"{vmin:.1g}")
                on_parameter_change()
            elif current_value > vmax:
                self.slider.setValue(vmax)
                self.editValue.setText(f"{vmax:.1g}")
                on_parameter_change()
            else:
                self.slider.blockSignals(True)
                self.slider.setValue(vmin)
                self.slider.setValue(current_value)
                self.slider.blockSignals(False)

        def on_fix_toggled(self):
            self.minuit.fixed[self.par] = self.fix.isChecked()
            if self.fix.isChecked():
                self.fit.setChecked(False)

        def on_fit_toggled(self):
            self.slider.setEnabled(not self.fit.isChecked())
            if self.fit.isChecked():
                self.fix.setChecked(False)
            on_parameter_change()

    # Set up the main window
    main_window = QtWidgets.QMainWindow()
    main_window.resize(1600, 1000)
    # Set the global font
    font = QtGui.QFont()
    font.setPointSize(12)
    main_window.setFont(font)
    # Create the central widget
    centralwidget = QtWidgets.QWidget(parent=main_window)
    main_window.setCentralWidget(centralwidget)
    central_layout = QtWidgets.QVBoxLayout(centralwidget)
    # Add tabs for interactive and results
    tab = QtWidgets.QTabWidget(parent=centralwidget)
    interactive_tab = QtWidgets.QWidget()
    tab.addTab(interactive_tab, "")
    results_tab = QtWidgets.QWidget()
    tab.addTab(results_tab, "")
    central_layout.addWidget(tab)
    # Interactive tab
    interactive_layout = QtWidgets.QGridLayout(interactive_tab)
    # Add the plot
    plot_group = QtWidgets.QGroupBox("", parent=interactive_tab)
    sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                       QtWidgets.QSizePolicy.Policy.Expanding)
    sizePolicy.setHeightForWidth(plot_group.sizePolicy().hasHeightForWidth())
    plot_group.setSizePolicy(sizePolicy)
    plot_layout = QtWidgets.QVBoxLayout(plot_group)
    canvas = FigureCanvasQTAgg(Figure())
    ax = canvas.figure.add_subplot(111)
    plot_layout.addWidget(canvas)
    interactive_layout.addWidget(plot_group, 0, 0, 2, 1)
    # Add buttons
    button_group = QtWidgets.QGroupBox("", parent=interactive_tab)
    sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                       QtWidgets.QSizePolicy.Policy.Fixed)
    sizePolicy.setHeightForWidth(button_group.sizePolicy().hasHeightForWidth())
    button_group.setSizePolicy(sizePolicy)
    button_layout = QtWidgets.QHBoxLayout(button_group)
    fit_button = QtWidgets.QPushButton(parent=button_group)
    fit_button.clicked.connect(do_fit)
    button_layout.addWidget(fit_button)
    update_button = QtWidgets.QPushButton(parent=button_group)
    update_button.clicked.connect(on_update_button_clicked)
    button_layout.addWidget(update_button)
    reset_button = QtWidgets.QPushButton(parent=button_group)
    reset_button.clicked.connect(on_reset_button_clicked)
    button_layout.addWidget(reset_button)
    algo_choice = QtWidgets.QComboBox(parent=button_group)
    algo_choice.setStyleSheet("QComboBox { text-align: center; }")
    algo_choice.addItems(["Migrad", "Scipy", "Simplex"])
    button_layout.addWidget(algo_choice)
    interactive_layout.addWidget(button_group, 0, 1, 1, 1)
    # Add the parameters
    parameter_group = QtWidgets.QGroupBox("", parent=interactive_tab)
    sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                       QtWidgets.QSizePolicy.Policy.Expanding)
    sizePolicy.setHeightForWidth(
        parameter_group.sizePolicy().hasHeightForWidth())
    parameter_group.setSizePolicy(sizePolicy)
    parameter_group_layout = QtWidgets.QVBoxLayout(parameter_group)
    scroll_area = QtWidgets.QScrollArea(parent=parameter_group)
    scroll_area.setWidgetResizable(True)
    scroll_area_widget_contents = QtWidgets.QWidget()
    scroll_area_widget_contents.setGeometry(QtCore.QRect(0, 0, 751, 830))
    parameter_layout = QtWidgets.QVBoxLayout(scroll_area_widget_contents)
    scroll_area.setWidget(scroll_area_widget_contents)
    parameter_group_layout.addWidget(scroll_area)
    interactive_layout.addWidget(parameter_group, 1, 1, 1, 1)
    # Results tab
    results_layout = QtWidgets.QVBoxLayout(results_tab)
    results_text = QtWidgets.QPlainTextEdit(parent=results_tab)
    font = QtGui.QFont()
    font.setFamily("FreeMono")
    results_text.setFont(font)
    results_text.setReadOnly(True)
    results_layout.addWidget(results_text)

    parameters = [Parameter(minuit, par) for par in minuit.parameters]


def _make_finite(x: float) -> float:
    sign = -1 if x < 0 else 1
    if abs(x) == np.inf:
        return sign * sys.float_info.max
    return x


def _guess_initial_step(val: float, vmin: float, vmax: float) -> float:
    if np.isfinite(vmin) and np.isfinite(vmax):
        return 1e-2 * (vmax - vmin)
    return 1e-2


def _round(x: float) -> float:
    return float(f"{x:.1g}")
