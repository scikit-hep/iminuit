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


    class FloatSlider(QtWidgets.QSlider):
        floatValueChanged = QtCore.pyqtSignal(float)

        def __init__(self):
            super().__init__(QtCore.Qt.Orientation.Horizontal)
            super().setMinimum(0)
            super().setMaximum(10000)
            super().setValue(5000)
            self._min = 0.0
            self._max = 1.0
            self.valueChanged.connect(self._emit_float_value_changed)

        def _emit_float_value_changed(self, value):
            float_value = self._int_to_float(value)
            self.floatValueChanged.emit(float_value)

        def _int_to_float(self, value):
            return self._min + (value / 10000) * (self._max - self._min)

        def _float_to_int(self, value):
            return int((value - self._min) / (self._max - self._min) * 10000)

        def setMinimum(self, min_value):
            val = self.value()
            if val <= min_value:
                val = min_value
            self._min = min_value
            self.setValue(val)

        def setMaximum(self, max_value):
            val = self.value()
            if val >= max_value:
                val = max_value
            self._max = max_value
            self.setValue(val)

        def setValue(self, value):
            super().setValue(self._float_to_int(value))

        def value(self):
            return self._int_to_float(super().value())


    class Parameter(QtWidgets.QGroupBox):
        def __init__(self, minuit, par, callback):
            super().__init__("")
            self.par = par
            self.callback = callback
            self.minuit = minuit
            # Set the size policy of the group box
            sizePolicy = QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                QtWidgets.QSizePolicy.Policy.Fixed)
            self.setSizePolicy(sizePolicy)
            # Set up the Qt Widget
            layout = QtWidgets.QVBoxLayout()
            self.setLayout(layout)
            # Add label
            label = QtWidgets.QLabel(
                par, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(QtCore.QSize(50, 0))
            # Add label to display slider value
            self.value_label = QtWidgets.QLabel(
                alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self.value_label.setMinimumSize(QtCore.QSize(50, 0))
            # Add value slider
            self.slider = FloatSlider()
            # Add spin boxes for changing the limits
            self.tmin = QtWidgets.QDoubleSpinBox(
                alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self.tmin.setRange(_make_finite(-np.inf), _make_finite(np.inf))
            self.tmax = QtWidgets.QDoubleSpinBox(
                alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self.tmax.setRange(_make_finite(-np.inf), _make_finite(np.inf))
            sizePolicy = QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                QtWidgets.QSizePolicy.Policy.Fixed)
            self.tmin.setSizePolicy(sizePolicy)
            self.tmax.setSizePolicy(sizePolicy)
            # Add buttons
            self.fix = QtWidgets.QPushButton("Fix")
            self.fix.setCheckable(True)
            self.fix.setChecked(minuit.fixed[par])
            self.fit = QtWidgets.QPushButton("Fit")
            self.fit.setCheckable(True)
            self.fit.setChecked(False)
            sizePolicy = QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed)
            self.fix.setSizePolicy(sizePolicy)
            self.fit.setSizePolicy(sizePolicy)
            # Add widgets to the layout
            layout1 = QtWidgets.QHBoxLayout()
            layout.addLayout(layout1)
            layout1.addWidget(label)
            layout1.addWidget(self.slider)
            layout1.addWidget(self.value_label)
            layout1.addWidget(self.fix)
            layout2 = QtWidgets.QHBoxLayout()
            layout.addLayout(layout2)
            layout2.addWidget(self.tmin)
            layout2.addWidget(self.tmax)
            layout2.addWidget(self.fit)
            # Add tooltips
            self.slider.setToolTip("Parameter Value")
            self.value_label.setToolTip("Parameter Value")
            self.tmin.setToolTip("Lower Limit")
            self.tmax.setToolTip("Upper Limit")
            self.fix.setToolTip("Fix Parameter")
            self.fit.setToolTip("Fit Parameter")
            # Set initial value and limits
            val = minuit.values[par]
            vmin, vmax = minuit.limits[par]
            self.step = _guess_initial_step(val, vmin, vmax)
            vmin2 = vmin if np.isfinite(vmin) else val - 100 * self.step
            vmax2 = vmax if np.isfinite(vmax) else val + 100 * self.step
            # Set up the spin boxes
            self.tmin.setValue(vmin2)
            self.tmin.setSingleStep(1e-1 * (vmax2 - vmin2))
            self.tmax.setValue(vmax2)
            self.tmax.setSingleStep(1e-1 * (vmax2 - vmin2))
            # Set up the slider
            self.slider.setMinimum(vmin2)
            self.slider.setMaximum(vmax2)
            self.slider.setValue(val)
            self.value_label.setText(f"{val:.3g}")
            # Set limits for the spin boxes
            self.tmin.setMinimum(_make_finite(vmin))
            self.tmax.setMaximum(_make_finite(vmax))
            # Connect signals
            self.slider.floatValueChanged.connect(self.on_val_change)
            self.fix.clicked.connect(self.on_fix_toggled)
            self.tmin.valueChanged.connect(self.on_min_change)
            self.tmax.valueChanged.connect(self.on_max_change)
            self.fit.clicked.connect(self.on_fit_toggled)

        def on_val_change(self, val):
            print("val change", val)
            self.minuit.values[self.par] = val
            self.value_label.setText(f"{val:.3g}")
            self.callback()

        def on_min_change(self):
            tmin = self.tmin.value()
            if tmin >= self.tmax.value():
                self.tmin.setValue(self.minuit.limits[self.par][0])
                return
            self.slider.setMinimum(tmin)
            lim = self.minuit.limits[self.par]
            minuit.limits[self.par] = (tmin, lim[1])

        def on_max_change(self):
            tmax = self.tmax.value()
            if tmax <= self.tmin.value():
                self.tmax.setValue(self.minuit.limits[self.par][1])
                return
            self.slider.setMaximum(tmax)
            lim = self.minuit.limits[self.par]
            minuit.limits[self.par] = (lim[0], tmax)

        def on_fix_toggled(self):
            print("fix toggled")
            self.minuit.fixed[self.par] = self.fix.isChecked()
            if self.fix.isChecked():
                self.fit.setChecked(False)

        def on_fit_toggled(self):
            self.slider.setEnabled(not self.fit.isChecked())
            if self.fit.isChecked():
                self.fix.setChecked(False)
                self.on_fix_toggled()
            self.callback()

        def reset(self, val, limits=None):
            self.slider.blockSignals(True)
            self.slider.setValue(val)
            self.value_label.setText(f"{val:.3g}")
            if limits:
                self.slider.setMinimum(limits[0])
                self.slider.setMaximum(limits[1])
                self.tmin.setValue(limits[0])
                self.tmax.setValue(limits[1])
            self.slider.blockSignals(False)


    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.resize(1200, 600)
            # Set the global font
            font = QtGui.QFont()
            font.setPointSize(12)
            self.setFont(font)
            # Create the central widget
            centralwidget = QtWidgets.QWidget(parent=self)
            self.setCentralWidget(centralwidget)
            central_layout = QtWidgets.QVBoxLayout(centralwidget)
            # Add tabs for interactive and results
            tab = QtWidgets.QTabWidget(parent=centralwidget)
            interactive_tab = QtWidgets.QWidget()
            tab.addTab(interactive_tab, "Interactive")
            results_tab = QtWidgets.QWidget()
            tab.addTab(results_tab, "Results")
            central_layout.addWidget(tab)
            # Interactive tab
            interactive_layout = QtWidgets.QGridLayout(interactive_tab)
            # Add the plot
            plot_group = QtWidgets.QGroupBox("", parent=interactive_tab)
            sizePolicy = QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                QtWidgets.QSizePolicy.Policy.MinimumExpanding)
            plot_group.setSizePolicy(sizePolicy)
            plot_layout = QtWidgets.QVBoxLayout(plot_group)
            fig, self.ax = plt.subplots()
            self.canvas = FigureCanvasQTAgg(fig)
            plot_layout.addWidget(self.canvas)
            plot_layout.addStretch()
            interactive_layout.addWidget(plot_group, 0, 0, 2, 1)
            # Add buttons
            button_group = QtWidgets.QGroupBox("", parent=interactive_tab)
            sizePolicy = QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed)
            button_group.setSizePolicy(sizePolicy)
            button_layout = QtWidgets.QHBoxLayout(button_group)
            self.fit_button = QtWidgets.QPushButton("Fit", parent=button_group)
            self.fit_button.setStyleSheet("background-color: #2196F3; color: white")
            self.fit_button.clicked.connect(self.do_fit)
            button_layout.addWidget(self.fit_button)
            self.update_button = QtWidgets.QPushButton("Continuous", parent=button_group)
            self.update_button.setCheckable(True)
            self.update_button.setChecked(True)
            self.update_button.clicked.connect(self.on_update_button_clicked)
            button_layout.addWidget(self.update_button)
            self.reset_button = QtWidgets.QPushButton("Reset", parent=button_group)
            self.reset_button.setStyleSheet("background-color: #F44336; color: white")
            self.reset_button.clicked.connect(self.on_reset_button_clicked)
            button_layout.addWidget(self.reset_button)
            self.algo_choice = QtWidgets.QComboBox(parent=button_group)
            self.algo_choice.setStyleSheet("QComboBox { text-align: center; }")
            self.algo_choice.addItems(["Migrad", "Scipy", "Simplex"])
            button_layout.addWidget(self.algo_choice)
            interactive_layout.addWidget(button_group, 0, 1, 1, 1)
            # Add the parameters
            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)
            sizePolicy = QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                QtWidgets.QSizePolicy.Policy.MinimumExpanding)
            scroll_area.setSizePolicy(sizePolicy)
            scroll_area_widget_contents = QtWidgets.QWidget()
            parameter_layout = QtWidgets.QVBoxLayout(scroll_area_widget_contents)
            scroll_area.setWidget(scroll_area_widget_contents)
            interactive_layout.addWidget(scroll_area, 1, 1, 1, 1)
            self.parameters = []
            for par in minuit.parameters:
                parameter = Parameter(minuit, par, self.on_parameter_change)
                self.parameters.append(parameter)
                parameter_layout.addWidget(parameter)
            parameter_layout.addStretch()
            # Results tab
            results_layout = QtWidgets.QVBoxLayout(results_tab)
            results_text = QtWidgets.QPlainTextEdit(parent=results_tab)
            font = QtGui.QFont()
            font.setFamily("FreeMono")
            results_text.setFont(font)
            results_text.setReadOnly(True)
            results_layout.addWidget(results_text)
            # Remember the original values and limits
            self.original_values = minuit.values[:]
            self.original_limits = minuit.limits[:]
            # Set the initial plot
            self.plot_with_frame(from_fit=False, report_success=True)

        def fit(self):
            if self.algo_choice.currentText() == "Migrad":
                minuit.migrad()
            elif self.algo_choice.currentText() == "Scipy":
                minuit.scipy()
            elif self.algo_choice.currentText() == "Simplex":
                minuit.simplex()
                return False
            else:
                assert False  # pragma: no cover, should never happen
            return True

        def on_parameter_change(self, from_fit=False,
                                report_success=False):
            if not from_fit:
                if any(x.fit.isChecked() for x in self.parameters):
                    saved = minuit.fixed[:]
                    for i, x in enumerate(self.parameters):
                        minuit.fixed[i] = not x.fit.isChecked()
                    from_fit = True
                    report_success = self.do_fit(plot=False)
                    minuit.fixed = saved

            self.canvas.figure.clear()
            self.plot_with_frame(from_fit, report_success)
            self.canvas.draw_idle()  

        def do_fit(self, plot=True):
            report_success = self.fit()
            for i, x in enumerate(self.parameters):
                x.reset(minuit.values[i])
            if not plot:
                return report_success
            self.on_parameter_change(
                from_fit=True, report_success=report_success)

        def on_update_button_clicked(self):
            for x in self.parameters:
                x.slider.setTracking(self.update_button.isChecked())

        def on_reset_button_clicked(self):
            minuit.reset()
            minuit.values = self.original_values
            minuit.limits = self.original_limits
            for i, x in enumerate(self.parameters):
                x.reset(minuit.values[i], minuit.limits[i])
            self.on_parameter_change()

        def plot_with_frame(self, from_fit, report_success):
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


    # Set up the Qt application
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec()


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
