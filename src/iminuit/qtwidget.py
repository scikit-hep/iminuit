"""Interactive fitting widget using PyQt6."""

import warnings
import numpy as np
from typing import Dict, Any, Callable
import sys
from functools import partial

try:
    from PyQt6 import QtCore, QtGui, QtWidgets
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
    qt_exec: bool,
):
    """Make interactive fitting widget."""
    original_values = minuit.values[:]
    original_limits = minuit.limits[:]

    class FloatSlider(QtWidgets.QSlider):
        # Qt sadly does not have a float slider, so we have to
        # implement one ourselves.
        floatValueChanged = QtCore.pyqtSignal(float)

        def __init__(self, label):
            super().__init__(QtCore.Qt.Orientation.Horizontal)
            super().setMinimum(0)
            super().setMaximum(int(1e8))
            super().setValue(int(5e7))
            self._min = 0.0
            self._max = 1.0
            self._value = 0.5
            self._label = label
            self.valueChanged.connect(self._emit_float_value_changed)

        def _emit_float_value_changed(self, value=None):
            if value is not None:
                self._value = self._int_to_float(value)
            self._label.setText(f"{self._value:.3g}")
            self.floatValueChanged.emit(self._value)

        def _int_to_float(self, value):
            return self._min + (value / 1e8) * (self._max - self._min)

        def _float_to_int(self, value):
            return int((value - self._min) / (self._max - self._min) * 1e8)

        def setMinimum(self, min_value):
            if self._max <= min_value:
                return
            self._min = min_value
            self.setValue(self._value)

        def setMaximum(self, max_value):
            if self._min >= max_value:
                return
            self._max = max_value
            self.setValue(self._value)

        def setValue(self, value):
            if value < self._min:
                self._value = self._min
                super().setValue(0)
                self._emit_float_value_changed()
            elif value > self._max:
                self._value = self._max
                super().setValue(int(1e8))
                self._emit_float_value_changed()
            else:
                self._value = value
                self.blockSignals(True)
                super().setValue(self._float_to_int(value))
                self.blockSignals(False)

        def value(self):
            return self._value

    class Parameter(QtWidgets.QGroupBox):
        def __init__(self, minuit, par, callback):
            super().__init__("")
            self.par = par
            self.callback = callback

            size_policy = QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                QtWidgets.QSizePolicy.Policy.Fixed)
            self.setSizePolicy(size_policy)
            layout = QtWidgets.QVBoxLayout()
            self.setLayout(layout)

            label = QtWidgets.QLabel(
                par, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(QtCore.QSize(50, 0))
            self.value_label = QtWidgets.QLabel(
                alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self.value_label.setMinimumSize(QtCore.QSize(50, 0))
            self.slider = FloatSlider(self.value_label)
            self.tmin = QtWidgets.QDoubleSpinBox(
                alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self.tmin.setRange(_make_finite(-np.inf), _make_finite(np.inf))
            self.tmax = QtWidgets.QDoubleSpinBox(
                alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self.tmax.setRange(_make_finite(-np.inf), _make_finite(np.inf))
            self.tmin.setSizePolicy(size_policy)
            self.tmax.setSizePolicy(size_policy)
            self.fix = QtWidgets.QPushButton("Fix")
            self.fix.setCheckable(True)
            self.fix.setChecked(minuit.fixed[par])
            self.fit = QtWidgets.QPushButton("Fit")
            self.fit.setCheckable(True)
            self.fit.setChecked(False)
            size_policy = QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed)
            self.fix.setSizePolicy(size_policy)
            self.fit.setSizePolicy(size_policy)
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

            val = minuit.values[par]
            vmin, vmax = minuit.limits[par]
            self.step = _guess_initial_step(val, vmin, vmax)
            vmin2 = vmin if np.isfinite(vmin) else val - 100 * self.step
            vmax2 = vmax if np.isfinite(vmax) else val + 100 * self.step
            self.tmin.setSingleStep(1e-1 * (vmax2 - vmin2))
            self.tmax.setSingleStep(1e-1 * (vmax2 - vmin2))
            self.tmin.setMinimum(_make_finite(vmin))
            self.tmax.setMaximum(_make_finite(vmax))
            self.reset(val, limits=(vmin, vmax))

            self.slider.floatValueChanged.connect(self.on_val_change)
            self.fix.clicked.connect(self.on_fix_toggled)
            self.tmin.valueChanged.connect(self.on_min_change)
            self.tmax.valueChanged.connect(self.on_max_change)
            self.fit.clicked.connect(self.on_fit_toggled)

        def on_val_change(self, val):
            minuit.values[self.par] = val
            self.callback()

        def on_min_change(self):
            tmin = self.tmin.value()
            if tmin >= self.tmax.value():
                self.tmin.blockSignals(True)
                self.tmin.setValue(minuit.limits[self.par][0])
                self.tmin.blockSignals(False)
                return
            self.slider.setMinimum(tmin)
            lim = minuit.limits[self.par]
            minuit.limits[self.par] = (tmin, lim[1])

        def on_max_change(self):
            tmax = self.tmax.value()
            if tmax <= self.tmin.value():
                self.tmax.blockSignals(True)
                self.tmax.setValue(minuit.limits[self.par][1])
                self.tmax.blockSignals(False)
                return
            self.slider.setMaximum(tmax)
            lim = minuit.limits[self.par]
            minuit.limits[self.par] = (lim[0], tmax)

        def on_fix_toggled(self):
            minuit.fixed[self.par] = self.fix.isChecked()
            if self.fix.isChecked():
                self.fit.setChecked(False)

        def on_fit_toggled(self):
            self.slider.setEnabled(not self.fit.isChecked())
            if self.fit.isChecked():
                self.fix.setChecked(False)
                minuit.fixed[self.par] = False
            self.callback()

        def reset(self, val, limits=None):
            # Set limits first so that the value won't be changed by the
            # FloatSlider
            if limits is not None:
                vmin, vmax = limits
                vmin = vmin if np.isfinite(vmin) else val - 100 * self.step
                vmax = vmax if np.isfinite(vmax) else val + 100 * self.step
                self.slider.blockSignals(True)
                self.slider.setMinimum(vmin)
                self.slider.blockSignals(True)
                self.slider.setMaximum(vmax)
                self.tmin.blockSignals(True)
                self.tmin.setValue(vmin)
                self.tmin.blockSignals(False)
                self.tmax.blockSignals(True)
                self.tmax.setValue(vmax)
                self.tmax.blockSignals(False)

            self.slider.blockSignals(True)
            self.slider.setValue(val)
            self.value_label.setText(f"{val:.3g}")
            self.slider.blockSignals(False)

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.resize(1200, 600)
            font = QtGui.QFont()
            font.setPointSize(12)
            self.setFont(font)
            centralwidget = QtWidgets.QWidget(parent=self)
            self.setCentralWidget(centralwidget)
            central_layout = QtWidgets.QVBoxLayout(centralwidget)
            tab = QtWidgets.QTabWidget(parent=centralwidget)
            interactive_tab = QtWidgets.QWidget()
            tab.addTab(interactive_tab, "Interactive")
            results_tab = QtWidgets.QWidget()
            tab.addTab(results_tab, "Results")
            central_layout.addWidget(tab)

            interactive_layout = QtWidgets.QGridLayout(interactive_tab)

            plot_group = QtWidgets.QGroupBox("", parent=interactive_tab)
            size_policy = QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                QtWidgets.QSizePolicy.Policy.MinimumExpanding)
            plot_group.setSizePolicy(size_policy)
            plot_layout = QtWidgets.QVBoxLayout(plot_group)
            # Use pyplot here to allow users to use pyplot in the plot
            # function (not recommended / unstable)
            self.fig, ax = plt.subplots()
            self.canvas = FigureCanvasQTAgg(self.fig)
            plot_layout.addWidget(self.canvas)
            plot_layout.addStretch()
            interactive_layout.addWidget(plot_group, 0, 0, 2, 1)
            try:
                plot(minuit.values, fig=self.fig)
                kwargs["fig"] = self.fig
            except Exception:
                pass
            try:
                plot(minuit.values, ax=ax)
                kwargs["ax"] = ax
            except Exception:
                pass
            self.fig_width = self.fig.get_figwidth()

            button_group = QtWidgets.QGroupBox("", parent=interactive_tab)
            size_policy = QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed)
            button_group.setSizePolicy(size_policy)
            button_layout = QtWidgets.QHBoxLayout(button_group)
            self.fit_button = QtWidgets.QPushButton("Fit", parent=button_group)
            self.fit_button.setStyleSheet(
                "background-color: #2196F3; color: white")
            self.fit_button.clicked.connect(partial(self.do_fit, plot=True))
            button_layout.addWidget(self.fit_button)
            self.update_button = QtWidgets.QPushButton(
                "Continuous", parent=button_group)
            self.update_button.setCheckable(True)
            self.update_button.setChecked(True)
            self.update_button.clicked.connect(self.on_update_button_clicked)
            button_layout.addWidget(self.update_button)
            self.reset_button = QtWidgets.QPushButton(
                "Reset", parent=button_group)
            self.reset_button.setStyleSheet(
                "background-color: #F44336; color: white")
            self.reset_button.clicked.connect(self.on_reset_button_clicked)
            button_layout.addWidget(self.reset_button)
            self.algo_choice = QtWidgets.QComboBox(parent=button_group)
            self.algo_choice.setStyleSheet("QComboBox { text-align: center; }")
            self.algo_choice.addItems(["Migrad", "Scipy", "Simplex"])
            button_layout.addWidget(self.algo_choice)
            interactive_layout.addWidget(button_group, 0, 1, 1, 1)

            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)
            size_policy = QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                QtWidgets.QSizePolicy.Policy.MinimumExpanding)
            scroll_area.setSizePolicy(size_policy)
            scroll_area_contents = QtWidgets.QWidget()
            parameter_layout = QtWidgets.QVBoxLayout(scroll_area_contents)
            scroll_area.setWidget(scroll_area_contents)
            interactive_layout.addWidget(scroll_area, 1, 1, 1, 1)
            self.parameters = []
            for par in minuit.parameters:
                parameter = Parameter(minuit, par, self.on_parameter_change)
                self.parameters.append(parameter)
                parameter_layout.addWidget(parameter)
            parameter_layout.addStretch()

            results_layout = QtWidgets.QVBoxLayout(results_tab)
            self.results_text = QtWidgets.QTextEdit(parent=results_tab)
            self.results_text.setReadOnly(True)
            results_layout.addWidget(self.results_text)

            self.plot_with_frame(from_fit=False, report_success=False)

        def plot_with_frame(self, from_fit, report_success):
            self.fig.set_figwidth(self.fig_width)
            try:
                with warnings.catch_warnings():
                    minuit.visualize(plot, **kwargs)
            except Exception:
                if raise_on_exception:
                    raise

                import traceback

                self.fig.text(
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
            self.fig.get_axes()[0].text(
                0.05,
                1.05,
                f"FCN = {fval:.3f}",
                transform=self.fig.get_axes()[0].transAxes,
                fontsize="x-large",
            )
            if from_fit and report_success:
                self.fig.get_axes()[-1].text(
                    0.95,
                    1.05,
                    f"{'success' if minuit.valid and minuit.accurate else 'FAILURE'}",
                    transform=self.fig.get_axes()[-1].transAxes,
                    fontsize="x-large",
                    ha="right",
                )

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
                    self.results_text.clear()
                    self.results_text.setHtml(minuit._repr_html_())
                    minuit.fixed = saved
            else:
                self.results_text.clear()
                self.results_text.setHtml(minuit._repr_html_())

            for ax in self.fig.get_axes():
                ax.clear()
            self.plot_with_frame(from_fit, report_success)
            self.canvas.draw_idle()

        def do_fit(self, plot=True):
            report_success = self.fit()
            for i, x in enumerate(self.parameters):
                x.reset(val=minuit.values[i])
            if not plot:
                return report_success
            self.on_parameter_change(
                from_fit=True, report_success=report_success)

        def on_update_button_clicked(self):
            for x in self.parameters:
                x.slider.setTracking(self.update_button.isChecked())

        def on_reset_button_clicked(self):
            minuit.reset()
            minuit.values = original_values
            minuit.limits = original_limits
            for i, x in enumerate(self.parameters):
                x.reset(val=minuit.values[i], limits=original_limits[i])
            self.on_parameter_change()

    # Set up the Qt application
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    main_window = MainWindow()
    main_window.show()

    if qt_exec:
        app.exec()
    return app, main_window


def _make_finite(x: float) -> float:
    sign = -1 if x < 0 else 1
    if abs(x) == np.inf:
        return sign * sys.float_info.max
    return x


def _guess_initial_step(val: float, vmin: float, vmax: float) -> float:
    if np.isfinite(vmin) and np.isfinite(vmax):
        return 1e-2 * (vmax - vmin)
    return 1e-2
