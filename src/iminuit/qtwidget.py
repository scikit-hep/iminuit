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

    class Parameter(QtWidgets.QGroupBox):
        def __init__(self, minuit, par, callback):
            super().__init__("")
            self.par = par
            self.callback = callback

            size_policy = QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            self.setSizePolicy(size_policy)
            layout = QtWidgets.QVBoxLayout()
            self.setLayout(layout)

            label = QtWidgets.QLabel(par, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(QtCore.QSize(50, 0))
            self.value_label = QtWidgets.QLabel(
                alignment=QtCore.Qt.AlignmentFlag.AlignCenter
            )
            self.value_label.setMinimumSize(QtCore.QSize(50, 0))
            self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.slider.setMinimum(0)
            self.slider.setMaximum(int(1e8))
            self.tmin = QtWidgets.QDoubleSpinBox(
                alignment=QtCore.Qt.AlignmentFlag.AlignCenter
            )
            self.tmin.setRange(_make_finite(-np.inf), _make_finite(np.inf))
            self.tmax = QtWidgets.QDoubleSpinBox(
                alignment=QtCore.Qt.AlignmentFlag.AlignCenter
            )
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
                QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
            )
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

            self.reset(minuit.values[par], limits=minuit.limits[par])

            self.tmin.setSingleStep(1e-1 * (self.vmax - self.vmin))
            self.tmax.setSingleStep(1e-1 * (self.vmax - self.vmin))
            self.tmin.setMinimum(_make_finite(minuit.limits[par][0]))
            self.tmax.setMaximum(_make_finite(minuit.limits[par][1]))

            self.slider.valueChanged.connect(self.on_val_changed)
            self.fix.clicked.connect(self.on_fix_toggled)
            self.tmin.valueChanged.connect(self.on_min_changed)
            self.tmax.valueChanged.connect(self.on_max_changed)
            self.fit.clicked.connect(self.on_fit_toggled)

        def _int_to_float(self, value):
            return self.vmin + (value / 1e8) * (self.vmax - self.vmin)

        def _float_to_int(self, value):
            return int((value - self.vmin) / (self.vmax - self.vmin) * 1e8)

        def on_val_changed(self, val):
            val = self._int_to_float(val)
            self.value_label.setText(f"{val:.3g}")
            minuit.values[self.par] = val
            self.callback()

        def on_min_changed(self):
            tmin = self.tmin.value()
            if tmin >= self.vmax:
                self.tmin.blockSignals(True)
                self.tmin.setValue(self.vmin)
                self.tmin.blockSignals(False)
                return
            self.vmin = tmin
            self.slider.blockSignals(True)
            if tmin > self.val:
                self.val = tmin
                minuit.values[self.par] = tmin
                self.slider.setValue(0)
                self.value_label.setText(f"{self.val:.3g}")
                self.callback()
            else:
                self.slider.setValue(self._float_to_int(self.val))
            self.slider.blockSignals(False)
            lim = minuit.limits[self.par]
            minuit.limits[self.par] = (tmin, lim[1])

        def on_max_changed(self):
            tmax = self.tmax.value()
            if tmax <= self.tmin.value():
                self.tmax.blockSignals(True)
                self.tmax.setValue(self.vmax)
                self.tmax.blockSignals(False)
                return
            self.vmax = tmax
            self.slider.blockSignals(True)
            if tmax < self.val:
                self.val = tmax
                minuit.values[self.par] = tmax
                self.slider.setValue(int(1e8))
                self.value_label.setText(f"{self.val:.3g}")
                self.callback()
            else:
                self.slider.setValue(self._float_to_int(self.val))
            self.slider.blockSignals(False)
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
            self.callback()

        def reset(self, val, limits=None):
            if limits is not None:
                vmin, vmax = limits
                step = _guess_initial_step(val, vmin, vmax)
                self.vmin = vmin if np.isfinite(vmin) else val - 100 * step
                self.vmax = vmax if np.isfinite(vmax) else val + 100 * step
                self.tmin.blockSignals(True)
                self.tmin.setValue(self.vmin)
                self.tmin.blockSignals(False)
                self.tmax.blockSignals(True)
                self.tmax.setValue(self.vmax)
                self.tmax.blockSignals(False)

            self.val = val
            if self.val < self.vmin:
                self.vmin = self.val
                self.tmin.blockSignals(True)
                self.tmin.setValue(self.vmin)
                self.tmin.blockSignals(False)
            elif self.val > self.vmax:
                self.vmax = self.val
                self.tmax.blockSignals(True)
                self.tmax.setValue(self.vmax)
                self.tmax.blockSignals(False)

            self.slider.blockSignals(True)
            self.slider.setValue(self._float_to_int(self.val))
            self.slider.blockSignals(False)
            self.value_label.setText(f"{self.val:.3g}")

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
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            )
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
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            button_group.setSizePolicy(size_policy)
            button_layout = QtWidgets.QHBoxLayout(button_group)
            self.fit_button = QtWidgets.QPushButton("Fit", parent=button_group)
            self.fit_button.setStyleSheet("background-color: #2196F3; color: white")
            self.fit_button.clicked.connect(partial(self.do_fit, plot=True))
            button_layout.addWidget(self.fit_button)
            self.update_button = QtWidgets.QPushButton(
                "Continuous", parent=button_group
            )
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

            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)
            size_policy = QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            )
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

        def on_parameter_change(self, from_fit=False, report_success=False):
            if any(x.fit.isChecked() for x in self.parameters):
                saved = minuit.fixed[:]
                for i, x in enumerate(self.parameters):
                    minuit.fixed[i] = not x.fit.isChecked()
                from_fit = True
                report_success = self.do_fit(plot=False)
                self.results_text.clear()
                self.results_text.setHtml(minuit._repr_html_())
                minuit.fixed = saved
            elif from_fit:
                self.results_text.clear()
                self.results_text.setHtml(minuit._repr_html_())
            else:
                self.results_text.clear()

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
            self.on_parameter_change(from_fit=True, report_success=report_success)

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

    if qt_exec:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        main_window = MainWindow()
        main_window.show()
        app.exec()
    else:
        return MainWindow()


def _make_finite(x: float) -> float:
    sign = -1 if x < 0 else 1
    if abs(x) == np.inf:
        return sign * sys.float_info.max
    return x


def _guess_initial_step(val: float, vmin: float, vmax: float) -> float:
    if np.isfinite(vmin) and np.isfinite(vmax):
        return 1e-2 * (vmax - vmin)
    return 1e-2
