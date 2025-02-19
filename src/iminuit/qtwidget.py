"""Interactive fitting widget using PySide6."""

from .util import _widget_guess_initial_step, _make_finite
import warnings
import numpy as np
from typing import Dict, Any, Callable
from contextlib import contextmanager

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


def make_widget(
    minuit: Any,
    plot: Callable[..., None],
    kwargs: Dict[str, Any],
    raise_on_exception: bool,
    run_event_loop: bool = True,
):
    """Make interactive fitting widget."""
    original_values = minuit.values[:]
    original_limits = minuit.limits[:]

    class Parameter(QtWidgets.QGroupBox):
        def __init__(self, minuit, par, callback):
            super().__init__("")
            self.par = par
            self.callback = callback

            vlayout = QtWidgets.QVBoxLayout(self)

            label = QtWidgets.QLabel(par)
            label.setMinimumWidth(40)
            self.value_label = QtWidgets.QLabel()
            self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.slider.setMinimum(0)
            self.slider.setMaximum(int(1e8))
            self.tmin = QtWidgets.QDoubleSpinBox()
            vmin, vmax = minuit.limits[par]
            self.tmin.setRange(_make_finite(vmin), _make_finite(np.inf))
            self.tmax = QtWidgets.QDoubleSpinBox()
            self.tmax.setRange(_make_finite(-np.inf), _make_finite(vmax))
            self.fix = QtWidgets.QPushButton("Fix")
            self.fix.setCheckable(True)
            self.fix.setChecked(minuit.fixed[par])
            self.fit = QtWidgets.QPushButton("Fit")
            self.fit.setCheckable(True)
            self.fit.setChecked(False)
            hlayout1 = QtWidgets.QHBoxLayout()
            vlayout.addLayout(hlayout1)
            hlayout1.addWidget(label)
            hlayout1.addWidget(self.slider)
            hlayout1.addWidget(self.value_label)
            hlayout1.addWidget(self.fix)
            hlayout2 = QtWidgets.QHBoxLayout()
            vlayout.addLayout(hlayout2)
            hlayout2.addWidget(self.tmin)
            hlayout2.addWidget(self.tmax)
            hlayout2.addWidget(self.fit)

            self.reset(minuit.values[par], limits=minuit.limits[par])

            step_size = 1e-1 * (self.vmax - self.vmin)
            decimals = max(int(-np.log10(step_size)) + 2, 0)
            self.tmin.setSingleStep(step_size)
            self.tmin.setDecimals(decimals)
            self.tmax.setSingleStep(step_size)
            self.tmax.setDecimals(decimals)
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
                with _block_signals(self.tmin):
                    self.tmin.setValue(self.vmin)
                return
            self.vmin = tmin
            with _block_signals(self.slider):
                if tmin > self.val:
                    self.val = tmin
                    minuit.values[self.par] = tmin
                    self.slider.setValue(0)
                    self.value_label.setText(f"{self.val:.3g}")
                    self.callback()
                else:
                    self.slider.setValue(self._float_to_int(self.val))
            lim = minuit.limits[self.par]
            minuit.limits[self.par] = (tmin, lim[1])

        def on_max_changed(self):
            tmax = self.tmax.value()
            if tmax <= self.tmin.value():
                with _block_signals(self.tmax):
                    self.tmax.setValue(self.vmax)
                return
            self.vmax = tmax
            with _block_signals(self.slider):
                if tmax < self.val:
                    self.val = tmax
                    minuit.values[self.par] = tmax
                    self.slider.setValue(int(1e8))
                    self.value_label.setText(f"{self.val:.3g}")
                    self.callback()
                else:
                    self.slider.setValue(self._float_to_int(self.val))
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
                step = _widget_guess_initial_step(val, vmin, vmax)
                self.vmin = vmin if np.isfinite(vmin) else val - 100 * step
                self.vmax = vmax if np.isfinite(vmax) else val + 100 * step
                with _block_signals(self.tmin, self.tmax):
                    self.tmin.setValue(self.vmin)
                    self.tmax.setValue(self.vmax)

            self.val = val
            if self.val < self.vmin:
                self.vmin = self.val
                with _block_signals(self.tmin):
                    self.tmin.setValue(self.vmin)
            elif self.val > self.vmax:
                self.vmax = self.val
                with _block_signals(self.tmax):
                    self.tmax.setValue(self.vmax)

            with _block_signals(self.slider):
                self.slider.setValue(self._float_to_int(self.val))
            self.value_label.setText(f"{self.val:.3g}")

    class Widget(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            font = QtGui.QFont()
            font.setPointSize(11)
            self.setFont(font)
            self.setWindowTitle("iminuit")

            hlayout = QtWidgets.QHBoxLayout(self)

            plot_group = self.make_plot_group()
            button_group = self.make_button_group()
            parameter_scroll_area = self.make_parameter_scroll_area()

            self.results_text = QtWidgets.QTextEdit(parent=self)
            self.results_text.setReadOnly(True)
            self.results_text.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            )
            self.results_text.setMaximumHeight(150)

            vlayout_left = QtWidgets.QVBoxLayout()
            vlayout_left.addWidget(plot_group)
            vlayout_left.addWidget(self.results_text)

            vlayout_right = QtWidgets.QVBoxLayout()
            vlayout_right.addWidget(button_group)
            vlayout_right.addWidget(parameter_scroll_area)

            hlayout.addLayout(vlayout_left)
            hlayout.addLayout(vlayout_right)

            self.plot_with_frame(from_fit=False, report_success=False)

        def make_plot_group(self):
            plot_group = QtWidgets.QGroupBox("", parent=self)
            plot_group.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            )
            plot_layout = QtWidgets.QVBoxLayout(plot_group)
            fig = plt.figure()
            manager = plt.get_current_fig_manager()
            self.canvas = FigureCanvasQTAgg(fig)
            self.canvas.manager = manager
            plot_layout.addWidget(self.canvas)
            return plot_group

        def make_button_group(self):
            button_group = QtWidgets.QGroupBox("", parent=self)
            button_group.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Minimum,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            button_layout = QtWidgets.QHBoxLayout(button_group)
            self.fit_button = QtWidgets.QPushButton("Fit", parent=button_group)
            self.fit_button.setStyleSheet("background-color: #2196F3; color: white")
            self.fit_button.clicked.connect(lambda: self.do_fit(plot=True))
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
            self.algo_choice.setEditable(True)
            self.algo_choice.lineEdit().setReadOnly(True)
            self.algo_choice.addItems(["Migrad", "Scipy", "Simplex"])
            button_layout.addWidget(self.algo_choice)
            return button_group

        def make_parameter_scroll_area(self):
            par_scroll_area = QtWidgets.QScrollArea(parent=self)
            par_scroll_area.setContentsMargins(0, 0, 0, 0)
            par_scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
            par_scroll_area.setWidgetResizable(True)
            par_scroll_area.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Minimum,
                QtWidgets.QSizePolicy.Policy.Minimum,
            )
            scroll_area_contents = QtWidgets.QWidget(par_scroll_area)
            parameter_layout = QtWidgets.QVBoxLayout(scroll_area_contents)
            parameter_layout.setContentsMargins(0, 0, 0, 0)
            par_scroll_area.setWidget(scroll_area_contents)
            self.parameters = []
            for par in minuit.parameters:
                parameter = Parameter(minuit, par, self.on_parameter_change)
                self.parameters.append(parameter)
                parameter_layout.addWidget(parameter)
            parameter_layout.addStretch()
            return par_scroll_area

        def plot_with_frame(self, from_fit, report_success):
            trans = plt.gca().transAxes
            try:
                with warnings.catch_warnings():
                    fig_size = plt.gcf().get_size_inches()
                    minuit.visualize(plot, **kwargs)
                    plt.gcf().set_size_inches(fig_size)
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
                self.results_text.clear()
                self.results_text.setHtml(
                    f"<div style='text-align: center;'>{minuit.fmin._repr_html_()}</div>"
                )
            else:
                self.results_text.clear()

        def fit(self):
            if self.algo_choice.currentText() == "Migrad":
                minuit.migrad()
            elif self.algo_choice.currentText() == "Scipy":
                minuit.scipy()
            elif self.algo_choice.currentText() == "Simplex":
                minuit.simplex()
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
                minuit.fixed = saved

            plt.clf()
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

    if run_event_loop:  # pragma: no cover, should not be executed in tests
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        widget = Widget()
        widget.show()
        app.exec()  # this blocks the main thread
    else:
        return Widget()


@contextmanager
def _block_signals(*widgets):
    for w in widgets:
        w.blockSignals(True)
    try:
        yield
    finally:
        for w in widgets:
            w.blockSignals(False)
