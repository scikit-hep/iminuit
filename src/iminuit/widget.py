from abc import ABC, abstractmethod
from contextlib import contextmanager
from .util import _widget_guess_initial_step, _make_finite
import numpy as np
import warnings
from typing import Callable, Sequence, Any, Tuple, List

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib import pyplot as plt
except ModuleNotFoundError as e:
    e.msg += "\n\nPlease install matplotlib to enable plotting."
    raise


class VLayout(ABC):
    @abstractmethod
    def __init__(self, *args): ...


class HLayout(ABC):
    @abstractmethod
    def __init__(self, *args): ...


class Label(ABC):
    @abstractmethod
    def __init__(self, text: str, *, width: int = 0): ...

    @abstractmethod
    def set_text(self, text: str): ...


class Container(ABC):
    @abstractmethod
    def set_layout(self, layout): ...


class SizePolicyMixin(ABC):
    @abstractmethod
    def set_size_policy(self, hpolicy: str, vpolicy: str): ...


class GroupBox(Container, SizePolicyMixin):
    @abstractmethod
    def __init__(self, title: str = ""): ...


class ScrollArea(Container, SizePolicyMixin): ...


class Widget(ABC, SizePolicyMixin):
    @abstractmethod
    def block_signals(self, block: bool): ...

    @abstractmethod
    def connect(self, callback): ...


class ComboBox(Widget):
    @abstractmethod
    def __init__(self, choices: Sequence[str]): ...

    @abstractmethod
    def text(self) -> str: ...


class HtmlView(Widget):
    @abstractmethod
    def set_html(self, text: str): ...


class Slider(Widget):
    @abstractmethod
    def __init__(self, vmin: float, vmax: float): ...

    @abstractmethod
    def set_value(self, value: float): ...

    @abstractmethod
    def set_enabled(self, on: bool): ...


class DoubleSpinBox(Widget):
    @abstractmethod
    def __init__(
        self,
        step: float,
        decimals: int,
        range: Tuple[float, float],
    ): ...

    @abstractmethod
    def value(self): ...

    @abstractmethod
    def set_value(self): ...


class Button(Widget):
    @abstractmethod
    def __init__(self, text: str): ...

    @abstractmethod
    def set_style_sheet(self, css: str): ...


class CheckButton(Button):
    @abstractmethod
    def __init__(self, text: str, checked: bool): ...

    @abstractmethod
    def is_checked(self): ...

    @abstractmethod
    def set_checked(self, checked: bool): ...


class MainWidget(Container):
    @abstractmethod
    def set_font_size(self, points: int): ...

    @abstractmethod
    def set_window_title(self, text: str): ...


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


def make_main_widget(
    backend: Backend, minuit: Any, plot: Callable, kwargs: Any, raise_on_exception: bool
):
    def visualize():
        minuit.visualize(plot, **kwargs)

    class Parameter(backend.Container):
        def __init__(self, par: str, callback: Callable):
            super().__init__()
            self.par = par
            self.callback = callback

            vmin, vmax = minuit.limits[par]
            step_size = 1e-1 * (self.vmax - self.vmin)
            decimals = max(int(-np.log10(step_size)) + 2, 0)

            self.par_label = backend.Label(par, width=40)
            self.value_label = backend.Label()

            self.slider = backend.Slider(vmin, vmax)

            self.tmin = backend.DoubleSpinBox(
                step=step_size,
                decimals=decimals,
                range=(_make_finite(vmin), _make_finite(np.inf)),
            )
            self.tmax = backend.DoubleSpinBox(
                step=step_size,
                decimals=decimals,
                range=(_make_finite(-np.inf), _make_finite(vmax)),
            )

            self.fix = backend.CheckButton("Fix", minuit.fixed[par])
            self.fit = backend.CheckButton("Fit", False)

            self.set_layout(
                backend.VLayout(
                    backend.HLayout(
                        self.par_label, self.slider, self.value_label, self.fix
                    ),
                    backend.HLayout(self.tmin, self.tmax, self.fit),
                )
            )

            self.reset(minuit.values[par], limits=minuit.limits[par])

            self.slider.connect(self.on_val_changed)
            self.fix.connect(self.on_fix_toggled)
            self.tmin.connect(self.on_min_changed)
            self.tmax.connect(self.on_max_changed)
            self.fit.connect(self.on_fit_toggled)

        def on_val_changed(self, val):
            self.value_label.set_text(f"{val:.3g}")
            minuit.values[self.par] = val
            self.callback()

        def on_min_changed(self):
            tmin = self.tmin.value()
            if tmin >= self.vmax:
                with _block_signals(self.tmin):
                    self.tmin.set_value(self.vmin)
                return
            self.vmin = tmin
            with _block_signals(self.slider):
                if tmin > self.val:
                    self.val = tmin
                    minuit.values[self.par] = tmin
                    self.slider.set_value(tmin)
                    self.value_label.set_text(f"{tmin:.3g}")
                    self.callback()
                else:
                    self.slider.set_value(self.val)
            lim = minuit.limits[self.par]
            minuit.limits[self.par] = (tmin, lim[1])

        def on_max_changed(self):
            tmax = self.tmax.value()
            if tmax <= self.tmin.value():
                with _block_signals(self.tmax):
                    self.tmax.set_value(self.vmax)
                return
            self.vmax = tmax
            with _block_signals(self.slider):
                if tmax < self.val:
                    self.val = tmax
                    minuit.values[self.par] = tmax
                    self.slider.set_value(tmax)
                    self.value_label.set_text(f"{tmax:.3g}")
                    self.callback()
                else:
                    self.slider.set_value(self.val)
            lim = minuit.limits[self.par]
            minuit.limits[self.par] = (lim[0], tmax)

        def on_fix_toggled(self):
            minuit.fixed[self.par] = self.fix.is_checked()
            if self.fix.is_checked():
                self.fit.set_checked(False)

        def on_fit_toggled(self):
            self.slider.set_enabled(not self.fit.is_checked())
            if self.fit.is_checked():
                self.fix.set_checked(False)
            self.callback()

        def reset(self, value, limits=None):
            if limits is not None:
                vmin, vmax = limits
                step = _widget_guess_initial_step(value, vmin, vmax)
                self.vmin = vmin if np.isfinite(vmin) else value - 100 * step
                self.vmax = vmax if np.isfinite(vmax) else value + 100 * step
                with _block_signals(self.tmin, self.tmax):
                    self.tmin.set_value(self.vmin)
                    self.tmax.set_value(self.vmax)

            self.val = value
            if self.val < self.vmin:
                self.vmin = self.val
                with _block_signals(self.tmin):
                    self.tmin.set_value(self.vmin)
            elif self.val > self.vmax:
                self.vmax = self.val
                with _block_signals(self.tmax):
                    self.tmax.set_value(self.vmax)

            with _block_signals(self.slider):
                self.slider.set_value(self.val)
            self.value_label.set_text(f"{self.val:.3g}")

    class MainWidget(backend.MainWidget):
        def __init__(self):
            super().__init__()

            plot_group = self.make_plot_group()
            button_group = self.make_button_group()
            parameter_scroll_area = self.make_parameter_scroll_area()

            self.results_text = backend.HtmlView()
            self.results_text.set_size_policy(
                "minimum_expanding", "minimum_expanding", maximum_height=150
            )

            hlayout = backend.HLayout(
                backend.VLayout(plot_group, self.results_text),
                backend.VLayout(button_group, parameter_scroll_area),
            )
            self.set_layout(hlayout)

            self.plot_with_frame(from_fit=False, report_success=False)

        def make_plot_group(self):
            plot_group = backend.GroupBox()
            plot_group.set_size_policy(
                "minimum_expanding",
                "minimum_expanding",
            )

            fig = plt.figure()
            manager = plt.get_current_fig_manager()
            self.canvas = FigureCanvasQTAgg(fig)
            self.canvas.manager = manager

            plot_group.set_layout(backend.VLayout(self.canvas))
            return plot_group

        def make_button_group(self):
            button_group = backend.GroupBox()
            button_group.set_size_policy(
                "minimum",
                "fixed",
            )
            self.fit_button = backend.Button("Fit")
            self.fit_button.set_style_sheet("background-color: #2196F3; color: white")
            self.fit_button.connect(lambda: self.do_fit(plot=True))

            self.update_button = backend.CheckButton("Continuous", True)
            self.update_button.connect(self.on_update_button_clicked)

            self.reset_button = backend.Button("Reset")
            self.reset_button.set_style_sheet("background-color: #F44336; color: white")
            self.reset_button.connect(self.on_reset_button_clicked)

            self.algo_choice = backend.ComboBox(["Migrad", "Scipy", "Simplex"])

            button_group.set_layout(
                backend.HLayout(
                    self.fit_button,
                    self.update_button,
                    self.reset_button,
                    self.algo_choice,
                )
            )

            return button_group

        def make_parameter_scroll_area(self):
            par_scroll_area = backend.ScrollArea()
            par_scroll_area.set_size_policy("minimum", "minimum")

            self.parameters: List[Parameter] = []
            for par in minuit.parameters:
                parameter = Parameter(minuit, par, self.on_parameter_change)
                self.parameters.append(parameter)
            par_scroll_area.set_layout(backend.VLayout(*self.parameters))

            return par_scroll_area

        def plot_with_frame(self, from_fit, report_success):
            trans = plt.gca().transAxes
            try:
                with warnings.catch_warnings():
                    fig_size = plt.gcf().get_size_inches()
                    visualize()
                    plt.gcf().set_size_inches(fig_size)
            except Exception:
                if self.raise_on_exception:
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
                self.results_text.set_html(
                    f"<div style='text-align: center;'>{minuit.fmin._repr_html_()}</div>"
                )
            else:
                self.results_text.set_html("")

        def fit(self):
            if self.algo_choice.text() == "Migrad":
                minuit.migrad()
            elif self.algo_choice.text() == "Scipy":
                minuit.scipy()
            elif self.algo_choice.text() == "Simplex":
                minuit.simplex()
            else:
                assert False  # pragma: no cover, should never happen
            return True

        def on_parameter_change(self, from_fit=False, report_success=False):
            if any(x.fit.checked() for x in self.parameters):
                saved = minuit.fixed[:]
                for i, x in enumerate(self.parameters):
                    minuit.fixed[i] = not x.fit.checked()
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
                x.slider.set_tracking(self.update_button.checked())

        def on_reset_button_clicked(self):
            minuit.reset()
            minuit.values = self.original_values
            minuit.limits = self.original_limits
            for i, x in enumerate(self.parameters):
                x.reset(val=minuit.values[i], limits=self.original_limits[i])
            self.on_parameter_change()

    return MainWidget()


@contextmanager
def _block_signals(*widgets: Widget):
    for w in widgets:
        w.block_signals(True)
    try:
        yield
    finally:
        for w in widgets:
            w.block_signals(False)
