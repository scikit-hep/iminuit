"""Interactive fitting widget for Jupyter notebooks."""

from .util import _widget_guess_initial_step, _make_finite
import warnings
import numpy as np
from typing import Dict, Any, Callable

with warnings.catch_warnings():
    # ipywidgets produces deprecation warnings through use of internal APIs :(
    warnings.simplefilter("ignore")
    try:
        import ipywidgets as widgets
        from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
        from IPython.display import clear_output
        from matplotlib import pyplot as plt
    except ModuleNotFoundError as e:
        e.msg += (
            "\n\nPlease install ipywidgets, IPython, and matplotlib to "
            "enable interactive"
        )
        raise


def make_widget(
    minuit: Any,
    plot: Callable[..., None],
    kwargs: Dict[str, Any],
    raise_on_exception: bool,
):
    """Make interactive fitting widget."""
    # Implementations makes heavy use of closures,
    # we frequently use variables which are defined
    # near the end of the function.
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

    class OnParameterChange:
        # Ugly implementation notes:
        # We want the plot when the user moves the slider widget, but not when
        # we update the slider value manually from our code. Unfortunately,
        # the latter also calls OnParameterChange, which leads to superfluous plotting.
        # I could not find a nice way to prevent that (and I tried many), so as a workaround
        # we optionally skip a number of calls, when the slider is updated.
        def __init__(self, skip: int = 0):
            self.skip = skip

        def __call__(self, change: Dict[str, Any] = {}):
            if self.skip > 0:
                self.skip -= 1
                return

            from_fit = change.get("from_fit", False)
            report_success = change.get("report_success", False)
            if not from_fit:
                for i, x in enumerate(parameters):
                    minuit.values[i] = x.slider.value

            if any(x.fit.value for x in parameters):
                saved = minuit.fixed[:]
                for i, x in enumerate(parameters):
                    minuit.fixed[i] = not x.fit.value
                from_fit = True
                report_success = do_fit(None)
                minuit.fixed = saved

            # Implementation like in ipywidegts.interaction.interactive_output
            with out:
                clear_output(wait=True)
                plot_with_frame(from_fit, report_success)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    show_inline_matplotlib_plots()

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

    class Parameter(widgets.HBox):
        def __init__(self, minuit, par):
            val = minuit.values[par]
            vmin, vmax = minuit.limits[par]
            step = _widget_guess_initial_step(val, vmin, vmax)
            vmin2 = vmin if np.isfinite(vmin) else val - 100 * step
            vmax2 = vmax if np.isfinite(vmax) else val + 100 * step

            tlabel = widgets.Label(par, layout=widgets.Layout(width=f"{longest_par}em"))

            tmin = widgets.BoundedFloatText(
                _round(vmin2),
                min=_make_finite(vmin),
                max=vmax2,
                step=1e-1 * (vmax2 - vmin2),
                layout=widgets.Layout(width="4.1em"),
            )

            tmax = widgets.BoundedFloatText(
                _round(vmax2),
                min=vmin2,
                max=_make_finite(vmax),
                step=1e-1 * (vmax2 - vmin2),
                layout=widgets.Layout(width="4.1em"),
            )

            self.slider = widgets.FloatSlider(
                val,
                min=vmin2,
                max=vmax2,
                step=step,
                continuous_update=True,
                readout_format=".3g",
                layout=widgets.Layout(min_width="50%"),
            )
            self.slider.observe(OnParameterChange(), "value")

            def on_min_change(change):
                self.slider.min = change["new"]
                tmax.min = change["new"]
                lim = minuit.limits[par]
                minuit.limits[par] = (self.slider.min, lim[1])

            def on_max_change(change):
                self.slider.max = change["new"]
                tmin.max = change["new"]
                lim = minuit.limits[par]
                minuit.limits[par] = (lim[0], self.slider.max)

            tmin.observe(on_min_change, "value")
            tmax.observe(on_max_change, "value")

            self.fix = widgets.ToggleButton(
                minuit.fixed[par],
                description="Fix",
                tooltip="Fix",
                layout=widgets.Layout(width="3.1em"),
            )

            self.fit = widgets.ToggleButton(
                False,
                description="Fit",
                tooltip="Fit",
                layout=widgets.Layout(width="3.5em"),
            )

            def on_fix_toggled(change):
                minuit.fixed[par] = change["new"]
                if change["new"]:
                    self.fit.value = False

            def on_fit_toggled(change):
                self.slider.disabled = change["new"]
                if change["new"]:
                    self.fix.value = False
                OnParameterChange()()

            self.fix.observe(on_fix_toggled, "value")
            self.fit.observe(on_fit_toggled, "value")
            super().__init__([tlabel, tmin, self.slider, tmax, self.fix, self.fit])

        def reset(self, value, limits=None):
            self.slider.unobserve_all("value")
            self.slider.value = value
            if limits:
                self.slider.min, self.slider.max = limits
            # Installing the observer actually triggers a notification,
            # we skip it. See notes in OnParameterChange.
            self.slider.observe(OnParameterChange(1), "value")

    longest_par = max(len(par) for par in minuit.parameters)
    parameters = [Parameter(minuit, par) for par in minuit.parameters]

    button_layout = widgets.Layout(max_width="8em")

    fit_button = widgets.Button(
        description="Fit",
        button_style="primary",
        layout=button_layout,
    )
    fit_button.on_click(do_fit)

    update_button = widgets.ToggleButton(
        True,
        description="Continuous",
        layout=button_layout,
    )
    update_button.observe(on_update_button_clicked)

    reset_button = widgets.Button(
        description="Reset",
        button_style="danger",
        layout=button_layout,
    )
    reset_button.on_click(on_reset_button_clicked)

    algo_choice = widgets.Dropdown(
        options=["Migrad", "Scipy", "Simplex"],
        value="Migrad",
        layout=button_layout,
    )

    ui = widgets.VBox(
        [
            widgets.HBox([fit_button, update_button, reset_button, algo_choice]),
            widgets.VBox(parameters),
        ]
    )
    out = widgets.Output()
    OnParameterChange()()
    return widgets.HBox([out, ui])


def _round(x: float) -> float:
    return float(f"{x:.1g}")
