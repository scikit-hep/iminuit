"""Interactive fitting widget for Jupyter notebooks."""

import warnings
from iminuit.util import _guess_initial_step
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
    original_values = minuit.values[:]

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
                0.01,
                0.5,
                traceback.format_exc(),
                ha="left",
                va="center",
                transform=trans,
                color="r",
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
        # Updating the slider asynchronously calls on_parameter_change. I could not find
        # a way to prevent that (and I tried many), so as a workaround we skip
        # two calls for each slider update, because updating the slider generates two
        # calls due to rounding (which is stupid).

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
                from_fit = True
                saved = minuit.fixed[:]
                for i, x in enumerate(parameters):
                    minuit.fixed[i] = not x.fit.value
                report_success = fit()
                for i, x in enumerate(parameters):
                    if x.fit.value:
                        x.reset(minuit.values[i])
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
        OnParameterChange()({"from_fit": True, "report_success": report_success})

    def on_update_button_clicked(change):
        for x in parameters:
            x.slider.continuous_update = not x.slider.continuous_update

    def on_reset_button_clicked(change):
        minuit.reset()
        minuit.values = original_values
        for i, x in enumerate(parameters):
            x.reset("value", minuit.values[i])
        OnParameterChange()()

    class Parameter(widgets.HBox):
        def __init__(self, minuit, par):
            self.minuit = minuit
            self.par = par
            val = minuit.values[par]
            step = _guess_initial_step(val)
            vmin, vmax = minuit.limits[par]
            # safety margin to avoid overflow warnings
            vmin = vmin + 1e-300 if np.isfinite(vmin) else val - 100 * step
            vmax = vmax - 1e-300 if np.isfinite(vmax) else val + 100 * step

            self.slider = widgets.FloatSlider(
                val,
                min=vmin,
                max=vmax,
                step=step,
                description=par,
                continuous_update=True,
                readout_format=".4g",
                layout=widgets.Layout(min_width="70%"),
            )
            self.slider.observe(OnParameterChange(), "value")

            self.fix = widgets.ToggleButton(
                minuit.fixed[par],
                description="Fix",
                layout=widgets.Layout(width="3.1em"),
            )

            def on_fix_toggled(change):
                minuit.fixed[par] = change["new"]
                if change["new"]:
                    self.fit.value = False

            self.fix.observe(on_fix_toggled, "value")

            self.fit = widgets.ToggleButton(
                False, description="Fit", layout=widgets.Layout(width="3.5em")
            )

            def on_fit_toggled(change):
                self.slider.disabled = change["new"]
                if change["new"]:
                    self.fix.value = False
                OnParameterChange()()

            self.fit.observe(on_fit_toggled, "value")
            super().__init__([self.slider, self.fix, self.fit])

        def reset(self, value):
            self.slider.unobserve_all("value")
            self.slider.value = value
            self.slider.observe(OnParameterChange(1), "value")

    parameters = [Parameter(minuit, par) for par in minuit.parameters]

    fit_button = widgets.Button(description="Fit")
    fit_button.on_click(do_fit)

    update_button = widgets.ToggleButton(True, description="Continuous")
    update_button.observe(on_update_button_clicked)

    reset_button = widgets.Button(description="Reset")
    reset_button.on_click(on_reset_button_clicked)

    algo_choice = widgets.Dropdown(
        options=["Migrad", "Scipy", "Simplex"], value="Migrad"
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
