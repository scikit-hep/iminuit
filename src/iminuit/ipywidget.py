import warnings
from iminuit.util import _guess_initial_step
import numpy as np
from contextlib import contextmanager

with warnings.catch_warnings():
    # ipywidgets produces deprecation warnings through use of internal APIs :(
    warnings.simplefilter("ignore")
    try:
        from ipywidgets import (
            HBox,
            VBox,
            Output,
            FloatSlider,
            Button,
            ToggleButton,
            Layout,
            Dropdown,
        )
        from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
        from IPython.display import clear_output
        from matplotlib import pyplot as plt
    except ModuleNotFoundError as e:
        e.msg += (
            "\n\nPlease install ipywidgets, IPython, and matplotlib to "
            "enable interactive"
        )
        raise


class IPyWidget(HBox):
    def __init__(self, minuit, plot, kwargs, raise_on_exception):
        def plot_with_frame(args, from_fit, report_success):
            trans = plt.gca().transAxes
            try:
                with warnings.catch_warnings():
                    if minuit._fcn._array_call:
                        plot([args], **kwargs)  # prevent unpacking of array
                    else:
                        plot(args, **kwargs)
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
            if from_fit:
                fval = minuit.fmin.fval
            else:
                fval = minuit._fcn(args)
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

        def on_slider_change(change):
            args = [x.slider.value for x in parameters]
            from_fit = False
            report_success = False
            if any(x.opt.value for x in parameters):
                save = minuit.fixed[:]
                minuit.fixed = [not x.opt.value for x in parameters]
                minuit.values = args
                report_success = fit()
                args = minuit.values[:]
                for x, val in zip(parameters, args):
                    with pause_slider_update(x.slider):
                        x.slider.value = val
                minuit.fixed = save
                from_fit = True
            with out:
                clear_output(wait=True)
                plot_with_frame(args, from_fit, report_success)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    show_inline_matplotlib_plots()

        def on_fit_button_clicked(change):
            for x in parameters:
                minuit.values[x.par] = x.slider.value
                minuit.fixed[x.par] = x.fix.value
            report_success = fit()
            for x in parameters:
                with pause_slider_update(x.slider):
                    val = minuit.values[x.par]
                    if val < x.slider.min:
                        x.slider.min = val
                    elif val > x.slider.max:
                        x.slider.max = val
                    x.slider.value = val
            with out:
                clear_output(wait=True)
                plot_with_frame(minuit.values, True, report_success)
                show_inline_matplotlib_plots()

        def on_update_button_clicked(change):
            for x in parameters:
                x.slider.continuous_update = not x.slider.continuous_update

        def on_reset_button_clicked(change):
            minuit.reset()
            for x in parameters:
                with pause_slider_update(x.slider):
                    x.slider.value = minuit.values[x.par]
            on_slider_change(None)

        @contextmanager
        def pause_slider_update(slider):
            slider.observe(lambda change: None, "value")
            yield
            slider.observe(on_slider_change, "value")

        class ParameterBox(HBox):
            def __init__(self, par, val, min, max, step, fix):
                self.par = par
                self.slider = FloatSlider(
                    val,
                    min=min,
                    max=max,
                    step=step,
                    description=par,
                    continuous_update=True,
                    readout_format=".4g",
                    layout=Layout(min_width="70%"),
                )
                self.fix = ToggleButton(
                    fix, description="Fix", layout=Layout(width="3.1em")
                )
                self.opt = ToggleButton(
                    False, description="Opt", layout=Layout(width="3.5em")
                )
                self.opt.observe(self.on_opt_toggled, "value")
                super().__init__([self.slider, self.fix, self.opt])

            def on_opt_toggled(self, change):
                self.slider.disabled = self.opt.value
                on_slider_change(None)

        parameters = []
        for par in minuit.parameters:
            val = minuit.values[par]
            step = _guess_initial_step(val)
            a, b = minuit.limits[par]
            # safety margin to avoid overflow warnings
            a = a + 1e-300 if np.isfinite(a) else val - 100 * step
            b = b - 1e-300 if np.isfinite(b) else val + 100 * step
            parameters.append(ParameterBox(par, val, a, b, step, minuit.fixed[par]))

        fit_button = Button(description="Fit")
        fit_button.on_click(on_fit_button_clicked)

        update_button = ToggleButton(True, description="Continuous")
        update_button.observe(on_update_button_clicked)

        reset_button = Button(description="Reset")
        reset_button.on_click(on_reset_button_clicked)

        algo_choice = Dropdown(options=["Migrad", "Scipy", "Simplex"], value="Migrad")

        ui = VBox(
            [
                HBox([fit_button, update_button, reset_button, algo_choice]),
                VBox(parameters),
            ]
        )

        out = Output()

        for x in parameters:
            x.slider.observe(on_slider_change, "value")

        show_inline_matplotlib_plots()
        on_slider_change(None)

        super().__init__([out, ui])
