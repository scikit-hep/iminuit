"""IPyWidgets backend."""

from typing import Sequence
import ipywidgets as widgets
from IPython.display import display


class SpinBox(widgets.BoundedFloatText):
    def __init__(
        self,
        decimals: int,
        step: float,
        value: float,
        vmin: float,
        vmax: float,
        on_change: callable,
    ):
        super().__init__(value, min=vmin, max=vmax, step=step, decimals=decimals)
        super().observe(lambda event: on_change(self.value))

    def set_value(self, value):
        super().value = value

    def set_min(self, value):
        super().min = value

    def set_max(self, value):
        super().max = value


class ComboBox(widgets.Dropdown):
    def __init__(self, choices: Sequence[str], value: str, on_change: callable):
        super().__init__(options=list(choices), value=value)
        super().observe(lambda event: on_change(self.value))


class Slider(widgets.FloatSlider):
    def __init__(self, vmin: float, vmax: float, value: float, on_change: callable):
        super().__init__(
            min=vmin, max=vmax, value=value, continuous_update=True, readout=False
        )
        super().observe(lambda event: on_change(self.value))

    def set_value(self, value: bool):
        self.value = value

    def set_enabled(self, on: bool):
        self.disabled = not on


class Button(widgets.Button):
    def __init__(self, label, on_click):
        super().__init__(description=label)
        super().on_click(lambda *args: on_click())

    def set_style(self, style: str):
        for part in style.split(";"):
            key, value = part.split(":")
            key = key.strip()
            value = value.strip()
            setattr(self.layout, key, value)


class HLayout(widgets.HBox):
    def __init__(self, *args):
        super().__init__(args)


class VLayout(widgets.VBox):
    def __init__(self, *args):
        super().__init__(args)


class Label(widgets.Label):
    def __init__(self, text, min_width=0):
        super().__init__(text)
        if min_width > 0:
            self.layout.min_width = f"{min_width}pt"

    def set_text(self, text):
        self.value = text


class MainWidget:
    def __init__(self, layout):
        self.layout = layout

    def render(self):
        display(self.layout)


class ToggleButton(widgets.ToggleButton):
    def __init__(self, label, on_click, checked=False):
        super().__init__(description=label, value=checked)
        super().observe(lambda event: on_click(self.value))

    def set_checked(self, checked: bool):
        self.value = checked

    def is_checked(self) -> bool:
        return self.value


class ScrollArea(widgets.VBox):
    def __init__(self, *args):
        super().__init__(args)
        self.layout.overflow = "hidden scroll"
        self.layout.display = "flex"
        self.layout.flex_flow = "column"
        self.layout.max_height = "200px"
        self.layout.object_position = "center top"
        self.layout.align_items = "flex-start"

        # Make each child maintain its natural size
        for widget in args:
            widget.layout.min_height = "min-content"


class HtmlView(widgets.HTML):
    def set_html(self, html: str):
        self.value = html


class GroupBox(widgets.Box):
    def __init__(self, layout, title: str = ""):
        super().__init__(children=layout.children, layout=layout.layout)
        super().layout.border = "solid 1px"
