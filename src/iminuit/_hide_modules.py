import sys
import contextlib


@contextlib.contextmanager
def hide_modules(*modules):
    saved = {}
    for m in tuple(sys.modules):
        for to_hide in modules:
            if m.startswith(to_hide):
                saved[m] = sys.modules[m]
                sys.modules[m] = None
    yield
    for name, mod in saved.items():
        sys.modules[name] = mod
