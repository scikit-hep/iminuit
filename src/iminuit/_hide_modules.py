import sys
import contextlib
from importlib.abc import MetaPathFinder


class HiddenModules(MetaPathFinder):
    def __init__(self, modules):
        super().__init__()
        self.modules = modules

    def find_spec(self, fullname, path, target=None):
        if fullname in self.modules:
            raise ModuleNotFoundError(fullname)


@contextlib.contextmanager
def hide_modules(*modules, reload=None):
    saved = {}
    for m in tuple(sys.modules):
        for to_hide in modules:
            if m.startswith(to_hide):
                saved[m] = sys.modules[m]
                del sys.modules[m]
    sys.meta_path.insert(0, HiddenModules(modules))
    if reload:
        del sys.modules[reload]
    yield
    if reload:
        del sys.modules[reload]
    sys.meta_path = sys.meta_path[1:]
    for name, mod in saved.items():
        sys.modules[name] = mod
