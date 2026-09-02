import sys
import contextlib
from importlib.abc import MetaPathFinder


class HiddenModules(MetaPathFinder):
    def __init__(self, modules):
        super().__init__()
        self.modules = modules

    def find_spec(self, fullname, path, target=None):
        if fullname in self.modules:
            raise ModuleNotFoundError(
                f"{fullname!r} is hidden", name=fullname, path=path
            )


def _forget(names):
    # A package caches its submodules as attributes, and `from pkg import mod`
    # reads that attribute instead of importing. Drop both, or the next import
    # returns the module that was loaded while the dependency was hidden.
    for name in names:
        sys.modules.pop(name, None)
        parent, _, child = name.rpartition(".")
        if parent in sys.modules:
            with contextlib.suppress(AttributeError):
                delattr(sys.modules[parent], child)


@contextlib.contextmanager
def hide_modules(*modules, reload=None):
    """Hide modules and optionally force a reload of one or more other modules."""
    if reload is None:
        reload = ()
    elif isinstance(reload, str):
        reload = (reload,)
    saved = {}
    for m in tuple(sys.modules):
        for to_hide in modules:
            if m.startswith(to_hide):
                saved[m] = sys.modules.pop(m)
    finder = HiddenModules(modules)
    sys.meta_path.insert(0, finder)
    try:
        _forget(reload)
        yield
    finally:
        _forget(reload)
        # already gone if the block itself modified sys.meta_path
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(finder)
        sys.modules.update(saved)
