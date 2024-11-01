from iminuit._hide_modules import hide_modules
from iminuit.cost import LeastSquares
import pytest
from unittest.mock import patch, MagicMock

pytest.importorskip("ipywidgets")


def test_interactive():
    pytest.importorskip("matplotlib")
    import iminuit

    cost = LeastSquares([1.1, 2.2], [3.3, 4.4], 1, lambda x, a: a * x)

    with patch("IPython.get_ipython") as mock_get_ipython:
        mock_shell = MagicMock()
        mock_shell.__class__.__name__ = "ZMQInteractiveShell"
        mock_shell.config = {"IPKernelApp": True}
        mock_get_ipython.return_value = mock_shell

        iminuit.Minuit(cost, 1).interactive()

        with hide_modules("ipywidgets", reload="iminuit.ipywidget"):
            with pytest.raises(ModuleNotFoundError, match="Please install"):
                iminuit.Minuit(cost, 1).interactive()
