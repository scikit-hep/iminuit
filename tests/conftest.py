import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_ipython():
    with patch("IPython.get_ipython") as mock_get_ipython:
        mock_shell = MagicMock()

        def has_trait(name):
            return True

        mock_shell.has_trait.side_effect = has_trait
        mock_get_ipython.return_value = mock_shell
        yield


@pytest.fixture
def debug():
    from iminuit._core import MnPrint

    prev = MnPrint.global_level
    MnPrint.global_level = 3
    MnPrint.show_prefix_stack(True)
    yield
    MnPrint.global_level = prev
    MnPrint.show_prefix_stack(False)
