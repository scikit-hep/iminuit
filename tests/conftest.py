import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_ipython():
    with patch("IPython.get_ipython") as mock_get_ipython:
        mock_shell = MagicMock()
        mock_shell.__class__.__name__ = "ZMQInteractiveShell"
        mock_shell.config = {"IPKernelApp": True}
        mock_get_ipython.return_value = mock_shell
        yield
