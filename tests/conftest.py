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
