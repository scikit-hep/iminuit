"""
Noxfile for iminuit.

Pass extra arguments to pytest after --
"""

import nox
import sys

sys.path.append(".")
import python_releases

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv|virtualenv"

ENV = {
    "JUPYTER_PLATFORM_DIRS": "1",  # Hides warnings in Jupyter
    "COVERAGE_CORE": "sysmon",  # faster coverage on Python 3.12
}

PYPROJECT = nox.project.load_toml("pyproject.toml")
MINIMUM_PYTHON = PYPROJECT["project"]["requires-python"].strip(">=")
LATEST_PYTHON = str(python_releases.latest())

nox.options.sessions = ["test", "mintest", "maxtest"]


@nox.session(reuse_venv=True)
def test(session: nox.Session) -> None:
    """Run all tests."""
    session.install("-e.[test]")
    extra_args = session.posargs if session.posargs else ("-n=auto",)
    session.run("pytest", *extra_args, env=ENV)


@nox.session(python=MINIMUM_PYTHON, venv_backend="uv")
def mintest(session: nox.Session) -> None:
    """Run tests on the minimum python version."""
    session.install("-e.", "--resolution=lowest-direct")
    session.install("pytest", "pytest-xdist")
    extra_args = session.posargs if session.posargs else ("-n=auto",)
    session.run("pytest", *extra_args)


@nox.session(python=LATEST_PYTHON)
def maxtest(session: nox.Session) -> None:
    """Run the unit and regular tests."""
    session.install("-e.", "scipy", "matplotlib", "pytest", "pytest-xdist", "--pre")
    extra_args = session.posargs if session.posargs else ("-n=auto",)
    session.run("pytest", *extra_args, env=ENV)


@nox.session(python="pypy3.9")
def pypy(session: nox.Session) -> None:
    """Run the unit and regular tests."""
    session.install("-e.")
    session.install("pytest", "pytest-xdist")
    session.run("pytest", "-n=auto", *session.posargs)


# Python-3.12 provides coverage info faster
@nox.session(python="3.12", venv_backend="uv", reuse_venv=True)
def cov(session: nox.Session) -> None:
    """Run covage and place in 'htmlcov' directory."""
    session.install("-e.[test,doc]")
    session.run("coverage", "run", "-m", "pytest", env=ENV)
    session.run("coverage", "html", "-d", "build/htmlcov")
    session.run("coverage", "report", "-m")


# 3.11 needed by Cython notebook
@nox.session(python="3.11", reuse_venv=True)
def doc(session: nox.Session) -> None:
    """Build html documentation."""
    session.install("-e.[test,doc]")

    # link check
    session.run(
        "sphinx-build",
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        "-v",
        "-b=html",
        "doc",
        "build/html",
    )


@nox.session(python="3.11", reuse_venv=True)
def linkcheck(session: nox.Session) -> None:
    """Check all links in the documentation."""
    session.install("-e.[test,doc]")

    # link check
    session.run(
        "sphinx-build",
        "-b=linkcheck",
        "doc",
        "build/html",
    )
