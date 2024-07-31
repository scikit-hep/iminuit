"""
Noxfile for iMinuit.

Use `-R` to instantly reuse an existing environment and
to avoid rebuilding the binary.
"""

import nox

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv|virtualenv"

ENV = {
    "JUPYTER_PLATFORM_DIRS": "1",  # Hides warnings in Jupyter
    "COVERAGE_CORE": "sysmon",  # faster coverage on Python 3.12
}


@nox.session(reuse_venv=True)
def test(session: nox.Session) -> None:
    """Run the unit and regular tests."""
    session.install("-e.[test]")
    session.run("pytest", *session.posargs, env=ENV)


@nox.session(reuse_venv=True)
def np2test(session: nox.Session) -> None:
    """Run the unit and regular tests."""
    session.install("-e.", "scipy", "pytest", "--pre")
    session.run("pytest", *session.posargs, env=ENV)


@nox.session(venv_backend="uv")
def mintest(session: nox.Session) -> None:
    """Run the unit and regular tests."""
    session.install("-e.", "--resolution=lowest-direct")
    session.install("pytest")
    session.run("pytest", *session.posargs)


# Python-3.12 provides coverage info faster
@nox.session(python="3.12", reuse_venv=True)
def cov(session: nox.Session) -> None:
    """Run covage and place in 'htmlcov' directory."""
    session.install("-e.[test,doc]")
    session.run("coverage", "run", "-m", "pytest", env=ENV)
    session.run("coverage", "html", "-d", "htmlcov")
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
