"""
Noxfile for iMinuit.

Use `-R` to instantly reuse an existing environment and
to avoid rebuilding the binary.
"""

import argparse

import nox

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv|virtualenv"

ENV = {
    "JUPYTER_PLATFORM_DIRS": "1",  # Hides warnings in Jupyter
    "COVERAGE_CORE": "sysmon",  # faster coverage on Python 3.12
}


@nox.session
def tests(session: nox.Session) -> None:
    """Run the unit and regular tests."""
    session.install("-e.[test]")
    session.run("pytest", *session.posargs, env=ENV)


@nox.session
def np2tests(session: nox.Session) -> None:
    """Run the unit and regular tests."""
    session.install("-e.", "scipy", "pytest", "--pre")
    session.run("pytest", *session.posargs, env=ENV)


@nox.session(venv_backend="uv")
def mintests(session: nox.Session) -> None:
    """Run the unit and regular tests."""
    session.install("-e.", "--resolution=lowest-direct")
    session.install("pytest")
    session.run("pytest", *session.posargs)


@nox.session(python="3.12")
def coverage(session: nox.Session) -> None:
    """Run covage and place in 'htmlcov' directory."""
    session.install("-e.[test,doc]")
    session.run("coverage", "run", "-m", "pytest", env=ENV)
    session.run("coverage", "html", "-d", "htmlcov")
    session.run("coverage", "report", "-m")


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """Build the docs. Use '--non-interactive' to avoid serving. Pass '-b linkcheck' to check links."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    args, posargs = parser.parse_known_args(session.posargs)

    serve = args.builder == "html" and session.interactive
    extra_installs = ["sphinx-autobuild"] if serve else []
    session.install("-e.[docs]", *extra_installs)

    session.chdir("doc")

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        ".",
        f"_build/{args.builder}",
        *posargs,
    )

    if serve:
        session.run("sphinx-autobuild", "--open-browser", *shared_args)
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)


@nox.session
def tutorial(session: nox.Session) -> None:
    """Start up a juptyer lab tutorial session."""
    session.install("jupyterlab", "matplotlib", "scipy", "ipykernel", "-e.")
    session.chdir("doc/tutorial")
    session.run("jupyter", "lab")
