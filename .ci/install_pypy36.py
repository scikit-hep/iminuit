import sys
from urllib.request import urlopen
import tarfile
import subprocess as subp
from pathlib import Path

if Path("pypy36/pypy3/bin/activate").exists():
    raise SystemExit

plat = {"darwin": "osx64", "linux": "linux64"}[sys.platform]
filename = "pypy3.6-v7.3.1-{}.tar.bz2".format(plat)

url = "https://bitbucket.org/pypy/pypy/downloads/"
plat = {"darwin": "osx64", "linux": "linux64"}[sys.platform]
url += filename

with urlopen(url) as f:
    with tarfile.open(fileobj=f) as t:
        t.extractall()

p = Path(filename)
dirname = p.stem[: p.stem.index(".tar.bz2")]
p = Path(dirname)
assert p.exists()
p.rename("pypy36")
p = Path("pypy36")

subp.check_call([p / "bin" / "pypy3", "-m", "venv", p / "pypy3"])
