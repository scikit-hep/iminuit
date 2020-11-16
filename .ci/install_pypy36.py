import sys
from urllib.request import urlopen
import tarfile
import subprocess as subp
from pathlib import Path
import tempfile
import shutil

if Path("pypy36/pypy3/bin/activate").exists():
    raise SystemExit

plat = {"darwin": "osx64", "linux": "linux64"}[sys.platform]
filename = f"pypy3.6-v7.3.1-{plat}.tar.bz2"

url = "https://bitbucket.org/pypy/pypy/downloads/"
plat = {"darwin": "osx64", "linux": "linux64"}[sys.platform]
url += filename

print("Download and extract sources")
with urlopen(url) as fi:
    with tempfile.TemporaryFile() as tmp:
        shutil.copyfileobj(fi, tmp)
        tmp.seek(0)
        with tarfile.open(fileobj=tmp) as t:
            t.extractall()

dirname = filename[: filename.index(".tar.bz2")]
p = Path(dirname)
assert p.exists()
p.rename("pypy36")
p = Path("pypy36")

print("Create virtual environment")
subp.check_call([str(p / "bin" / "pypy3"), "-m", "venv", str(p / "pypy3")])
