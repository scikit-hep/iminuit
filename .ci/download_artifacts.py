import requests
import json
import shutil
from urllib.request import urlopen
from pathlib import Path
import zipfile
import tempfile

organization = "scikit-hep"
project = "iminuit"


def get(url):
    return json.loads(requests.get(url).content)


def get_latest_successful_build():
    # get list of all builds
    artifact = get(
        f"https://api.github.com/repos/{organization}/{project}/actions/artifacts"
    )["artifacts"][0]
    return artifact["id"], artifact["updated_at"], artifact["archive_download_url"]


id, date, url = get_latest_successful_build()

# TODO needs authentication to work
print(f"Downloading artifacts for {id} {date}")

dist = Path("dist")
if dist.exists():
    if any(dist.iterdir()):
        raise SystemExit("Error: ./dist directory is not empty; please delete content")
else:
    dist.mkdir()

print(f"Downloading {id}...")
with tempfile.TemporaryFile(mode="w+b") as tmp:
    with urlopen(url) as src:
        shutil.copyfileobj(src, tmp)
    with zipfile.ZipFile(tmp) as zip:
        for zip_name in zip.namelist():
            fn = dist / Path(zip_name).name
            print("... extracting", fn)
            with zip.open(zip_name) as src, open(fn, "wb") as dst:
                shutil.copyfileobj(src, dst)
