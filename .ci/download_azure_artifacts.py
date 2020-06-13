import requests
import json
import shutil
from urllib.request import urlopen
from pathlib import Path
import zipfile
import tempfile

organization = "scikit-hep"
project = "iMinuit"


def get(url):
    return json.loads(requests.get(url).content)


def get_latest_successful_build():
    # get list of all builds
    builds = get(
        f"https://dev.azure.com/{organization}/{project}/_apis/build/builds?api-version=5.0"
    )
    finishTime, id, number = sorted(
        (
            (b["finishTime"], b["id"], b["buildNumber"])
            for b in builds["value"]
            if b.get("result", None) == "succeeded"
        ),
        reverse=True,
    )[0]
    return id, number, finishTime


def get_artifacts(buildId):
    artifacts = get(
        f"https://dev.azure.com/{organization}/{project}/_apis/build/builds/{buildId}/artifacts?api-version=5.1"
    )

    for item in artifacts["value"]:
        local_filename = item["name"] + ".zip"
        url = item["resource"]["downloadUrl"]
        yield local_filename, url


buildId, buildNumber, finishTime = get_latest_successful_build()
print(f"Downloading artifacts for {buildNumber}, finished {finishTime}")

dist = Path("dist")
if dist.exists():
    if any(dist.iterdir()):
        raise SystemExit("Error: ./dist directory is not empty; please delete content")
else:
    dist.mkdir()

for fn, url in get_artifacts(buildId):
    print("Downloading", fn, "...")
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        with urlopen(url) as src:
            shutil.copyfileobj(src, tmp)
        with zipfile.ZipFile(tmp) as zip:
            for zip_name in zip.namelist():
                fn = dist / Path(zip_name).name
                print("... extracting", fn)
                with zip.open(zip_name) as src, open(fn, "wb") as dst:
                    shutil.copyfileobj(src, dst)
