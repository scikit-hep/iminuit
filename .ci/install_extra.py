import importlib_metadata
import subprocess as subp

targets = []
for x in importlib_metadata.requires("iminuit"):
    if ";" not in x:
        continue
    a, b = x.split(";")
    if b == ' extra == "test"':
        targets.append(a)

subp.check_call(
    ["python", "-m", "pip", "install", "--prefer-binary"] + targets, stdout=subp.DEVNULL
)
