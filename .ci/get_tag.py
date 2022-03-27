d = {}
with open("src/iminuit/version.py") as f:
    exec(f.read(), d)
print(d["version"])
