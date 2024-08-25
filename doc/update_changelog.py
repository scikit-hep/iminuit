from pathlib import Path
import re
import subprocess as subp
import datetime
import warnings
import sys
from iminuit._parse_version import parse_version

cwd = Path(__file__).parent


new_version_string = (
    subp.check_output([sys.executable, cwd.parent / "version.py"]).strip().decode()
)
new_version = parse_version(new_version_string)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    latest_tag = next(
        iter(
            sorted(
                (
                    parse_version(x.lstrip("v"))
                    for x in subp.check_output(["git", "tag"])
                    .decode()
                    .strip()
                    .split("\n")
                ),
                reverse=True,
            )
        )
    )

with open(cwd / "changelog.rst") as f:
    content = f.read()

# find latest entry
m = re.search(r"([0-9]+\.[0-9]+\.[0-9]+) \([^\)]+\)\n-*", content, re.MULTILINE)
assert m is not None
previous_version_string = m.group(1)
previous_version = parse_version(previous_version_string)
position = m.span(0)[0]

# sanity checks
assert new_version > previous_version, f"{new_version} > {previous_version}"

git_log = re.findall(
    r"[a-z0-9]+ ([^\n]+ \(#[0-9]+\))",
    subp.check_output(
        ["git", "log", "--oneline", f"v{previous_version_string}..HEAD"]
    ).decode(),
)

today = datetime.date.today()
header = f"{new_version_string} ({today.strftime('%B %d, %Y')})"

new_content = f"{header}\n{'-' * len(header)}\n"
if git_log:
    for x in git_log:
        if x.startswith("[pre-commit.ci]"):
            continue
        x = re.sub(
            "#([0-9]+)", r"`#\1 <https://github.com/scikit-hep/iminuit/pull/\1>`_", x
        )
        new_content += f"- {x.capitalize()}\n"
else:
    new_content += "- Minor improvements\n"
new_content += "\n"

print(new_content, end="")

with open(cwd / "changelog.rst", "w") as f:
    f.write(f"{content[:position]}{new_content}{content[position:]}")
