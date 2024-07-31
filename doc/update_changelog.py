from pathlib import Path
import re
import subprocess as subp
from packaging.version import Version, InvalidVersion
import datetime
import warnings
import sys

cwd = Path(__file__).parent


def parse_version_with_fallback(version_string):
    try:
        return Version(version_string)
    except InvalidVersion:
        return Version("0.0.1")


version = (
    subp.check_output([sys.executable, cwd.parent / "version.py"]).strip().decode()
)
new_version = parse_version_with_fallback(version)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    latest_tag = next(
        iter(
            sorted(
                (
                    parse_version_with_fallback(x)
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
previous_version = Version(m.group(1))
position = m.span(0)[0]

# sanity checks
assert new_version > previous_version, f"{new_version} > {previous_version}"

git_log = re.findall(
    r"[a-z0-9]+ ([^\n]+ \(#[0-9]+\))",
    subp.check_output(
        ["git", "log", "--oneline", f"v{previous_version}..HEAD"]
    ).decode(),
)

today = datetime.date.today()
header = f"{new_version} ({today.strftime('%B %d, %Y')})"

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
