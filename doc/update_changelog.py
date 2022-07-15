from pathlib import PurePath as Path
import re
import subprocess as subp
from pkg_resources import parse_version
import datetime

cwd = Path(__file__).parent

with open(cwd.parent / "src/iminuit/version.py") as f:
    version = {}
    exec(f.read(), version)
    new_version = parse_version(version["version"])

latest_tag = next(
    iter(
        sorted(
            (
                parse_version(x)
                for x in subp.check_output(["git", "tag"]).decode().strip().split("\n")
            ),
            reverse=True,
        )
    )
)

with open(cwd / "changelog.rst") as f:
    content = f.read()

# find latest entry
m = re.search(r"([0-9]+\.[0-9]+\.[0-9]+) \([^\)]+\)\n-*", content, re.MULTILINE)
previous_version = parse_version(m.group(1))
position = m.span(0)[0]

# sanity checks
assert previous_version == latest_tag
assert new_version > previous_version

git_log = re.findall(
    r"[a-z0-9]+ ([^\n]+ \(#[0-9]+\))",
    subp.check_output(
        ["git", "log", "--oneline", f"v{previous_version}..HEAD"]
    ).decode(),
)

today = datetime.date.today()
header = f"{new_version} ({today.strftime('%B %d, %Y')})"
content2 = f"""{content[:position]}{header}
{'-' * len(header)}
"""
for x in git_log:
    content2 += f"- {x}\n"
content2 += "\n"
content2 += content[position:]

with open(cwd / "changelog.rst", "w") as f:
    f.write(content2)
