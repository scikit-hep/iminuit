"""Get the latest Python release which is online."""

import urllib.request
import re
from html.parser import HTMLParser
import gzip
from packaging.version import Version


class PythonVersionParser(HTMLParser):
    """Specialized HTMLParser to get Python version number."""

    def __init__(self):
        super().__init__()
        self.versions = set()
        self.found_version = False

    def handle_starttag(self, tag, attrs):
        """Look for the right tag and store result in an attribute."""
        if tag == "a":
            for attr in attrs:
                if attr[0] == "href" and "/downloads/release/python-" in attr[1]:
                    self.found_version = True
                    return

    def handle_data(self, data):
        """Extract Python version from entry."""
        if self.found_version:
            self.found_version = False
            match = re.search(r"Python (\d+\.\d+)", data)
            if match:
                self.versions.add(Version(match.group(1)))


def versions():
    """Get all Python release versions."""
    req = urllib.request.Request("https://www.python.org/downloads/")
    req.add_header("Accept-Encoding", "gzip")

    with urllib.request.urlopen(req) as response:
        raw = response.read()
        if response.info().get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        html = raw.decode("utf-8")

    parser = PythonVersionParser()
    parser.feed(html)

    return parser.versions


def latest():
    """Return version of latest Python release."""
    return max(versions())


def main():
    """Print all discovered release versions."""
    print(" ".join(str(x) for x in sorted(versions())))


if __name__ == "__main__":
    main()
