from docutils import nodes
from docutils.parsers.rst import Directive
from pathlib import Path
import json


class Tutorials(Directive):
    def run(self):

        repo = "scikit-hep/iminuit"
        branch = "develop"
        nbviewer = "https://nbviewer.jupyter.org/github"

        tutorials = (Path(__file__).parent.parent.parent / "tutorial").glob("*.ipynb")

        paragraphs = []
        for t in sorted(tutorials):
            label = t.stem
            with open(t) as f:
                d = json.load(f)
                first = d["cells"][0]
                line = first["source"][0]
                if line.startswith("#"):
                    label = line[1:].strip()

            url = f"{nbviewer}/{repo}/blob/{branch}/tutorial/{t.name}"
            p = nodes.paragraph(
                "",
                "",
                nodes.reference("", label, internal=False, refuri=url),
            )
            paragraphs.append(p)
        return paragraphs


def setup(app):

    app.add_directive("tutorials", Tutorials)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
