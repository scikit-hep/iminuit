iminuit logo
============

The iminuit logo uses glyphs generated from the free Gentium Plus font.

How to make a release
=====================
- create a branch on the master, called release_<version>
- merge develop into this branch and do release edits there
- increase version number in iminuit/version.py
- update doc/changelog.rst
- check that new tutorials are listed in the tutorials section of the docs
- run `make integration` to do integration tests (if these fail, add tests to iminuit!)
- run `.ci/download_azure_artifacts.py` to download all wheels from the latest pipeline
- run `python -m twine upload dist/*` if everything looks ok
  (missing files can be uploaded later, but existing files cannot be overridden!)
- merge release branch to master (do not squash!)
- create release on Github
- conda-forge should pick this up automatically and generate conda packages
