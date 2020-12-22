iminuit logo
============

The iminuit logo uses glyphs generated from the free Gentium Plus font.

How to make a release
=====================

- On develop
  - Update `src/iminuit/version.py` to released version
  - Update `doc/changelog.rst` to released version and add today's date
  - Note: A prerelease can be published simply by adding `.rcN` to `iminuit_version`,
    where N is a number >= 0
- Merge develop into master
  - Option: do a `git rebase -i` to make history edits
- Check that all wheels are build and run fine
  - If there are problems with the wheels, create a beta/<release-version> branch and
    commit fixes there, once the problem is fixed, squash-merge the branch to master
- Create release on Github
  - Message: [See changelog on RTD](https://iminuit.readthedocs.io/en/stable/changelog.html)
- This triggers an upload of the latest build artefacts to PyPI
  - If this fails unexpectedly, download the artefact from the actions page and upload
    it manually with twine, then fix the issue for the next release
  - Upload uses API tokens configured in PyPI and Github "Secrets"
- conda-forge should pick this up automatically and generate conda packages
