"""iminuit version."""

# We use the semantic version scheme: MAJOR.MINOR.MAINTENANCE
# - See https://packaging.python.org/guides/distributing-packages-using-setuptools
# - Increase MAJOR when making breaking changes
# - Increase MINOR when adding backward-compatible features
# - Increase MAINTENANCE when fixing bugs without adding features
# - During development, add suffix .devN with N >= 0
# - For release candidates, add suffix .rcN with N >= 0
iminuit_version = "2.3.0"

# We list the corresponding ROOT version of the C++ Minuit2 library here
root_version = "v6-23-01-RF-binSampling-267-g2ef12408ee"

version = f"{iminuit_version}+ROOT-{root_version}"
