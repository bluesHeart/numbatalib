# Releasing `numbatalib` to PyPI

## Recommended order

1. Push to GitHub (public repo) and tag a release.
2. Upload to TestPyPI.
3. Upload to PyPI.

Publishing after a Git tag makes it easy for users to audit the exact source for each version.

## Credentials

This repo does not store PyPI credentials. Use one of:

- **API token + Twine** (manual): set `TWINE_USERNAME=__token__` and `TWINE_PASSWORD=pypi-...`
- **Trusted Publishing** (GitHub Actions): configure a trusted publisher on PyPI, then push a tag `vX.Y.Z`

## Pre-flight

- Run tests: `pytest -q`
- (Optional) Update parity/bench CSVs: `python tools/compare_vs_talib.py --bench --write-checklist`

## Build

- Clean old artifacts: `Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue`
- Build: `python -m build`
- Validate metadata: `twine check dist/*`

## Upload

- (Recommended) Upload to TestPyPI first:
  - `twine upload -r testpypi dist/*`
  - Verify install: `python -m pip install -U --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple numbatalib`
- Upload to PyPI:
  - `twine upload dist/*`

## Post-release

- Tag the release (example): `git tag v0.1.0 && git push --tags`
