# Contributing

Thank you for helping. Issues and pull requests are welcome, in English or French.

## Development setup

The repository is a [uv](https://docs.astral.sh/uv/) workspace with one lockfile (`uv.lock`) for
the package (`libs/langchain-ivoire`).

```bash
uv sync                      # Python env with the package (editable) and the dev tools
pre-commit install           # optional: ruff, uv-lock, gitleaks, mypy on every commit
```


## Checks

```bash
cd libs/langchain-ivoire
uv run pytest tests/unit_tests                      # network is blocked (pytest-socket)
uv run pytest tests/unit_tests --cov=langchain_ivoire --cov-fail-under=85
uv run ruff check . && uv run ruff format --check .
uv run mypy langchain_ivoire tests                  # strict
cd ../.. && uv run pytest benchmarks                # scoring code
```

Integration tests talk to a real server and are skipped unless you point them at one:

```bash
IVOIRE_TEST_BASE_URL=http://<host>:8090 IVOIRE_TEST_AUDIO=/path/to/come_and_eat.wav \
  uv run pytest tests/integration_tests -m integration
```

## Rules of the road

- Unit tests never use the network; mock HTTP with `respx`.
- Never commit secrets, real API keys, private recordings or test-set texts. Benchmark results hold
  scores only.
- Keep the remote client light: no new required dependency without discussion.
- Public APIs need docstrings and type hints (`mypy --strict` must pass).
- Add a line under `## [Unreleased]` in `CHANGELOG.md`.
- Translation and speech claims need numbers with set sizes and confidence intervals.

## Releasing (maintainers)

One-time setup:

1. On GitHub, create two environments: `testpypi`, and `pypi` with **required reviewers** and a
   deployment tag rule for `v*`.
2. On TestPyPI and PyPI, add a **trusted publisher** (pending publisher if the project does not exist
   yet) with owner `AbyshireAI`, repository `langchain-ivoire`, workflow `release.yml`:

   | Index    | Project            | Environment       |
   |----------|--------------------|-------------------|
   | TestPyPI | `langchain-ivoire` | `testpypi`        |
   | PyPI     | `langchain-ivoire` | `pypi`            |

3. Enable **private vulnerability reporting** (Settings → Code security) for `SECURITY.md`.

Each release:

1. Set the version in `libs/langchain-ivoire/pyproject.toml` and
   `libs/langchain-ivoire/langchain_ivoire/_version.py` (PEP 440, e.g. `0.1.0a1` for a preview); run
   `uv lock`; date the `CHANGELOG.md` entry.
2. Merge to `main`, then `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. The `Release` workflow builds, publishes to TestPyPI, smoke-tests the upload, waits for approval
   of the `pypi` environment, publishes to PyPI with attestations, and creates the GitHub release
   with the CycloneDX SBOMs.
