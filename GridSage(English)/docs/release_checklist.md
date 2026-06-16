# Release Checklist

Use this checklist before publishing a GridSage ZIP or GitHub release.

## Repository Hygiene

- [ ] `README.md` is in the actual project root.
- [ ] `README.md` and `README.zh-CN.md` are valid UTF-8 and have no mojibake.
- [ ] `LICENSE` exists.
- [ ] `.gitignore` exists.
- [ ] `frontend/node_modules/` is not tracked.
- [ ] `__pycache__/` and `*.pyc` are not tracked.
- [ ] Logs, TensorBoard outputs, runtime results, and local model weights are not tracked.
- [ ] Local research drafts and paper-writing artifacts are not included in the public release unless intentionally documented.

## Installation

- [ ] Python 3.11 virtual environment can be created from a fresh clone.
- [ ] `pip install -r backend/requirements.txt` succeeds.
- [ ] `npm install` succeeds in `frontend/`.
- [ ] Manual backend startup works with `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`.

## Verification

- [ ] Python source files parse successfully.
- [ ] `python -B -m unittest -v tests.test_backend_smoke` passes.
- [ ] `cd frontend && npm run build` passes.
- [ ] `/api/health` reports package and solver diagnostics clearly.
- [ ] `python test_run.py` is either passing or documented as requiring Gurobi.

## Release Notes

Document known limitations:

- Full baseline optimization requires Gurobi.
- Some workflows may require trained RL model files that are not included in the source repository.
- The current release is a research prototype, not a packaged Python or npm library.
