# Repository Guidelines

## Project Structure & Module Organization
- Core Python package lives in `sam3d_objects/` (`config/`, `model/`, `pipeline/`, `utils/`), with runnable entrypoints like `demo.py` and `gradio_demo.py` at repo root.
- Notebooks and images are under `notebook/` and `doc/`, while environment definitions sit in `environments/` and model assets/checkpoints belong in `checkpoints/<tag>/`.
- Patching utilities for upstream dependencies reside in `patching/`; keep custom fixes there instead of modifying vendored packages directly.

## Environment, Build, and Run
- Create the suggested environment with uv and install editable deps (extra
  index URLs are already declared in the requirements files):  
  `uv venv --python 3.11 .venv && source .venv/bin/activate`  
  `uv pip install -e '.[dev]' && uv pip install -e '.[p3d]' && uv pip install -e '.[inference]'`
- Run the single-image demo: `python demo.py`. Launch the Gradio UI locally: `python gradio_demo.py`.

## Coding Style & Naming Conventions
- Follow the repo’s preference for 2-space indentation and ~80-character lines (see `CONTRIBUTING.md`); use snake_case for files, modules, and variables, PascalCase for classes.
- Format with `black .` and lint with `flake8 .`; use `autoflake` to drop unused imports (`autoflake -r --in-place sam3d_objects tests`).
- Keep configs in YAML under `sam3d_objects/config/`; avoid committing secrets (HF tokens, AWS creds) to `*.yaml` or notebooks.

## Testing Guidelines
- Pytest is the expected harness. Place unit tests in `tests/` mirroring package paths (e.g., `tests/pipeline/test_inference_utils.py`).
- Name tests with `test_*` functions and files; prefer small, deterministic samples. Mock heavyweight GPU calls when possible.
- Run locally with `pytest` or `pytest tests -q`; add targeted runs for slow suites via markers you introduce (e.g., `@pytest.mark.slow`).

## Checkpoints & Assets
- Obtain checkpoints via the HF flow in `doc/setup.md`; store under `checkpoints/<tag>/` alongside `pipeline.yaml`. Do not commit downloaded weights.
- Large assets (PLY, PNG, MP4) belong in Git LFS or should be ignored; keep sample outputs compact.

## Commit & Pull Request Guidelines
- Commit messages in history use short, imperative summaries (e.g., “Update setup.md with HF access instructions.”). Follow that tone and keep subjects under ~72 characters.
- For PRs: describe the change set, runtime impact, and testing performed; link issues, and attach screenshots or GIFs for UI/visual output.
- Ensure lint/tests pass before opening a PR, and note any required checkpoints or data to reproduce results.
