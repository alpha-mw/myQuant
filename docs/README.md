# Documentation Index

Current documentation describes the single mainline, the research pipeline, and
the live workspace modules. The repository launcher for the research workspace
is `./run_web.sh`; the Python CLI entrypoint is `quant-investor`.

## Architecture

- [Entrypoints and Versioning](architecture/entrypoints_and_versioning.md)
- [Research Pipeline and Protocols](architecture/research_pipeline_and_protocols.md)

## Modules

- [Module Map](modules/module_map.md)
- [Macro v2 Observer Contract](modules/macro_v2_observer.md)
- [Macro Risk Reference](modules/macro_risk_reference.md)

## Runtime Notes

- `quant-investor research run` executes the current single mainline.
- `quant-investor market maintain --staged` runs bounded CN staged maintenance:
  it validates Parquet canonical health, fills only target-date gaps in limited
  batches, and writes progress artifacts without blocking formal review.
- `quant-investor market maintain` without `--staged` preserves the legacy
  maintenance path for explicit full refresh work.
- `quant-investor market download` is a compatibility alias for the maintenance
  path.
- `quant-investor web` serves the FastAPI research workspace backend and the
  React/Vite research workspace frontend boundary.
