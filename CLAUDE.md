# fitforgephotoai

A PUBLIC repo. The prototype that builds a synthetic dataset of garments draped on bodies with
exact ease and elasticity labels, to test whether a photo model can learn them. The spec of what
the FitForge repos would need to make it a real pipeline is `docs/spec-synthetic-ease-elasticity-dataset.md`.

- **Runs inside the solver's environment** (`../fitforgesolver/.venv`, built per that README) with
  this package installed editable into it. The solver and the body model are reached only through
  `photoai/engine.py`; nothing else imports them. Never import the solver's old dataset lane
  (`fitforge.pipeline`, `fitforge.surrogate`, `fitforge.grading`, `fitforge.retex`): it is being deleted.
- **Nothing generated is committed** (`outputs/`, `datasets/`): samples are built from licensed
  inputs (GarmentCode drafts on Anny bodies; CAESAR-derived plus-size bodies) and this repo is public.
- CPU only here; several workers each need their own `WARP_CACHE_PATH` (`scripts/run_workers.sh`).
- Every session: read `issues/`, commit before ending. Plain English in everything a person reads.
