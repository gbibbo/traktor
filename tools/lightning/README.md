# tools/lightning — OPTIONAL / LEGACY

These scripts belong to the **retired** Lightning AI Studio workflow. The project is now
local, CPU-first (see `AGENTS.md`). They are kept here for provenance and are **inert by
default**: nothing in this repository runs them automatically.

**Never run these unless Gabriel explicitly asks in the current conversation.** They spend
Lightning credits and depend on `lightning_sdk` and a running Studio, so they do nothing on a
local workstation.

- `switch_machine.py` — switches the current Lightning Studio machine (CPU ↔ GPU) via
  `lightning_sdk`. Switching to a GPU spends credits. Requires a live Studio; fails otherwise.
- `gpu_autorun.sh` — was invoked from `~/.lightning_studio/on_start.sh` inside the Studio to run
  an unattended Phase 1 GPU extraction and then switch back to CPU. It hardcodes
  `/teamspace/studios/this_studio/traktor` and exits immediately when there is no GPU or no
  `artifacts/gpu_autorun.flag`, so it is a no-op outside Lightning.

There is intentionally **no** auto-approval for these scripts. The previous pre-approval in
`.claude/settings.json` (which let `switch_machine.py` spend credits without asking) has been
removed. Any use now requires explicit, in-conversation approval.

For the current local CPU workflow, use the commands in `docs/V4_USAGE.md` section 0.
