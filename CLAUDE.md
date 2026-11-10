# TRAKTOR ML

@AGENTS.md

Claude Code should treat `AGENTS.md` as the canonical project instructions. The current workflow is local, CPU-first development; heavy work goes to the gpuq GPU queue with its data on Gabriel's Google Drive, following the procedure in the AGENTS.md Compute section and `docs/GPUQ.md`. Surrey HPC, Slurm, and Lightning AI Studio are optional legacy/remote infrastructure and must never be started or billed automatically.

Claude Code specifics:

* A `SessionStart` hook in `.claude/settings.json` injects `docs/STATUS.md` at the start of every session and after compaction.
* The `auditar` skill (`.claude/skills/auditar/`) governs audits, diagnoses, and comparisons.
* When compacting the conversation, preserve the decisions made in this session and who made them (Gabriel, often by listening, or the agent), the files modified, the open items and next steps, and any pending update to `docs/STATUS.md`.

Cloudflare Concept AI (2026-11-09): before working on the phone concept assistant, read `cloudflare/README.md` (the phone-to-playlist circuit, Worker v5), `docs/reports/clap_texto_2026-11-09.md` (why the LLM instructions say what they say) and the handoff `docs/reports/traktor_concept_ai_cloudflare_2026-11-09.md`. Deploying to production is Gabriel's step (`python cloudflare/publicar.py`); nothing deploys automatically. Tests against the real queue must use `--job <id>` so they do not consume real concepts.
