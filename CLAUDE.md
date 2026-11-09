# TRAKTOR ML

@AGENTS.md

Claude Code should treat `AGENTS.md` as the canonical project instructions. The current workflow is local, CPU-first development. Surrey HPC, Slurm, and Lightning AI Studio are optional legacy/remote infrastructure and must never be started or billed automatically.

Claude Code specifics:

* A `SessionStart` hook in `.claude/settings.json` injects `docs/STATUS.md` at the start of every session and after compaction.
* The `auditar` skill (`.claude/skills/auditar/`) governs audits, diagnoses, and comparisons.
* When compacting the conversation, preserve the decisions made in this session and who made them (Gabriel, often by listening, or the agent), the files modified, the open items and next steps, and any pending update to `docs/STATUS.md`.

Cloudflare Concept AI handoff (2026-11-09): before working on the remote DJ concept assistant or its local integration, read `docs/reports/traktor_concept_ai_cloudflare_2026-11-09.md` and the v3 Worker source snapshot `cloudflare/concept-ai-worker.js`. The Cloudflare deployment is currently manual; this Git snapshot is not automatically deployed. Do not run `/api/device/claim` until the local processor can complete claimed jobs.
