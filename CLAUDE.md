# TRAKTOR ML

@AGENTS.md

Claude Code should treat `AGENTS.md` as the canonical project instructions. The current workflow is local, CPU-first development. Surrey HPC, Slurm, and Lightning AI Studio are optional legacy/remote infrastructure and must never be started or billed automatically.

Claude Code specifics:

* A `SessionStart` hook in `.claude/settings.json` injects `docs/STATUS.md` at the start of every session and after compaction.
* The `auditar` skill (`.claude/skills/auditar/`) governs audits, diagnoses, and comparisons.
* When compacting the conversation, preserve the decisions made in this session and who made them (Gabriel, often by listening, or the agent), the files modified, the open items and next steps, and any pending update to `docs/STATUS.md`.