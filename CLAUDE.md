# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BrainHacks 2026 — **SCALP**: a brain-signal-augmented computer use agent. Uses EEG signals as a runtime gate/feedback mechanism for a constrained desktop agent that relies on YOLO + OCR for visual state extraction instead of a VLM.

**V2 architecture:** Screen pixels → YOLO + OCR → structured text, combined with brain signals → encoder → discrete label, both fed into a text-only LLM agent that proposes and executes actions.

Key design docs: `scalp.md` (full architecture + brain signal types), `implementation-proposal.md` (MCP server design).

## Repository Structure

- Root level: project planning docs (README.md, scalp.md, implementation-proposal.md, names.md)
- `yolodex/`: Autonomous YOLO training data generation pipeline (a subproject with its own README, AGENTS.md, and docs/)

## Yolodex Pipeline

Pipeline stages: **collect** → **label** → **augment** → **train** → **eval** (loop until target mAP@50 reached).

### Setup and Running

```bash
cd yolodex && bash setup.sh     # install all deps (requires Python 3.11+, uv, ffmpeg)
```

**Autonomous loop:**
```bash
bash yolodex.sh                 # iterates via codex exec until target_accuracy met
```

**Manual skill execution:**
```bash
uv run .agents/skills/collect/scripts/run.py
bash .agents/skills/label/scripts/dispatch.sh 4   # parallel labeling with N agents
uv run .agents/skills/augment/scripts/run.py
uv run .agents/skills/train/scripts/run.py
uv run .agents/skills/eval/scripts/run.py
```

### Yolodex Architecture

- Skills live in `yolodex/.agents/skills/<name>/` — each has a `SKILL.md` and `scripts/` directory
- Orchestration: `yolodex.sh` (autonomous) + `AGENTS.md` (iteration logic read by codex)
- Config: `yolodex/config.json` — set `project` for named output under `runs/<project>/`
- Shared code: `yolodex/shared/utils.py` (BoundingBox helpers, `load_config()` resolves output_dir)
- State memory: `yolodex/progress.txt` (append-only cross-iteration log)
- Labeling supports multiple modes: `codex` (subagent, no API keys), `cua+sam`, `gemini`, `gpt`
- Parallel labeling uses git worktrees via `dispatch.sh`

### Key Conventions

- Python with type hints; use `uv run` to execute scripts (not raw python)
- YOLO model default: `yolov8n.pt` (Ultralytics)
- Vision labeling default: `gpt-5-nano`
- Output goes to `runs/<project>/` when project is set in config.json
- `OPENAI_API_KEY` env var required for vision labeling (not needed for `codex` label mode)
