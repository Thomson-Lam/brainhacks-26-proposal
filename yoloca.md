# YOLOCA — YOLO + OCR Computer Agent

## Abstract

Current industry standard computer use agents that operate desktop applications autonomously primarily depend on Vision-Language Models (VLMs) to interpret screenshots, incurring high cost (~$0.01-0.03 per step), high latency (1-3 seconds per inference), and opaque visual reasoning. We propose **YOLOCA**, a novel architecture that decomposes VLM-based perception into two lightweight stages of vision and reasoning: a YOLO object detector that localizes UI elements and an OCR pass that extracts their text labels. The combined output is a structured text representation of the screen state, which a **text-only LLM** consumes to reason and act. This eliminates the need for a more compute intensive vision model at inference time. On constrained single-application benchmarks, YOLOCA can achieve superior performance, such as a 5-30x reduction in per-step cost and 2-3x reduction in latency compared to VLM-based agents. We incorporate **yolodex** in our training process, an autonomous pipeline that trains application-specific YOLO models from raw screen recordings with no explicit manual labeling.

> NOTE: the system architecture TBD, might not need OCR if the work is done during data labelling, and is reusable. Hence, a single pass would be more than enough here.

---

## Introduction

### Problem

Every major computer use agent today, such as Claude CUA, OpenAI CUA, ScreenAgent, follows the same pattern: capture a screenshot, send it to a VLM, and ask the model to interpret the visual state and decide what to do next. The VLM is doing three jobs at once: **perceiving** UI elements, **reading** their labels, and **reasoning** about the next action. This is powerful but wasteful. For a constrained task on a known application, a button is a button. A text field is a text field. The semantic meaning comes from the text label, not the pixels. Additionally, within a finite search space, a billion dollar foundational vision model is overkill, incurring excessive costs.

The cost and risk adds up as context increases exponentially. A single VLM screenshot analysis runs $0.01-0.03. An agent completing a 20-step task costs $0.20-0.60 per run. At production scale, such as thousands of tasks per day, the single VLM framework becomes prohibitive, with latency compounding at 1-3 seconds per VLM call means a 20-step task takes 20-60 seconds of inference time alone.

Additionally, images of prior states of the desktop may not be always beneficial to encode and compute attention over (the next action does not always depend on the previous one). This representation method is slow and costly, with the added latency making it harder for practical, end-to-end use. We argue that utilizing a textual representation of UI elements, and the semantic meaning of each element, in a text-native approach, is a more efficient and flexible method, with a mature context engineering ecosystem and best industry practices battle-tested and developed for LLM agents first.

In layman terms: the argument is that instead one big expensive model that does everything, which is unpredictable, slow and expensive, we break it down and use YOLO nano, which can be run on your laptop just fine, for the
vision work; we can train specialized YOLO models for specific applications or desktops for use, and make a LLM agent first approach to center the YOLO as submodules and tools for the central AI agent to call and use. 

Additional context optimizations can be explored upon later.

### Our Approach

YOLOCA separates the VLM's three jobs into specialized components:

1. **Perceive** — A YOLOv8n model detects UI elements (buttons, text fields, dropdowns, checkboxes) and outputs bounding boxes with class labels. Inference: ~5ms on CPU.
2. ((Optional)) **Read** — OCR (EasyOCR / Tesseract) runs on each cropped detection to extract the text label. "Submit", "Cancel", "Enter email" — the semantic content the LLM needs. Inference: ~50-200ms total.
3. **Reason** — A text-only LLM receives the structured text representation and decides the next action. No image tokens, no vision encoder — just text in, tool call out.

The LLM never sees pixels. It sees:

```
Detected UI elements:
- button "Submit" at (0.50, 0.30), confidence 0.94
- text_field "Enter email" at (0.50, 0.50), confidence 0.91
- button "Cancel" at (0.50, 0.70), confidence 0.89

Task: Submit the registration form.
Action history: [typed "user@email.com" in text_field at (0.50, 0.50)]
```

A text-only LLM can reason over this. It knows what "Submit" means, it knows the spatial layout, and it can decide the next action. No expensive and slow image encoding is done, and the attention mechanism can focus on rich textual meanings instead of accumulating images.

### Key Optimizations and Novelties

**1. YOLO + OCR as a perception-to-text bridge.** No published computer use agent decomposes VLM perception into YOLO + OCR and routes structured text to a text-only LLM. This is the core architectural contribution — it trades VLM generality for 5-30x cost reduction on constrained applications.

**2. CLI-first / GUI-fallback execution hierarchy.** Most computer use agents default to GUI pixel-clicking. But LLMs are text-native — `open -a Safari` is more reliable than "find the Safari icon and click it." YOLOCA establishes a strict hierarchy: programmatic execution first (bash, Playwright, code), GUI interaction only when no programmatic interface exists. This reduces the agent's dependence on visual perception for action execution.

**3. Visual assertion pattern.** YOLO's primary role is not action guidance — it's **verification**. After the agent executes an action (programmatically or via GUI), it calls `capture_and_detect()` to confirm the screen state changed as expected. YOLO becomes a visual assertion tool, analogous to how a human glances at the screen after clicking something.

**4. Autonomous training pipeline (yolodex).** Training an application-specific YOLO model typically requires manual labeling. Yolodex automates the full pipeline: record video of app usage → extract frames → label with vision models (GPT, Gemini, or Codex subagents) → augment (5x multiplier) → train YOLOv8n → evaluate → iterate until target mAP@50 is reached. Parallel labeling via git worktrees scales to large frame sets.

### Constrained Scope

Instead of targeting all desktop applications, we constrain the agent to **one specific app with 5-10 defined actions**:

- **YOLO classes become finite and small**: `button`, `text_field`, `dropdown`, `slider`, `checkbox` — 5-8 classes total
- **Training data is manageable**: 10-15 minutes of app usage → ~50-100 labeled frames → ~250-500 samples after augmentation — sufficient for YOLOv8n
- **Action space is enumerable**: `click(x, y)`, `type(text)`, `scroll(direction)`, keyboard shortcuts
- **Success criteria are clear**: "Did the agent complete the multi-step task?"

This constraint is what makes the YOLO + OCR decomposition viable. A general desktop agent needs VLM-level generalization. A constrained agent on a known application does not.

### Deliverables

1. **A trained YOLO model** for UI element detection on a target application, produced by the yolodex pipeline
2. **An MCP server** that exposes perception (YOLO + OCR) and action (click, type, run command) tools to any LLM agent
3. **A working agent demo** completing a constrained multi-step task on a specific application
4. **Cost and latency benchmarks** comparing YOLOCA against VLM-based computer use agents

---

## Risk Assessment & Strong Points

### 1. Runnable by TAs?

**YES.** The project is fully automated and reproducible:

- `cd yolodex && bash setup.sh` installs all dependencies (Python 3.11+, uv, ffmpeg)
- `bash yolodex.sh` runs the full training pipeline autonomously until target accuracy is met
- `config.json` drives all parameters — a TA edits the config, runs the pipeline, and gets a trained YOLO model
- The MCP server is a single Python process with standard dependencies (Ultralytics, EasyOCR/Tesseract, PyAutoGUI)
- Demo: clone repo, run setup, edit config for target app, run pipeline, launch MCP server, connect any MCP client, watch the agent complete a constrained task

### 2. Straightforward to implement?

**YES, with caveats.** The project decomposes into well-defined components:

- **Perception pipeline (YOLO + OCR → structured text):** Well-defined, mostly built via the yolodex training pipeline. Adding OCR on top of YOLO detections is a thin layer (EasyOCR/Tesseract on cropped bounding boxes)
- **Training pipeline (yolodex):** Fully implemented — collect → label → augment → train → eval, with parallel labeling and autonomous iteration
- **MCP server:** ~3-5 days of focused Python. The Python MCP SDK handles protocol boilerplate; the tools are thin wrappers around YOLO inference, OCR, PyAutoGUI, and subprocess
- **Main effort:** Context engineering — designing the LLM prompt so the text-only model makes good decisions from structured screen state. This is an iteration-heavy process, not a build-heavy one

### 3. Novel for undergrad final project?

**YES.** This is a genuine architectural contribution:

- Existing computer use agents (Claude CUA, OpenAI CUA, ScreenAgent) send full screenshots to VLMs for interpretation. No published work decomposes perception into YOLO + OCR and routes the structured text to a text-only LLM
- The approach provides **measurable comparisons** against VLM baselines: cost per step, latency per step, task completion rate
- The yolodex training pipeline (autonomous YOLO dataset generation from video) is itself a novel tool
- The CLI-first / GUI-fallback execution hierarchy is a design insight not present in existing computer use agent literature

### 4. Innovative / commercial potential?

**YES.** The cost structure is the key differentiator:

- **10-100x cost reduction per step:** A VLM screenshot analysis costs ~$0.01-0.03 per call. YOLO inference (~5ms on CPU) + OCR on a few crops + a text-only LLM call is 1-2 orders of magnitude cheaper
- **MCP server pattern is extensible:** Anyone can add tools, swap the LLM client, or integrate into their own workflows
- **Training pipeline is reusable:** yolodex can train YOLO models for any target application from video recordings
- **Commercial paths:** Open-source lightweight computer agent framework, perception-as-a-service API, or enterprise deployment where VLM costs are prohibitive at scale

---

## Methods and Processes 

We can benchmark processing time, accuracy, computational costs, and context rot on known computer use benchmarks and datasets.

**TODO: research more here!**


## System Architecture

### MCP Server Design

The agent is implemented as an **MCP server** that exposes tools to any MCP-compatible client (Claude Code, Claude Desktop, or a custom CLI).

```
┌──────────────────────────────────────────────────────┐
│ MCP Server (Python)                                  │
│                                                      │
│ Perception tools:                                    │
│ ├─ capture_and_detect()  → screenshot + YOLO + OCR   │
│ │   returns: list of {type, label, position, conf}   │
│ └─ verify_screen(expected) → YOLO check + match      │
│                                                      │
│ Action tools:                                        │
│ ├─ run_command(cmd)      → bash/subprocess           │
│ ├─ click(x, y)          → PyAutoGUI (GUI fallback)   │
│ ├─ type_text(text)       → PyAutoGUI (GUI fallback)  │
│ └─ browser_action(...)   → Playwright (web fallback) │
│                                                      │
│ Context resources:                                   │
│ ├─ action_history        → log of past actions       │
│ ├─ task_description      → current goal              │
│ └─ available_actions     → what the agent can do     │
└───────────────┬──────────────────────────────────────┘
                │ MCP protocol
┌───────────────▼──────────────────────────────────────┐
│ MCP Client (Claude Code, Claude Desktop, custom CLI) │
│ └─ LLM agent reasons over tools + context            │
│    └─ User sees tool calls + reasoning in terminal   │
└──────────────────────────────────────────────────────┘
```

Tool call visibility comes for free: the user sees every `capture_and_detect()` call, every `click()`, every action. No custom UI needed.

Additionally, a CLI-first fallback approach can be done for the purpose of this project.

### CLI-First / GUI-Fallback Execution Hierarchy

```
Can this be done programmatically?
  ├─ YES → run_command() / browser_action() / code execution
  │         then verify_screen() to confirm result
  │
  └─ NO (GUI-only surface) → capture_and_detect() → click() / type_text()
```

The two execution models cover different surfaces:

| Surface | CLI handles | GUI handles |
|---------|-------------|-------------|
| File operations | `cp`, `mv`, `mkdir` | N/A |
| App launching | `open -a X`, `xdg-open` | N/A |
| Web interaction | Playwright / Selenium | Fallback if no automation API |
| Custom desktop apps (no API) | N/A | YOLO + OCR → click |
| Games | N/A | YOLO + OCR → click |
| **Verification after any action** | N/A | `capture_and_detect()` confirms result |

Tool descriptions in the MCP schema enforce the hierarchy:

```
run_command:
  "Execute a shell command. PREFER this over GUI tools
   whenever the task can be done programmatically."

click:
  "Click at screen coordinates. Use ONLY when no
   programmatic alternative exists for the current task."
```

### Agent Decision Loop

```python
while task_not_complete:
    screenshot = capture_screen()
    detections = yolo_model.detect(screenshot)

    screen_state = []
    for det in detections:
        cropped = crop(screenshot, det.bbox)
        text_label = ocr(cropped)
        screen_state.append({
            "type": det.class_name,
            "label": text_label,
            "position": det.normalized_coords,
            "confidence": det.confidence
        })

    proposed_action = llm.decide(screen_state, task, history)
    execute(proposed_action)
    history.append(proposed_action)

    # Visual assertion: verify the action had the expected effect
    verification = capture_and_detect()
    if not matches_expected_state(verification, proposed_action):
        history.append(f"action '{proposed_action}' may have failed — unexpected screen state")
        # LLM re-plans on next iteration
```

---

## Processes and Methods

### Perception Pipeline: Screenshot → YOLO → OCR → Structured Text

**Step 1: Screenshot capture.** Capture the current screen state as a full-resolution image.

**Step 2: YOLO inference.** Run the trained YOLOv8n model on the screenshot. The model outputs bounding boxes with class labels (`button`, `text_field`, `dropdown`, etc.) and confidence scores. Inference takes ~5ms on CPU.

**Step 3: OCR on cropped regions.** For each YOLO detection, crop the bounding box region from the screenshot and run OCR (EasyOCR or Tesseract) to extract the text label. This is what distinguishes "Submit" from "Cancel" — YOLO knows they're both buttons, OCR tells us which button is which.

**Step 4: Structured text formatting.** Format the results as structured text that the LLM can consume:

```
Detected UI elements:
- button "Submit" at (0.50, 0.30), confidence 0.94
- text_field "Enter email" at (0.50, 0.50), confidence 0.91
- button "Cancel" at (0.50, 0.70), confidence 0.89
```

Coordinates are normalized (0-1 range) for resolution independence. The LLM receives this as part of its prompt context, alongside the task description and action history.

### LLM Agent Consumption

The text-only LLM receives:

1. **Current screen state** — structured text from the perception pipeline
2. **Task description** — what the agent is trying to accomplish
3. **Action history** — log of past actions and their outcomes
4. **Available actions** — what tools the agent can call

The LLM reasons over this text context and outputs a tool call (e.g., `click(0.50, 0.30)` to click the "Submit" button, or `run_command("open -a Safari")` for a programmatic action).

---

## Training Pipeline

### Yolodex: Autonomous YOLO Training

The yolodex pipeline trains YOLO models from raw video with minimal manual intervention. Five stages run in sequence, iterating until the target accuracy (mAP@50) is reached.

```
Video source → Collect → Label → Augment → Train → Eval
                                                      │
                                              meets target?
                                              ├─ YES → done
                                              └─ NO  → loop
```

#### Collect

Download or copy a source video and extract frames at configurable FPS.

- **Input:** YouTube URL or local video file (set in `config.json`)
- **Process:** Download with `yt-dlp`, extract frames with `ffmpeg` at configured FPS (default: 1 fps)
- **Output:** `frames/frame_000001.jpg`, `frame_000002.jpg`, etc.

For a target desktop application: record 10-15 minutes of yourself using the app, producing ~600-900 frames at 1 fps.

#### Label

Annotate frames with bounding boxes using one of four labeling strategies:

| Mode | Method | Requirements |
|------|--------|--------------|
| `codex` | Codex subagents view images directly | None (no API keys) |
| `gpt` | GPT vision with structured output | `OPENAI_API_KEY` |
| `gemini` | Google Gemini native bbox detection | `GEMINI_API_KEY` |
| `cua+sam` | OpenAI CUA clicks + SAM segmentation | `OPENAI_API_KEY` |

Output format: YOLO normalized coordinates — `class_id center_x center_y norm_width norm_height` — one detection per line in a `.txt` file alongside each frame.

#### Parallel Labeling with Git Worktrees

For large frame sets, labeling can be parallelized:

```bash
bash .agents/skills/label/scripts/dispatch.sh 4   # 4 parallel workers
```

The dispatch script:
1. Creates N independent git worktrees under `/tmp/yolodex-workers/`
2. Splits frames into batches (`batch_size = ceil(total_frames / num_agents)`)
3. Each worktree runs labeling independently via `codex exec`
4. After completion, `merge_classes.py` unifies class indices across workers

#### Augment

Generate 4 synthetic variations per labeled frame:

| Augmentation | Method | Label adjustment |
|-------------|--------|-----------------|
| Horizontal flip | Mirror image | Invert bounding box x-coordinates |
| Brightness jitter | Random factor [0.6, 1.4] | None |
| Contrast jitter | Random factor [0.7, 1.3] | None |
| Gaussian noise | std=15.0, clipped [0, 255] | None |

**Dataset growth:** N labeled frames → 5N total samples (1 original + 4 augmented).

~50-100 raw frames → ~250-500 training samples after augmentation — sufficient for YOLOv8n on a small class set.

#### Train

Split dataset (default 80/20 train/val), generate Ultralytics `dataset.yaml`, and train:

- **Model:** YOLOv8n (nano) — lightweight, fast inference
- **Image size:** 640px
- **Epochs:** 50 (configurable)
- **Output:** `weights/best.pt`

#### Eval

Evaluate the trained model and decide whether to iterate:

- **Metrics:** mAP@50, mAP@50-95, precision, recall, per-class AP@50
- **Target:** mAP@50 ≥ 0.75 (configurable via `target_accuracy`)
- **Failure analysis:** Identifies the 3 weakest classes by AP@50, enabling targeted re-labeling or additional data collection

If the target is not met, the pipeline loops: collect more data or re-label problem classes, augment, re-train, re-eval.

### Data Requirements

For a constrained application with 5-8 UI element classes:

| Stage | Volume |
|-------|--------|
| Raw video | 10-15 minutes of app usage |
| Extracted frames | ~600-900 at 1 fps |
| Labeled frames | 50-100 minimum (can label a subset) |
| After augmentation | 250-500 training samples |
| YOLOv8n training | ~50 epochs, minutes on GPU |

---

## Cost Analysis

### Per-Step Cost Comparison

| Component | VLM approach | YOLOCA approach |
|-----------|-------------|-----------------|
| Screenshot analysis | VLM call: ~$0.01-0.03 | YOLO inference: ~free (local, ~5ms CPU) |
| Text extraction | Included in VLM | OCR on crops: ~free (local, ~50-200ms) |
| Action reasoning | Included in VLM | Text-only LLM: ~$0.001-0.005 |
| **Total per step** | **~$0.01-0.03** | **~$0.001-0.005** |
| **Cost reduction** | baseline | **5-30x cheaper** |

For agents that take 10-50 steps per task, the savings compound. At scale (thousands of tasks), YOLOCA is orders of magnitude cheaper.

### Latency Comparison

| Stage | VLM approach | YOLOCA approach |
|-------|-------------|-----------------|
| Screenshot → understanding | 1-3 seconds (VLM inference) | ~5ms YOLO + ~100ms OCR |
| Action reasoning | Included above | ~0.5-1s (text-only LLM) |
| **Total per step** | **1-3 seconds** | **~0.6-1.1 seconds** |

The YOLOCA pipeline is faster because YOLO and OCR run locally, and text-only LLM calls have lower latency than VLM calls (smaller input, no image encoding).

### Scaling Characteristics

| Dimension | VLM approach | YOLOCA approach |
|-----------|-------------|-----------------|
| Cost per 1K tasks (20 steps each) | $200-600 | $20-100 |
| Local compute | None (API-only) | YOLO + OCR (CPU, no GPU needed) |
| Model retraining | N/A | Required per target app |
| New app support | Zero-shot (VLM generalizes) | Requires yolodex training (~1 hour pipeline) |
| Offline capability | No (API dependency) | Partial (perception runs locally, only LLM needs API) |

**Trade-off:** VLMs generalize to any app zero-shot. YOLOCA requires training a YOLO model per target application. The payoff is dramatically lower per-step cost and latency, making it viable for high-volume or latency-sensitive use cases.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| YOLO can't differentiate UI elements | Low | With a constrained app and small class set (5-8 classes), YOLO handles this. Buttons, text fields, and checkboxes are visually distinct. |
| OCR quality on target app | Medium | Test early. Custom fonts, icon-only buttons, or heavy styling may degrade OCR — use hardcoded label mappings for known icons as fallback. |
| Training data volume | Low | For 5-8 UI classes, ~200-500 labeled frames is sufficient for YOLOv8n. Yolodex augmentation (5x multiplier) gets there from ~50-100 raw frames. |
| Action execution reliability | Low | PyAutoGUI maps normalized YOLO coords → screen coords → mouse click. Straightforward. CLI-first hierarchy reduces reliance on GUI clicking. |
| Context engineering for LLM | Medium | Prompt design matters — the LLM needs current state, action history, task goal, and available actions. Getting this right takes iteration, not engineering. |
| End-to-end integration | Medium | MCP server + client architecture provides clean boundaries. Components can be tested independently. |

---

## Implementation Effort

| Component | Effort | Notes |
|-----------|--------|-------|
| YOLO training pipeline (yolodex) | Done | Fully implemented — collect, label, augment, train, eval |
| OCR integration | Low | EasyOCR / Tesseract on cropped YOLO detections. A few lines per detection. |
| MCP server boilerplate | Low | Python MCP SDK handles protocol. Tool registration is straightforward. |
| YOLO integration in server | Low | Load model, run inference, return JSON — yolodex already has this pattern. |
| Action execution tools | Low | PyAutoGUI + subprocess. Simple wrappers. |
| State tracking / history | Low | Append-only list in memory. |
| CLI frontend | Free | Use Claude Code or Claude Desktop as the MCP client. |
| Context engineering / prompts | Medium | Iteration-heavy. Design structured text format, test on target app. |
| **Total** | **~1-2 weeks** | MCP server is ~3-5 days. Context engineering is the rest. |

---

## Sources

- Yolodex pipeline: see `yolodex/` directory
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- PyAutoGUI: https://pyautogui.readthedocs.io/
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- MCP (Model Context Protocol): https://modelcontextprotocol.io/
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
