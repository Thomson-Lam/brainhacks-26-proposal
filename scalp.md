# SCALP : Brain-Signal-Augmented Computer Use Agent

## Problem

Computer use agents are expensive and unreliable across long context windows due to the vision + language + agentic complexity.

This project aims to explore integration of implicit brain signals as a new source of data to augment computer use agents, and to make computer use agents more reliable and safe across long multi-step scenarios.

## Goals

- Build an EEG headset (Ultracortex Mark III + PiEEG, 8 channels)
- Sample brain signal data from the EEG
- Build a constrained computer use agent that uses YOLO + OCR instead of a VLM for visual state extraction, with brain signals as a runtime feedback/gating mechanism

---

## Architecture Evolution

### V1: Initial Proposal (Rejected)

Use Claude Code or Codex, and use the Yolodex skill to train lightweight YOLO vision transformer nano models for computer use classification. Instead of having a single big model, use specialized YOLO vision models as subagents, and use the original computer use model as a fall back or as an orchestrator instead. Contextual metadata or information is gathered and depending on brain signals, the agent decides on the next best course of action. Instead of allowing a model trained on brain signals to make decisions, this architecture relies on the intelligent agent to make the final call, especially for breaking down long and abstract ideas into actionable and verifiable steps to ensure task completion beyond just the triggering of a single action.

```
Brain signals -> lightweight encoder/classifier -> + metadata -> AI agent (VLM) -> trigger YOLO subagents or vision model takes over
```

Additional system optimization: provide caching mechanisms beyond training knowledge:

1. User performs a slightly newer or unknown action -> click button
2. Brain signal + contextual information is recorded
3. This is saved:
   1. When a similar brain signal is detected again, check for similarity and load context to the AI agent, or execute a similar action again
   2. The data is stored and a data pipeline to update models live, or for training, is run. The YOLO submodel(s) learn of the task and no storage of specific action is needed.

#### Why V1 was rejected

1. **The YOLO ensemble added complexity without removing the expensive model.** The AI agent in the middle still needed to be a capable VLM/LLM to interpret context and make decisions. YOLO subagents could tell you *where* things are on screen, but not *what to do with them*. The expensive model wasn't eliminated — complexity was added around it.

2. **"Brain signals" was dangerously underspecified.** The proposal never named what neural signal was being targeted (P300? SSVEP? Motor imagery? ERN?). Without this, data collection would produce uninterpretable results.

3. **The caching/learning loop was a second research project.** Online learning + retrieval-augmented action execution is a multi-month engineering effort on its own.

4. **Full scope was not feasible in 1 month.**

---

### V2: Revised Architecture (Current)

The key insight: LLM agents consume text, but brain signals are tensors and screen pixels are images. Rather than using an expensive VLM to process screenshots, we can use **YOLO + OCR** to convert the screen into structured text, and a **brain signal encoder** to convert EEG into a discrete label. A **text-only LLM** then reasons over both streams.

```
Screen pixels ──→ YOLO (detect UI elements + bounding boxes)
                    → OCR on each detection (extract text labels)
                        → structured text ──┐
                                            ├──→ LLM agent (text-only) ──→ action
Brain signals ──→ encoder/classifier        │
                    → discrete label ───────┘
```

#### Why this works

1. **YOLO as a perception-to-text bridge is proven.** The yolodex pipeline already does this — a trained detector outputs bounding boxes + class labels, which are structured enough for an LLM to consume as context. No vision model needed downstream.

2. **OCR closes the semantic gap.** YOLO alone can't tell "Submit" from "Cancel" — they're both buttons. But running OCR (Tesseract or EasyOCR) on each detected bounding box extracts the text label. The LLM then knows exactly what each element says and means. This turns the problem into a **context engineering** challenge: can we provide enough structured text from the screen for a text-only LLM to make good decisions?

3. **Clear separation of concerns.** Perception (YOLO + OCR), brain state (encoder), reasoning (LLM) — each component has a well-defined interface. They can be developed and tested independently.

4. **Cost structure makes sense.** One YOLO inference is ~5ms on CPU. OCR on a few cropped regions is fast. A text-only LLM call is cheaper than a VLM call. Brain encoder inference is negligible. The whole pipeline is cheaper per step than a single VLM screenshot analysis.

5. **Brain signals as a reward function / soft constraint.** In RL terms: YOLO + OCR extracts the **state**, the LLM proposes an **action**, and the brain signal provides a **reward/constraint** before execution. We're not training with the reward (that's model-free RL and out of scope). We're using it as a **runtime gate** — the LLM proposes, the brain signal confirms or vetoes, then the action executes or the LLM re-plans.

6. **Yolodex's training pipeline is directly reusable.** Video → frames → label → augment → train → eval. The tooling already exists.

#### Constrained scope for feasibility

Instead of targeting all desktop applications, we constrain the agent to **one specific app with 5-10 defined actions**. This changes everything:

- **YOLO classes become finite and small**: `button`, `text_field`, `dropdown`, `slider`, `checkbox`: ~5-8 classes total
- **Training data is manageable**: Record yourself using the app for 10-15 minutes, run yolodex pipeline. With augmentation (5x multiplier), ~50-100 raw frames yields ~250-500 training samples, which is sufficient for YOLOv8n
- **Action space is enumerable**: `click(x, y)`, `type(text)`, `scroll(direction)`, keyboard shortcuts are only a handful of actions
- **Success criteria are clear**: "Did it complete the 5-step task?"

#### What the LLM receives

```
Detected UI elements:
- button "Submit" at (0.50, 0.30), confidence 0.94
- text_field "Enter email" at (0.50, 0.50), confidence 0.91
- button "Cancel" at (0.50, 0.70), confidence 0.89

Brain signal: user_confirms_intent

Task: Submit the registration form.
Action history: [typed "user@email.com" in text_field at (0.50, 0.50)]
```

A text-only LLM can reason over this. It knows what "Submit" means, it knows the spatial layout, and it has the brain signal as a go/no-go constraint.

#### Agent loop pseudocode

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

    brain_feedback = encoder.classify(eeg_window)

    if brain_feedback == "confirm":
        execute(proposed_action)
        history.append(proposed_action)
    else:
        history.append(f"action '{proposed_action}' vetoed by brain signal")
        # LLM re-plans on next iteration
```

---

## Risk Assessment (V2)

| Risk                                 | Severity   | Notes                                                                                                                                                                                        |
| ------------------------------------ | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| YOLO can't differentiate UI elements | **Low**    | With a constrained app and small class set (5-8 classes), YOLO handles this. Buttons, text fields, and checkboxes are visually distinct.                                                     |
| Semantic meaning of elements         | **Low**    | OCR solves this. Tesseract/EasyOCR on cropped YOLO detections extracts the label text.                                                                                                       |
| Training data volume                 | **Medium** | For 5-8 UI classes in one app, ~200-500 labeled frames is sufficient for YOLOv8n. Yolodex with augmentation (5x multiplier) gets there from ~50-100 raw frames.                              |
| Action execution                     | **Low**    | PyAutoGUI maps normalized YOLO coords → screen coords → mouse click. Straightforward.                                                                                                        |
| Brain signal encoder                 | **High**   | Hardest component. Needs a concrete signal type (SSVEP recommended for reliability with budget hardware) and a working classifier.                                                           |
| EEG hardware build                   | **High**   | 2+ weeks for 3D printing, assembly, electrode fitting, PiEEG setup, and debugging.                                                                                                           |
| End-to-end integration               | **Medium** | Five components (EEG → encoder → screenshot → YOLO+OCR → LLM) all need to communicate in a loop. Integration bugs will eat time.                                                             |
| Context engineering for LLM          | **Medium** | Prompt design matters. The LLM needs: current state, action history, task goal, brain signal, and available actions. Getting this right takes iteration.                                     |
| OCR quality on target app            | **Medium** | Test early. Run OCR on screenshots of the target app. If it uses custom fonts, icon-only buttons, or heavy styling, OCR will struggle. We may need hardcoded label mappings for known icons. |

---

## Brain Signal Types: TODO: Pick One

| Signal                                             | Feasibility (8ch dry EEG)                    | Usefulness to agent                          | Notes                                                     |
| -------------------------------------------------- | -------------------------------------------- | -------------------------------------------- | --------------------------------------------------------- |
| **SSVEP** (steady-state visually evoked potential) | **High**: most reliable with budget hardware | Medium: requires flickering UI targets       | Best bet for actually working. Use FFT peak detection.    |
| **P300** (event-related potential)                 | Moderate: needs ~50+ averaged trials         | High: maps to "yes this one" selection       | Slow, but conceptually clean for confirming actions.      |
| **ERN** (error-related negativity)                 | Low: subtle, hard in single trials           | High: detects "that's wrong" responses       | Ideal conceptually, but may not work with 8 dry channels. |
| **Motor imagery**                                  | Low: needs per-user calibration              | Low: doesn't map naturally to confirm/reject | Not recommended for this use case.                        |
| Alpha power (attention proxy)                      | Moderate: simple spectral analysis           | Medium: "attending" vs "not attending"       | Could work as a coarse engagement signal.                 |

**Recommendation:** Start with **SSVEP** (most likely to produce clean results) or **alpha power** (simplest to implement). Use existing BCI libraries like MNE-Python for signal processing.

---

## 1-Month Sprint Plan: Parallel Streams 

The EEG hardware and the YOLO+OCR+LLM pipeline are independent. Develop them in parallel and integrate in the final week.

#### Stream A: EEG Hardware (Weeks 1-3)

1. **Week 1-2:** 3D print and assemble the Ultracortex Mark III frame + PiEEG. Get clean data flowing from PiEEG to a laptop.
2. **Week 3:** Pick SSVEP or alpha power. Implement the classifier using MNE-Python. Validate that the encoder produces a usable discrete label.

#### Stream B: YOLO + OCR + LLM Agent (Weeks 1-3)

1. **Week 1:** Choose the target app. Record 10-15 minutes of usage. Run yolodex pipeline to train YOLO on the app's UI elements.
2. **Week 2:** Add OCR layer on top of YOLO detections. Build the structured text output format. Test OCR accuracy on the target app.
3. **Week 3:** Wire up the LLM agent loop. Use a **keyboard mock** for brain signals during development (press Y = confirm, N = reject). Demonstrate the agent completing a constrained task.

#### Integration (Week 4)

1. Swap keyboard mock for real EEG encoder output.
2. End-to-end demo: user wears EEG → agent proposes action → brain signal confirms/vetoes → action executes.
3. Record demo video. Collect performance metrics (task completion rate, latency per step, veto accuracy).

### Key tip

Use the keyboard mock for brain signals throughout Stream B development. This **decouples the two workstreams** so hardware delays don't block software progress.

---

## Feasibility Verdict (V2)

| Aspect                                                  | Verdict                                                  |
| ------------------------------------------------------- | -------------------------------------------------------- |
| Core thesis (brain signals augment computer use agents) | Sound                                                    |
| Hardware choice (Mark III + PiEEG)                      | Reasonable for budget PoC                                |
| YOLO + OCR replacing VLM for state extraction           | **Valid architecture**: a proven pattern, cost-effective |
| Brain signals as runtime gate / soft reward             | Clean framing, achievable with simple classifier         |
| Constrained agent (one app, 5-10 actions)               | Feasible scope for 1-month demo                          |
| Full general desktop agent                              | Future work; not feasible in 1 month                     |
| Context engineering (structured text → LLM)             | Solvable with iteration on prompt design                 |

**Overall: feasible as a proof-of-concept demo within 1 month, if scope stays constrained to one app and brain signal type.**

The biggest remaining risks are EEG hardware build time and OCR accuracy on the target app. Test both early. If hardware slips, the YOLO+OCR+LLM agent still works as a standalone demo (without brain signals) and the brain integration becomes a follow-up.

---

## Sources

- Yolodex pipeline: see `yolodex/` directory for the full video → dataset → YOLO training pipeline
- MNE-Python (EEG processing): https://mne.tools/
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- PyAutoGUI (action execution): https://pyautogui.readthedocs.io/
- EasyOCR: https://github.com/JaidedAI/EasyOCR
