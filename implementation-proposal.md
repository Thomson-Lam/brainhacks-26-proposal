# Implementation Proposal

## Proposal

The project will be implemented as a **MCP server + CLI frontend** for the AI agent.

- The MCP server provides context (YOLO detections, OCR results, brain signals, action execution) to the LLM agent through standardized tool interfaces.
- The CLI frontend allows the end user to see and track actions the agent undertakes in real time.
- The primary benefit of the CLI implementation is that it is **lightweight, simple, and works on all operating systems**.
- It is also **flexible** — anyone can integrate their own agents or additional implementations with the MCP server outside of the project.
- Other well-developed techniques in the AI agent ecosystem — programmatic tool calling, code execution, bash commands — can also be used by the AI agent as a further enhancement for desktop agents **instead of through the GUI**, because LLM agents are text-first and this nativeness will be more efficient and flexible.

---

## MCP Server as the Context Layer

MCP is designed for exactly this: exposing tools and context to LLM agents through a standardized protocol. YOLO detections, OCR results, brain signals, and action execution are all tools the MCP server serves. The architecture maps cleanly:

```
┌─────────────────────────────────────────────────┐
│  MCP Server                                     │
│                                                 │
│  Tools:                                         │
│  ├─ capture_and_detect() → YOLO + OCR pipeline  │
│  ├─ get_brain_signal()   → EEG encoder output   │
│  ├─ click(x, y)          → PyAutoGUI            │
│  ├─ type_text(text)      → PyAutoGUI            │
│  ├─ run_command(cmd)     → subprocess           │
│  └─ get_action_history() → state tracking       │
│                                                 │
│  Resources:                                     │
│  ├─ current screen state (structured text)      │
│  └─ task context / goal                         │
└───────────────┬─────────────────────────────────┘
                │ MCP protocol
┌───────────────▼─────────────────────────────────┐
│  MCP Client (Claude Code, custom CLI, etc.)     │
│  └─ LLM agent reasons over tools + context      │
│     └─ User sees tool calls + reasoning in CLI  │
└─────────────────────────────────────────────────┘
```

Any MCP-compatible client can connect — Claude Code, Claude Desktop, or a custom CLI. Tool call visibility comes for free: the user sees every `capture_and_detect()` call, every `click()`, every brain signal check. This satisfies the "tracking actions the agent undertakes" requirement without building custom UI.

---

## CLI Frontend

A CLI is the right trade-off for a PoC. No React app, no Electron wrapper, no web dashboard. The agent's reasoning and tool calls are inherently text — a terminal is the natural display. Cross-platform, zero dependencies beyond Python and the MCP server.

If Claude Code is used as the MCP client, the CLI frontend is already built. Only the MCP server needs to be written.

---

## Programmatic Actions Over GUI

This is the strongest design decision in the proposal.

Most computer use agent demos show an agent clicking through GUIs with pixel coordinates. But an LLM with bash/code execution can do the same tasks more reliably:

| Task | GUI approach (fragile) | Programmatic approach (reliable) |
|------|------------------------|----------------------------------|
| Open an app | Screenshot → find icon → click | `open -a Spotify` / `xdg-open` |
| Search for a file | Click Finder → navigate → scroll | `find / -name "report.pdf"` |
| Send an API request | Click through Postman UI | `curl -X POST ...` |
| Edit a document | Click into text field → type | Script/CLI tool for the app |
| Browser interaction | Screenshot → find button → click | Playwright / Selenium |
| System settings | Navigate Settings app | `defaults write ...` (macOS) |

The LLM is a text-native agent. Giving it bash, code execution, and programmatic tools plays to its strengths. GUI pixel-clicking is a **fallback** for when no programmatic interface exists — not the primary interaction mode.

This reframes the role of YOLO in the architecture. YOLO + OCR isn't the main action pathway — it's the **fallback perception layer** for when the agent encounters a GUI-only surface. The primary pathway is programmatic tool use.

---

## Concerns and Considerations

### 1. MCP is request-response, not streaming

EEG is a continuous signal. MCP tools are called on-demand. The agent has to explicitly call `get_brain_signal()` to check — it won't be interrupted mid-reasoning. This means:

- The brain signal gate is **polled, not pushed**. The agent decides when to check.
- If the agent doesn't call the tool, the brain signal is ignored.
- For a PoC this is acceptable: the agent calls `get_brain_signal()` before every action execution.

Limitation: the brain signal cannot interrupt or abort a running action. It is checked at decision points, not continuously.

### 2. If most actions are programmatic, when does YOLO matter?

If the agent can `open -a Safari` and use Playwright to interact with a web page, it never needs to look at the screen. YOLO becomes unused.

Two ways to resolve this:

1. **Choose a target app that requires GUI interaction** — a game, a desktop app with no CLI/API, a custom UI.
2. **Position YOLO as the monitoring/verification layer** — the agent acts programmatically but uses YOLO to verify the result ("did the button actually get clicked? did the page change?").

The second framing is interesting: YOLO becomes a **visual assertion tool**, not the primary action driver. The agent runs a command, then calls `capture_and_detect()` to verify the screen state changed as expected — analogous to how a human glances at the screen after clicking something.

### 3. Which MCP client?

- **Claude Code / Claude Desktop**: CLI + tool calling for free, but tied to Anthropic's client behavior and token costs.
- **Custom CLI client**: Full control, but need to handle MCP client protocol.

For a PoC, start with Claude Code or Claude Desktop as the client. Only build a custom one if limitations are hit.

### 4. The MCP server is the bulk of implementation work

The server needs to:
- Manage a YOLO model instance (load weights, run inference)
- Run OCR on cropped detections
- Interface with PiEEG / the brain signal encoder
- Execute actions (PyAutoGUI, subprocess)
- Track action history and state
- Handle tool schemas and MCP protocol boilerplate

Budget at least a week for this.

---

## Detailed MCP Server Architecture

```
┌──────────────────────────────────────────────────────┐
│ MCP Server (Python)                                  │
│                                                      │
│ Perception tools:                                    │
│ ├─ capture_and_detect()  → screenshot + YOLO + OCR   │
│ │   returns: list of {type, label, position, conf}   │
│ ├─ get_brain_signal()    → EEG encoder output        │
│ │   returns: "confirm" | "reject" | "uncertain"      │
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
└──────────────────────────────────────────────────────┘
```

The LLM agent's decision loop:

1. Call `capture_and_detect()` to read screen state
2. Reason about what to do (prefer `run_command` over `click` when possible)
3. Call `get_brain_signal()` before executing
4. Execute action
5. Call `capture_and_detect()` or `verify_screen()` to confirm result
6. Repeat

---

## Implementation Effort Estimate

| Component | Effort | Notes |
|-----------|--------|-------|
| MCP server boilerplate | Low | Python MCP SDK exists, tool registration is straightforward |
| YOLO integration in server | Low | Load model, run inference, return JSON — yolodex already has this pattern |
| OCR integration | Low | EasyOCR / Tesseract, a few lines per detection |
| Brain signal tool | Medium | Depends on PiEEG interface. Needs a running EEG data stream that the tool samples from on each call |
| Action execution tools | Low | PyAutoGUI + subprocess. Simple wrappers |
| State tracking / history | Low | Append-only list in memory |
| CLI frontend | **Free** | Use Claude Code or Claude Desktop as the MCP client |
| Testing without EEG | Easy | Mock `get_brain_signal()` to return keyboard input |

The MCP server is approximately **3-5 days of focused Python work**, excluding the brain signal integration. With the keyboard mock for brain signals, the full agent loop can be working within a week.

---

## Feasibility Summary

| Assumption | Verdict |
|------------|---------|
| MCP server for context/tools | Correct abstraction. Clean tool boundaries, standardized protocol, any client can connect. |
| CLI frontend | Right trade-off. Use an existing MCP client (Claude Code) instead of building one. |
| Programmatic actions preferred over GUI | Strongest insight. LLMs are text-native — bash/code execution is more reliable than pixel-clicking. Position YOLO as fallback/verification, not primary. |
| Cross-platform and lightweight | True, assuming dependencies (YOLO, OCR, PyAutoGUI) install cleanly. Test on target OS early. |
| Others can extend via MCP | True. Well-defined tool interface means anyone can add tools or swap the client. |

**The implementation plan is sound.** The main decision to settle is the role of YOLO — is it the primary perception layer, or a fallback/verification tool behind programmatic actions? The answer determines the demo target and app choice.

---

## Resolving the Two Core Tensions

### Tension 1: Continuous EEG stream vs request-response MCP

MCP tools are request-response, but EEG is a continuous signal. This is a **soft conflict** — it resolves cleanly because the agent never actually needs continuous access to the EEG stream. It needs a brain signal classification **at decision points**, right before committing to an action.

The continuous stream lives inside the MCP server process. The tool call just samples from it:

```
EEG hardware ──→ continuous buffer (ring buffer on server)
                        │
                        │  agent calls get_brain_signal()
                        ▼
                  take latest N-second window
                        │
                        ▼
                  encoder/classifier
                        │
                        ▼
                  return "confirm" / "reject"
```

This is standard practice — audio processing, sensor APIs, and monitoring systems all work this way. The MCP protocol boundary is request-response, but the internal data pipeline is continuous. No architectural conflict.

The only design decision is **window size**: how many seconds of EEG does the encoder need to classify?

| Signal type | Window size needed |
|-------------|--------------------|
| SSVEP | ~1-2 seconds |
| Alpha power | ~2-5 seconds |
| P300 | ~0.5-1 second per trial, but needs multiple averaged trials |

This determines the ring buffer size. The tool call either returns the latest complete window, or blocks briefly until enough data is available.

### Tension 2: GUI execution vs CLI execution — resolving overlapping concerns

The conflict is real: if the agent has both `click(x, y)` and `run_command("open -a Safari")`, which does it use? Without explicit guidance, the LLM will inconsistently pick one or the other, or worse, try both.

The resolution: **CLI is primary, GUI is fallback. No overlap.** A clear execution hierarchy:

```
Can this be done programmatically?
  ├─ YES → run_command() / browser_action() / code execution
  │         then verify_screen() to confirm result
  │
  └─ NO (GUI-only surface) → capture_and_detect() → click() / type_text()
```

To enforce this without relying on the LLM to always choose correctly, two options:

**Option A: Prompt engineering (recommended for PoC).** Include the hierarchy in tool descriptions within the MCP schema:

```
run_command:
  "Execute a shell command. PREFER this over GUI tools
   whenever the task can be done programmatically."

click:
  "Click at screen coordinates. Use ONLY when no
   programmatic alternative exists for the current task."
```

Simple and works reasonably well. For a demo with 5-10 constrained actions, behavior can be validated manually.

**Option B: Conditional tool exposure (production).** Only expose GUI tools conditionally — the agent first attempts programmatic execution, and only if it explicitly reports failure or calls a `request_gui_mode` tool does the server expose the GUI tools. Heavier to implement but eliminates ambiguity entirely.

### How the two execution models complement each other

The two models aren't competing — they cover different surfaces:

| Surface | CLI handles | GUI handles |
|---------|-------------|-------------|
| File operations | `cp`, `mv`, `mkdir` | N/A |
| App launching | `open -a X`, `xdg-open` | N/A |
| Web interaction | Playwright / Selenium | Fallback if no automation API |
| Terminal/shell tasks | Direct execution | N/A |
| Custom desktop apps (no API) | N/A | YOLO + OCR → click |
| Games | N/A | YOLO + OCR → click |
| Verification after any action | N/A | `capture_and_detect()` confirms result |

The last row is key: **GUI perception complements CLI execution.** The agent runs a command, then uses YOLO to verify the screen changed as expected. They aren't overlapping — one acts, the other observes.

### Summary of tensions

Both tensions are real but resolvable:

1. **EEG vs request-response**: soft conflict. The continuous stream lives inside the server; the MCP tool samples from it at decision points. Standard pattern.
2. **GUI vs CLI execution**: real conflict, resolved by establishing a CLI-first hierarchy with GUI as fallback/verification. Enforce through tool descriptions (PoC) or conditional tool exposure (production).
