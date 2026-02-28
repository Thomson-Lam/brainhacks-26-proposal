# Development process 

How should we approach this problem, and what assumptions do we need to make to build out a working pipeline?

Before you build any harness, you need empirical answers to:
- Does the LLM make bad decisions when given free rein, or does it behave reasonably with good tool descriptions?
- Where does context actually blow up — is it action history, screen state, or both?
- Does the brain signal gate introduce enough decision-point clarity that the LLM doesn't need further constraint?

You can't design a harness to fix failures you haven't observed yet. Running the PoC with a free-form agent and structured MCP tools gives you the data to answer these questions. If free-form agent and structured MCP tools gives you the data to answer these questions. If you can implement the fix server-side (smarter tool responses) before reaching for client-side control flow. The one lightweight "harness" that is appropriate at PoC stage: tool descriptions. Encoding the CLI-first/GUI-fallback hierarchy directly in the MCP tool schemas ("prefer run_command — use click only when no programmatic alternative exists") is prompt-level guidance that costs nothing to implement and is fully standards-compliant. That's as far as you need to go on the client side.

Bottom line: Keep the architecture simple. Free-form LLM + MCP tools + smart server-side context shaping. Instrument the PoC to log what context the LLM actually receives at each step, watch where it fails, and optimize get_action_history() and capture_and_detect() return values as your first response. You'll get better signal from one real run than from a harness designed against imagined failure modes.

Start with minimal assumptions and breadth (what is the most likely to work for all AI providers), then optimize our standard provider (context provider, internal harness of the MCP tools), instead of client harnesses.

1. go with the broadest and most free agentic harness, then benchmark the models
2. observe failures, and improve the system against edge cases. The system is already generalizable with the YOLO pipeline too, but make sure to test against a new case.

Process:

```
Normal agent client to MCP (note: you can test this with CLIs) -> evaluate and do context engineering on the MCP tools -> make smarter MCP tools and knowledge base on the MCP server
```

Model eval process: Consider benchmarking the models against OmniParser, or different variations of YOLO and different training processes? (ie. training YOLO with data from more than 1 app/GUI/desktop instead of training a single YOLO on a single app/desktop/UI type) 

Granularity ensures that for a specific app, it always works best. But what is the sweet spot? This is what we are trying to look for as well.
