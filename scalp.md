

Problem: computer use agents are expensive and unreliable across long context windows due to the vision + language + agentic complexity.

This project aims to explore integration of implicit brain signals as a new source of data to augment computer use agents, and to make computer use agents more reliable and safe across long multi step scenarios.

GOALS: 

- build a EEG 
- sample brain signal data from the EEG 
- using brain signals, train an ensemble system that provides more high quality data and signals for computer agents to work with 

Initial system design: 

Use Claude Code or Codex, and use the Yolodex skill to train lightweight YOLO vision transformer nano models for computer use classification. Instead of having a single big model, use specialized YOLO vision models as subagents, and use the original computer use model as a fall back or as an orchestrator instead. Contextual metadata or information is gathered and the depending on brain signals, the agent decides on the next best course of action. Instead of allowing a model trained on brain signals to make decisions, this architecture relies on the intelligent agent to make the final call, especially for breaking down long and abstract ideas into actionable and verifiable steps to ensure task completion beyond just the triggering of a single action.

Brain signals -> lightweight encoder model/classifier-> + metatdata (contextual information) -> AI agent -> trigger YOLO subagents or vision model takes over.

Additional system optimization: provide caching mechanisms beyond training knowledge:

1. user performs a slightly newer or unknown action -> click button 
2. brain signal + contextual information is recorded 
3. this is saved:
	1. when a similar brain signal is detected again, check for similarity and load context to the AI agent, or execute a similar action again 
	2. the data is stored and a data pipeline to update models live, or for training, is run. The YOLO submodel(s) become learn of the task and no storage of specific action is needed.
