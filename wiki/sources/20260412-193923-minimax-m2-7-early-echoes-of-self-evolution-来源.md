---
title: "MiniMax M2.7: Early Echoes of Self-Evolution"
tldr: "MiniMax releases M2.7, their first model that participated in its own training loop - builds agent harnesses, runs RL experiments, and iterates on its own scaffold, achieving near-SOTA on SWE-Pro (56.22%) and strong office/agent benchmarks"
date_created: 2026-04-12
date_modified: 2026-04-12
type: source
tags: [llm, self-evolution, agent, benchmarks, reinforcement-learning, multi-agent]
source_type: article
source_file: "[[20260412-193717-minimax-m2-7-early-echoes-of-self-evolution-文章]]"
original_url: "https://www.minimax.io/news/minimax-m27-en"
explored: false
confidence: medium
---

# MiniMax M2.7: Early Echoes of Self-Evolution

## Summary

[[20260412-194210-minimax-实体|MiniMax]] released **M2.7**, positioned as the first model to deeply participate in its own evolution. The key narrative is a shift from human-driven model iteration to model-assisted self-improvement loops.

### Self-Evolution Workflow

M2.7 was used internally to build and iterate on research agent harnesses that:
- Assist RL researchers with literature review, experiment tracking, data pipelines, and experiment launches
- Monitor experiments, trigger debugging, metric analysis, code fixes, merge requests, and smoke tests
- Handle 30-50% of the researcher workflow autonomously

A notable demonstration: M2.7 ran an autonomous loop of "analyze failure trajectories -> plan changes -> modify scaffold code -> run evaluations -> compare results -> decide to keep or revert" for over 100 rounds, achieving a **30% performance improvement** on internal evaluation sets.

On **MLE Bench Lite** (22 ML competitions), M2.7 used a self-feedback and self-optimization harness over 24-hour runs, achieving a 66.6% average medal rate (best run: 9 gold, 5 silver, 1 bronze). This places it behind Opus-4.6 (75.7%) and GPT-5.4 (71.2%), tying with Gemini-3.1 (66.6%).

### Software Engineering

- **SWE-Pro**: 56.22% (matching GPT-5.3-Codex)
- **VIBE-Pro**: 55.6% (end-to-end project delivery, near Opus 4.6)
- **Terminal Bench 2**: 57.0% (complex engineering system comprehension)
- **SWE Multilingual**: 76.5
- **Multi SWE Bench**: 52.7

### Agent Capabilities

- Native **Agent Teams** (multi-agent collaboration) - role boundaries, adversarial reasoning, protocol adherence
- 97% skill adherence rate across 40+ complex skills (each >2,000 tokens)
- **MM Claw** accuracy: 62.7% (close to Sonnet 4.6)
- **Toolathon**: 46.3%
- **GDPval-AA**: ELO 1495 (highest among open-source models)

### Professional Work & Entertainment

- Financial analysis capabilities: can read annual reports, build revenue models, produce PPT/Word deliverables
- Character consistency and emotional intelligence for interactive entertainment
- **OpenRoom**: open-source agent-based interactive entertainment system

## Key Takeaways

- The "model improving its own training infrastructure" narrative is significant - this is a concrete step toward recursive self-improvement
- The 30-50% automation of researcher workflow is a practical claim about current agent capabilities
- Benchmark positioning is strong but not leading: consistently behind Opus 4.6 and GPT-5.4
- The [[20260412-194111-agent-teams-概念|Agent Teams]] concept of native multi-agent capabilities (not just prompting) is worth tracking
- MLE Bench Lite results demonstrate autonomous ML experimentation is viable at near-frontier quality

## Concepts & Entities Mentioned

- [[20260412-194111-self-evolving-ai-概念|Self-Evolving AI]] - core thesis of the release
- [[20260412-194111-agent-teams-概念|Agent Teams]] - multi-agent collaboration paradigm
- [[20260412-194210-minimax-实体|MiniMax]] - the company
- [[20260412-194030-reinforcement-learning-概念|Reinforcement Learning]] - context for self-evolution workflow
- [[SWE-Pro]] - software engineering benchmark
- [[MLE Bench]] - ML engineering benchmark
- [[20260412-194030-agentic-ai-概念|Agentic AI]] - broader paradigm
- [[20260412-194030-sparse-attention-概念|Sparse Attention]] - not directly mentioned but related to inference optimization

## Counter-arguments

- "Self-evolution" framing is marketing-heavy - the model assists in its training loop but doesn't truly self-modify; researchers still set goals, evaluate outcomes, and make key decisions
- The 30% scaffold improvement claim is on internal evaluation sets - no external reproducibility
- MLE Bench Lite results (66.6%) are behind frontier models, suggesting self-evolution hasn't yet closed the gap
- The 97% skill adherence rate is tested on their own proprietary skill set - hard to compare externally
- "30-50% of researcher workflow" is vague - what constitutes the remaining 50-70%?

## Data gaps

- No details on M2.7's training data, model size, or architecture
- Self-evolution loop details are high-level - unclear what specifically the model changes vs what humans control
- No comparison of self-evolved harness performance vs human-designed harness
- OpenRoom entertainment claims lack evaluation metrics
- Missing cost/latency data for API users
