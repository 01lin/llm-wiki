---
title: "Reinforcement Learning"
tldr: "Training paradigm where models learn through interaction and feedback rather than static demonstrations - evolving from PPO to GRPO/RLVR for LLM training, with environments becoming the central design object"
date_created: 2026-04-12
date_modified: 2026-04-12
type: concept
tags: [training, rl, grpo, rlvr, agents]
sources: ["[[unsloth-nvidia-2026-rl-environments]]", "[[minimax-2026-m27]]"]
explored: false
confidence: high
---

# Reinforcement Learning

**Reinforcement learning (RL)** in the LLM context is a training paradigm where models learn by generating outputs, receiving feedback (rewards), and optimizing their policy accordingly. It contrasts with [[Supervised Fine-Tuning]] which learns from static demonstrations.

## Key Ideas

### Algorithm Evolution
- **[[PPO]]** (Proximal Policy Optimization) - standard but resource-intensive, requires reward + critic models
- **[[DPO]]** (Direct Preference Optimization) - sidesteps RL loop, treats alignment as classification on preference data. Light but lacks exploration.
- **[[GRPO]]** (Group Relative Policy Optimization) - eliminates critic/reward models by scoring groups of outputs against verifiers. The current state of the art for efficiency.

### RLVR Paradigm
**[[RLVR]]** (Reinforcement Learning from Verifiable Rewards) shifts the center of gravity from the optimizer to the environment. Instead of subjective scoring, explicit correctness checks (unit tests pass, answer matches, tool call correct) drive learning. The environment defines "what better means."

### Practical Workflow
1. [[Supervised Fine-Tuning|SFT]] for warm-starting (teach format/style)
2. RL for scaling (exploration, self-correction, reasoning)

Industry trend: allocating more compute to RL stages as environments become more sophisticated.

### Self-Evolution
[[MiniMax]] demonstrated using M2.7 in its own training loop: the model builds agent harnesses, runs RL experiments, iterates on its own scaffold. Achieved 30% improvement over 100+ autonomous iteration rounds.

## How It Connects

- [[Agentic AI]] - RL enables agents to learn from multi-step interaction
- [[GRPO]] - current dominant algorithm for LLM RL
- [[NeMo Gym]] - NVIDIA's framework for building RL environments
- [[Unsloth]] - efficient RL training framework
- [[Self-Evolving AI]] - RL as the mechanism for recursive self-improvement

## Counter-arguments

- RL is significantly harder to debug and reproduce than SFT
- Binary reward signals may be too coarse for complex multi-step tasks
- The "environment defines intelligence" framing overstates RL's role vs pre-training
- RL gains may not generalize beyond the specific environments used for training

## Data gaps

- Concrete comparisons of GRPO vs PPO on identical tasks
- How RL training scales with environment complexity
- Cost analysis for RL at scale vs SFT
