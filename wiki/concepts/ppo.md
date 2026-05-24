---
title: "PPO"
tldr: "Proximal Policy Optimization - standard RL algorithm for LLM training, resource-intensive due to requiring reward and critic models, being superseded by GRPO"
date_created: 2026-04-12
date_modified: 2026-04-12
type: concept
tags: [reinforcement-learning, algorithm, training]
sources: ["[[unsloth-nvidia-2026-rl-environments]]"]
explored: false
confidence: high
---

# PPO

**Proximal Policy Optimization (PPO)** was the standard RL algorithm for LLM training (used in RLHF). Requires separate reward and critic models, making it resource-intensive. Being superseded by [[GRPO]] which eliminates these requirements.

## How It Connects

- [[Reinforcement Learning]], [[GRPO]], [[DPO]]
