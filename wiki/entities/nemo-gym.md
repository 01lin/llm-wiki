---
title: "NeMo Gym"
tldr: "NVIDIA's open-source library for building and scaling RL environments for LLM training, providing decoupled rollout collection, trajectory standardization, and resource lifecycle management"
date_created: 2026-04-12
date_modified: 2026-04-12
type: entity
tags: [framework, reinforcement-learning, nvidia, open-source]
sources: ["[[unsloth-nvidia-2026-rl-environments]]"]
explored: false
confidence: high
---

# NeMo Gym

**NeMo Gym** is [[NVIDIA]]'s open-source library for building and orchestrating RL environments for LLM training. Battle-tested on the Nemotron 3 model family. Architecture: Agent Server (interaction loop) + Resources Server (tools/state/rewards) + Model Interface (generation). Uses OpenAI Responses API for trajectory standardization. Integrates with [[Unsloth]], HuggingFace TRL, and NeMo RL.

## Related

- [[NVIDIA]], [[Reinforcement Learning]], [[GRPO]], [[RLVR]], [[Unsloth]]
