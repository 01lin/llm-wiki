---
title: "GRPO"
tldr: "Group Relative Policy Optimization - efficient RL algorithm for LLMs that eliminates critic and reward models by scoring groups of outputs against deterministic verifiers"
date_created: 2026-04-12
date_modified: 2026-04-12
type: concept
tags: [reinforcement-learning, algorithm, training]
sources: ["[[unsloth-nvidia-2026-rl-environments]]"]
explored: false
confidence: medium
---

# GRPO

**Group Relative Policy Optimization (GRPO)** is an optimized version of [[PPO]] for LLM training that replaces heavy critic models with generating groups of outputs and scoring them against deterministic verifiers. Supports binary (0/1) and continuous rewards. A key enabler of the [[RLVR]] paradigm.

## Key Ideas

- Eliminates value model and reward model from PPO, significantly reducing memory overhead
- Central to scaling reasoning capabilities in modern LLMs
- Works with [[NeMo Gym]] environments and training frameworks like [[Unsloth]]

## How It Connects

- [[Reinforcement Learning]] - parent paradigm
- [[RLVR]] - paradigm it enables
- [[PPO]] - predecessor algorithm
- [[DPO]] - alternative approach (no RL loop)

## Data gaps

- Direct GRPO vs PPO comparisons on identical tasks
