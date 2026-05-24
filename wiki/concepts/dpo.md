---
title: "DPO"
tldr: "Direct Preference Optimization - alignment method that treats preference learning as classification on static data, bypassing the RL loop entirely"
date_created: 2026-04-12
date_modified: 2026-04-12
type: concept
tags: [alignment, training, preference-optimization]
sources: ["[[unsloth-nvidia-2026-rl-environments]]"]
explored: false
confidence: medium
---

# DPO

**Direct Preference Optimization (DPO)** sidesteps the RL loop by treating alignment as a classification problem on labeled preference pairs ("Response A > Response B"). Computationally light and stable, ideal for safety/tone/style alignment. Lacks exploration and is less effective for multi-step agentic tasks.

## How It Connects

- [[Reinforcement Learning]], [[GRPO]], [[PPO]]
