---
title: "Supervised Fine-Tuning"
tldr: "Training on instruction-response pairs to teach models format and behavior - effective for style/format but limited in adaptivity compared to RL, often used as warm-start before RL training"
date_created: 2026-04-12
date_modified: 2026-04-12
type: concept
tags: [training, fine-tuning]
sources: ["[[unsloth-nvidia-2026-rl-environments]]"]
explored: false
confidence: high
---

# Supervised Fine-Tuning

**Supervised Fine-Tuning (SFT)** trains models on instruction-response pairs. Great for teaching format and style but limited: models imitate answers rather than learning the reasoning process, and struggle outside training distribution. Best used as warm-start before [[Reinforcement Learning]].

## How It Connects

- [[Reinforcement Learning]], [[DPO]], [[GRPO]]
