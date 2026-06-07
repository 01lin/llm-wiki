---
title: "Supervised Fine-Tuning"
tldr: "Training on instruction-response pairs to teach models format and behavior - effective for style/format but limited in adaptivity compared to RL, often used as warm-start before RL training"
date_created: 2026-04-12
date_modified: 2026-04-12
type: concept
tags: [training, fine-tuning]
sources: ["[[20260412-193923-rl-environments-and-how-to-build-them-来源]]"]
explored: false
confidence: high
---

# Supervised Fine-Tuning

**Supervised Fine-Tuning (SFT)** trains models on instruction-response pairs. Great for teaching format and style but limited: models imitate answers rather than learning the reasoning process, and struggle outside training distribution. Best used as warm-start before [[20260412-194030-reinforcement-learning-概念|Reinforcement Learning]].

## How It Connects

- [[20260412-194030-reinforcement-learning-概念|Reinforcement Learning]], [[20260412-194126-dpo-概念|DPO]], [[20260412-194111-grpo-概念|GRPO]]
