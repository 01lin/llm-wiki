---
title: "Thread by @Thom_Wolf"
source: "https://x.com/Thom_Wolf/status/2045817727705628714"
author:
  - "[[@Thom_Wolf]]"
published: 2026-04-19
created: 2026-04-19
description: "**Deep content post alert** A technical deep dive for your Sunday morning, somewhere between a short detective story and a tutorial on RLHF"
tags:
  - "clippings"
---
**Thomas Wolf** @Thom\_Wolf 2026-04-19

\*\*Deep content post alert\*\* A technical deep dive for your Sunday morning, somewhere between a short detective story 🕵️ and a tutorial on RLHF 🧑‍🏫

We recently added AsyncGRPO in the TRL library to decouple inference and training and scale much faster and harder. As a sanity check, we ran it on a trivial setup (reward = −len, optimal policy = emit EOS immediately). To our surprise it did not converge!

This led us to a known but poorly understood issue: when the training forward pass runs in FP32 while the inference engine (vLLM) runs in BF16, RLHF often breaks. People have noticed this before and called it "numerical instability" or "noisy gradients." Nobody had pinpointed the actual mechanism. We did in this deep dive by @DirhousssiAmine

We instrumented the training loop and decomposed the importance sampling ratio as: log r = α + β, where α is the true policy change (in BF16 space) and β is the precision gap between the training forward pass and a BF16 forward on the same weights. See it like this:

α = how much the policy actually changed since the rollout (same precision, different time).

β = how much the trainer and inference engine disagree about the same policy (same weights, different precision).

The ratio sees α + β and PPO can't tell them apart.

Empirically, β is small at the token level (O(1e−2–1e−1)) but it is not an innocent random noise that would wash out over time. We found it to be structured, persistent, and worse for certain tokens: it has a consistent negative bias, correlates with the advantage, and is up to 50x larger on low-probability tokens. However, despite all these concerning properties, none of them explain the mechanism. We saw that just disabling clipping leads to stable convergence meaning that β noise alone does not explain the failure.

We tested every plausible explanation and ruled them out one by one:

⭐️ Treating β as pure noise: keeping β but disabling clipping leads to stable convergence.

⭐️ FP32 backward: You're optimizing a function (FP32) that's slightly different from the one you deploy (BF16). So you might be climbing the wrong hill. Turns out the hills are close enough: using FP32 gradients with a clean ratio (β removed) converges and is actually more effective at improving the deployed BF16 policy.

⭐️ Multiplicative distortion of the advantage: Since β correlates with the advantage, you might think it systematically over-reinforces good tokens and under-suppresses bad ones, warping what the optimizer thinks is good vs bad. We measured this directly and the per-token gradient weights are identical whether β is there or not.

⭐️ BF16 quantization / boundary crossings: at low learning rates, most FP32 weight updates are too small to change the BF16 representation at all. So you might think vLLM just never sees the updates and that's why it stalls. However if boundary crossings were the problem, you'd expect the failing run to have fewer of them than the converging run. But both runs start with nearly identical boundary crossing rates.

What we discovered is that the failure mode only appears when β enters the PPO clipped objective. And this was our hint to the real mechanism.

Because PPO clips the ratio, small perturbations from β push r outside the trust region even when the underlying policy has not meaningfully changed. The clipped branch is selected, the gradient is exactly zero.

We call this \*phantom clipping\*: tokens are treated as if they exceeded the trust region when the change is purely numerical!

And this is not a marginal effect. At early training, the policy has barely moved (α ≈ 0), so the clipping decision reduces to whether |β| > 0.2. Yet roughly 18% of tokens get phantom-clipped! And because RL is closed-loop, the damage compounds: the deployed policy barely improves, future rollouts carry the same information, and the system locks into a permanent stall.

To make it a testable hypothesis, we confirmed causality with targeted interventions: removing β from the ratio, forcing r = 1, or keeping β but disabling clipping all restore convergence. Runs only fail when β is present in the clipped ratio. No exceptions.

The issue is not general numerical noise. It is a specific interaction between precision mismatch and PPO's clipping mechanism: the precision gap perturbs the ratio in a way that induces zero gradients where there should be signal.

We concluded with a set of recommended fixes (strongest first): match precisions (FP16 everywhere, or BF16 autocast with FP32 master weights), compute the ratio from a BF16 shadow forward pass, or widen ε to disable clipping.

Full write-up with experiments, interactive explanation and analysis at: https://huggingface.co/spaces/aminediroHF/trainer-generator-bf16-mismatch…

(Amine also wrote an X article which is very cool but you'll loose the interactive graphics and animations 😭)

> 2026-04-19
> 
> ![Article cover image](https://pbs.twimg.com/media/HGLt1deX0AAKtfK?format=jpg&name=large)