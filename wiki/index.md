---
title: "Wiki Index"
tldr: "Master catalog of all wiki pages with TLDRs for fast scanning"
date_created: 2026-04-12
date_modified: 2026-04-12
explored: false
confidence: medium
---

# Wiki Index

_Scan TLDRs to find relevant pages. Load full pages only when needed._

## Sources

| Page | TLDR |
|------|------|
| [[lmsys-2026-hisparse]] | LMSYS introduces HiSparse, a hierarchical memory system for sparse attention that offloads inactive KV cache to host memory, achieving 3-5x throughput gains on long-context LLM inference |
| [[lmsys-2026-hisparse-tweet]] | LMSYS Twitter thread announcing HiSparse blog with key results: 3x throughput at 256 concurrent requests, 5x on long-context, supports DeepSeek-V3.2 and GLM-5.1 |
| [[vllm-2026-llm-compressor]] | vLLM's llm-compressor tool hits 3K GitHub stars, supports NVFP4 and FP8 quantization for Gemma 4 and Qwen 3.5 models for faster inference |
| [[minimax-2026-m27]] | MiniMax releases M2.7, their first model that participated in its own training loop - builds agent harnesses, runs RL experiments, and iterates on its own scaffold, achieving near-SOTA on SWE-Pro (56.22%) |
| [[unsloth-nvidia-2026-rl-environments]] | Comprehensive guide by Unsloth and NVIDIA on building RL environments for LLM training, covering the shift from PPO to GRPO/RLVR, NeMo Gym architecture, and practical environment design patterns |

## Concepts

| Page | TLDR |
|------|------|
| [[sparse-attention]] | Attention mechanism attending to only a subset of KV caches, reducing compute but not inherently solving the memory capacity bottleneck |
| [[kv-cache]] | Stores precomputed attention keys/values during inference - memory footprint is the primary bottleneck for long-context and high-concurrency serving |
| [[reinforcement-learning]] | Training paradigm where models learn through interaction and feedback - evolving from PPO to GRPO/RLVR for LLM training |
| [[agentic-ai]] | AI systems capable of multi-step reasoning, tool use, and autonomous decision-making |
| [[quantization]] | Reducing model precision (FP16->FP8->FP4) to decrease memory and increase throughput |
| [[grpo]] | Group Relative Policy Optimization - efficient RL algorithm eliminating critic/reward models |
| [[rlvr]] | RL from Verifiable Rewards - paradigm replacing subjective rewards with deterministic correctness checks |
| [[self-evolving-ai]] | AI systems that participate in their own improvement loop (stub) |
| [[agent-teams]] | Multi-agent collaboration with role boundaries and adversarial reasoning (stub) |
| [[prefill-decode-disaggregation]] | Deployment architecture separating prefill and decode instances (stub) |
| [[dpo]] | Direct Preference Optimization - alignment as classification on preference data (stub) |
| [[ppo]] | Proximal Policy Optimization - standard but resource-intensive RL algorithm (stub) |
| [[supervised-fine-tuning]] | Training on instruction-response pairs, best as warm-start before RL (stub) |

## Entities

| Page | TLDR |
|------|------|
| [[lmsys]] | Research org behind Chatbot Arena, SGLang, and HiSparse |
| [[sglang]] | LLM serving framework by LMSYS with RadixAttention, HiCache, HiSparse |
| [[vllm]] | Popular open-source LLM serving framework with PagedAttention and llm-compressor |
| [[minimax]] | Chinese AI company behind M2 model series, pioneering model self-evolution |
| [[unsloth]] | Open-source framework for efficient LLM fine-tuning and RL training |
| [[nvidia]] | GPU manufacturer, develops NeMo ecosystem for LLM training |
| [[nemo-gym]] | NVIDIA's open-source library for building RL environments |
| [[deepseek]] | Chinese AI lab, DeepSeek-V3.2 uses DeepSeek Sparse Attention |
| [[qwen]] | Alibaba's LLM family, Qwen 3.5 supported by vLLM quantization |
| [[gemma]] | Google's open-weight LLM family, Gemma 4 supported by vLLM quantization |
| [[glm]] | Zhipu AI's LLM family, GLM-5.1 uses DSA, supported by HiSparse |

## Projects

_No projects yet._

## SOPs

_No SOPs yet._

## Syntheses

_No syntheses yet._

## Outputs

_No outputs yet. Run `/wiki-query` to ask a question._
