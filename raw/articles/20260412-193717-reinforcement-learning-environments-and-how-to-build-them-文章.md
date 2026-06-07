---
title: "Reinforcement Learning environments and how to build them"
source: "https://unsloth.ai/blog/rl-environments"
original_url: "https://unsloth.ai/blog/rl-environments"
author: "Unsloth, NVIDIA"
published: 2026-03-12
created: 2026-04-12
fetched: 2026-04-12
source_tool: obsidian-web-clipper
description: "Guide to building RL environments for LLM training using Unsloth and NVIDIA NeMo Gym, covering GRPO, RLVR, and environment design"
tags:
  - "reinforcement-learning"
  - "rl-environments"
  - "grpo"
  - "unsloth"
  - "nvidia"
---
[![unsloth sticker logo](https://unsloth.ai/cgi/image/unsloth_green_sticker_cME6ryC59BlZg-VtqGN4p.png?width=128&quality=80&format=auto)](https://unsloth.ai/)<iframe src="https://ghbtns.com/github-btn.html?user=unslothai&amp;repo=unsloth&amp;type=star&amp;count=true&amp;size=large" frameborder="0" width="170" height="30" title="GitHub"></iframe>

![6 cute pastel coloured sloths staring at their computer screens happy](https://unsloth.ai/cgi/image/green_sloth_1aakUlHhEsMK8d8eaBRQ3.png?width=2048&quality=80&format=auto)

What are RL environments and how to build them

**Authors:** Daniel, Michael, and from NVIDIA: Shashank Verma, Sylendran Arunagiri, Chris Wing, Brian Yu

[Reinforcement learning (RL)](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) has shaped AI for decades, from early control systems to game-playing agents and, more recently, large language models that learn through interaction. At its core, RL works by teaching a model to learn, respond, and receive feedback, improving the model over the course of time.

![](https://unsloth.ai/cgi/image/Figure2_krKcCuK8VRU_nQ8EbQD03.png?width=2048&quality=80&format=auto)

However, as AI becomes [agentic](https://www.nvidia.com/en-us/glossary/ai-agents/), capable of multi-step reasoning, tool use, and decision-making, we are entering the ["Era of Experience"](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf), where progress is driven by systems that learn from their own experience rather than just static data. RL must evolve from optimizing single responses to shaping behaviors across entire trajectories. In this context, learning happens through interaction with "environments" that define permissible actions, state changes, and the definition of success.

An RL workflow unifies a policy model, a training algorithm, and an environment, along with a method to verify agent responses. This interaction loop enables agents to plan, adapt, and recover from failure.

This blog explores how RL is evolving for agentic AI, why environments are central to this shift, and how open tools like [Unsloth](https://unsloth.ai/), [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL), [NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym), and [NVIDIA NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) help developers build these RL workflows efficiently.

## Comparing SFT and RL

Before building an environment, it is critical to understand when RL is the right tool.

Supervised Fine-Tuning (SFT) fits best when you can provide clear target behaviors through demonstrations, or instruction-response pairs. It is great for teaching format and style. However, SFT has limitations.

- **Imitation over adaptivity.** When the dataset is small, models learn to mimic the answer rather than learn the process to get there.
- **Brittleness.** SFT models often struggle when scenarios fall outside their training distribution, so the dataset needs to be diverse and large.

Reinforcement Learning (RL) becomes the better choice as complexity grows. Instead of telling the model "say exactly this," you provide a goal and a way to verify it. This allows the model to explore reasoning paths, making it resilient to edge cases. This tends to work well for tasks like math, code, and tool calling, among other things that have a clear path to answer verification.

In practice, SFT and RL are not mutually exclusive, and a hybrid strategy is often employed.

1. **SFT for warm-starting RL.** Use a high-quality set of demonstrations to teach the chat template, tool-calling format, and general readability. This helps RL avoid wasting time trying to learn the format of your dataset.
2. **RL for scaling.** Transition to RL to allow the model to explore and self-correct. This "post-training" refinement is where reasoning and robustness are truly forged.

For example, the [NVIDIA Nemotron 3](https://research.nvidia.com/labs/nemotron/Nemotron-3/) family of models utilizes SFT as a substantial first stage to ground the model before moving into RL refinement. The ultimate choice depends on your compute budget, data availability, and the level of generalization your agent requires. The industry is generally shifting toward allocating more compute during RL stages, especially as RL environments become more sophisticated and accessible.

## From Algorithms to Environments and the Rise of RLVR

Traditionally, RL methods like [PPO (Proximal Policy Optimization)](https://arxiv.org/abs/1707.06347v2) were the standard. However, their resource intensiveness, which requires multiple complex and compute-intensive models like the reward and critic models, has driven a shift toward more scalable algorithms.

Modern workflows are increasingly adopting more efficient methods like DPO and GRPO to handle different aspects of model improvement.

**Direct Preference Optimization (DPO)** sidesteps the RL loop entirely, treating alignment as a classification problem on static preference data.

- **Reward type.** Pairwise. It relies on labeled preferences ("Response A > Response B").
- **Efficiency.** Computationally light and stable, making it ideal for alignment tasks such as safety, tone, and style.

However, DPO lacks explicit reward optimization or exploration. It learns from fixed preference pairs, which prevents it from discovering new strategies or optimizing long-horizon outcomes. Furthermore, because the DPO algorithm models relative output preference rather than trajectory reward, it is less effective for agentic workflows that require multi-step reasoning and tool use.

To address these limitations in agentic domains, developers are turning to algorithms that leverage verifiable rewards.

One such algorithm is **[Group Relative Policy Optimization (GRPO)](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide#from-rlhf-ppo-to-grpo-and-rlvr)**, which is an optimized version of PPO. In this setup, heavy critic models are replaced by generating groups of outputs and scoring them against a deterministic verifier.

- **Reward type.** Typically binary (`0` or `1`), but it also supports continuous values (`-∞` to `+∞`). While it thrives when an environment can programmatically say "Yes" or "No," for example by checking whether a unit test passes, it also supports complex rewards where scores may exceed `1` to provide more granular feedback.
- **Efficiency.** Eliminating the value model and the reward model from PPO significantly reduces memory overhead and is a key factor in scaling reasoning capabilities.

This broader shift toward verifiable correctness is distinct from any single algorithm. While verification can drive improvements even in supervised settings, such as rejection sampling, it is central to the paradigm of **Reinforcement Learning from Verifiable Rewards (RLVR)**. By replacing subjective scoring with explicit checks, such as whether the agent produced the correct answer or called the right tools, RLVR moves the center of gravity from the optimizer to the environment. Algorithms like GRPO simply provide an efficient mechanism to optimize against these environmental signals.

In RLVR, the environment becomes the contract between learning and behavior.

Let’s now define more concretely what we mean by an environment.

## What Is an Environment?

The environment is everything outside the absolute control of the agent. An environment is defined by the task the agent must accomplish, the actions the agent can take, and the state of the world the agent observes and acts upon. The environment also determines how the agent’s performance is evaluated, including what constitutes success and how reward is assigned.

Before we move further, it’s important to formally introduce key terminology.

**Rollout.** The process of executing a policy in an environment to generate experience. It emphasizes the act of collecting data by stepping through the environment, taking actions, and recording what happens.

**Trajectory.** The resulting sequence of states, actions, and rewards produced by a rollout. It emphasizes the data itself, or the ordered record of what happened. In practice, most codebases and papers treat the terms as synonymous, since a rollout produces exactly one trajectory, and when people say "trajectory," they usually imply it came from rolling out a policy.

## Challenges of Building and Scaling Environments

- **Decoupling environments from training.** Many RL workflows tightly couple environment logic with the training pipeline, making it difficult to integrate complex agent loops, iterate on environment design, and run controlled ablations.
- **Representing agentic trajectories consistently.** The community widely uses Chat Completions today, but it was designed for stateless, single-turn interactions. Yet agentic rollouts include interleaved reasoning, tool calls, and text across multiple turns. Without a schema that natively represents this, you will need to custom-parse and serialize model outputs for every environment.
- **Resource management.** Environments often depend on external resources such as sandboxed execution, databases, APIs, and more. Each rollout needs isolated instances, and those instances must be reliably initialized and cleaned up.
- **Scalability.** Training may require thousands of parallel rollouts. Environment instances must scale accordingly, with distribution, load balancing, and fault tolerance.

## NVIDIA NeMo Gym

[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) is an open-source library for building and scaling RL environments, battle-tested through the development of the Nemotron 3 model family.

NeMo Gym is designed to address these challenges by providing a clean decoupling of rollout collection from training, standardizing trajectories using the OpenAI Responses API, and providing infrastructure to manage resource lifecycles that scale to thousands of parallel environments.

In NeMo Gym, tasks define what the agent must accomplish. Resources provide the external state the agent interacts with, for example tools, databases, sandboxed execution, as well as the verification logic that scores performance. The Model Interface handles generation, producing the model’s actions at each turn, such as text, tool calls, or code. The Agent orchestrates each rollout by calling the model to generate actions, updating the environment state via resource servers, and collecting the final reward.

![](https://unsloth.ai/cgi/image/Figure_1_gOriy5UAqPJ-cmKMadLYp.png?width=3840&quality=80&format=auto)

**Figure 1:** The architecture of NeMo Gym, which works alongside an RL training framework, illustrating the decoupling of environment rollout orchestration from model training and generation.

NVIDIA NeMo Gym integrates with RL training libraries such as NeMo RL, Unsloth, Hugging Face TRL, and others that implement training algorithms (for example, GRPO) to update the model. NeMo Gym collects rollout trajectories and rewards from the environment and passes them to the training framework, which manages policy updates and serves the updated model for the next round of rollouts.

## RLVR User Journey: From Benchmarking to Training

Before writing a single line of code, it is essential to understand the two-phase journey of an RLVR practitioner.

**Figure 2:** In the RLVR workflow, environment preparation precedes and shapes model training.

![](https://unsloth.ai/cgi/image/Figure2_krKcCuK8VRU_nQ8EbQD03.png?width=3840&quality=80&format=auto)

## Phase 1: Environment Preparation

1. **Benchmarking:** Evaluate your base model to identify specific capability gaps (for example, it fails at multi-step math or hallucinates tool arguments).
2. **Defining capabilities:** Map these failures to target capabilities.
3. **Environment development:** Either adapt an existing environment or build a new one.
4. **Task generation:** Curate data to create diverse task sets that exercise the environment. This often involves Synthetic Data Generation (SDG).
5. **Reward profiling:** "Sanity check" the environment by running rollouts across different models (including large frontier models) to ensure that environment output aligns the targeted capability with actual capabilities.

## Phase 2: Model Training

1. **Optimization:** Train the model using an algorithm like GRPO, which uses the environment’s verifiable signals to update weights.
2. **Validation:** Verify that performance on the specific environment improves and, more importantly, that this translates to improvements on broader downstream benchmarks.

The key insight is that environment preparation is how you define what "better" means. The training phase simply optimizes for the signal you’ve built.

For the purposes of this blog post, we will assume that you have a good understanding of the capability or benchmark you want to see the model improve on. The following section covers, specifically, concepts around Phase 1.3 above, that is, building an environment.

## Building an RL Environment for Model Training

NeMo Gym, an open-source library within the [NVIDIA NeMo](https://github.com/NVIDIA-NeMo/) framework, defines and orchestrates RL environments and generates scalable, verifiable rollout data, while Unsloth consumes these rollouts for efficient RL training.

Within the NeMo Gym ecosystem, building an environment relies on three foundational pillars, culminating in model training executed via an integrated RL framework.

## 1\. Task Preparation

Agents need to be exposed to a diverse set of scenarios in order to specialize and improve on a given task. For instance, in the [Workplace Assistant environment](https://docs.nvidia.com/nemo/gym/latest/training-tutorials/nemo-rl-grpo/about-workplace-assistant.html), task data consists of natural language business requests that require the agent to autonomously navigate simulated databases and tools over multiple steps. A simple single-step example of a user query and expected response would be:

**User query:**

> "Send an email to [john.smith@atlas.com](mailto:john.smith@atlas.com) with the subject 'Team Meeting' and body 'Let's meet tomorrow at 2pm to discuss the project.'"

**Expected tool call:**

```
email_send_email(
    recipient="john.smith@atlas.com",
    subject="Team Meeting",
    body="Let's meet tomorrow at 2pm to discuss the project."
)
```

When short on task-specific data, developers can turn to synthetic data generation (SDG) using tools such as [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) to programmatically create task queries, and potentially corresponding ground truths. To train effectively, you need thousands of diverse prompts that exercise the environment’s tools. For example, if you are building a coding environment, you might use an LLM to generate 5,000 unique Python word problems, while a deterministic script generates the unit tests (the ground truth) used to verify the answers.

Understanding the task is the first step in designing the environment itself.

## 2\. Environment Design

Referring back to the left half of Figure 1, the environment design consists of three primary components:

- **The Agent Server:** The central component of environment design is the agent itself. The agent orchestrates all interaction logic, such as calling the model and using tools. It acts as the scaffolding that ties everything together, managing the conversation loop (send to model, execute tool calls, repeat).
- **The Resources Server:** This component hosts the tools, maintains session state, and computes the reward.
- **The Model Interface:** This provides a standardized interface to communicate with the generation backend.

## 2.1 Agent Server

Take a look at an example Agent Server pseudocode below. It sends the conversation to the model, gets back a response, and, if the model makes any tool calls, it routes the tool calls to the resources server and feeds the results back to the model. This repeats until the model replies with a plain text message (no tool calls), hits the token limit, or exceeds `max_steps`.

```
# Agent Server pseudocode (based on SimpleAgent)

async def run(task_data):
    # 1. Initialize episode
    resource_server.seed_session(task_data)

    # 2. Run the agent loop
    response = self.responses(task_data.prompt, task_data.tools)

    # 3. Grade the result
    reward = resource_server.verify(response, task_data.ground_truth)
    return response, reward

async def responses(prompt, tools):
    conversation = prompt
    step = 0

    while step < max_steps:
        model_output = model_server.responses(conversation, tools)
        conversation.append(model_output)

        if model_output is text:
            break  # model is done, no more tool calls

        for tool_call in model_output.function_calls:
            result = resource_server.post(
                f"/{tool_call.name}",
                tool_call.arguments,
            )
            conversation.append(result)

        step += 1

    return conversation
```

Importantly, you can use an [existing agent in NeMo Gym](https://github.com/NVIDIA-NeMo/Gym/tree/main/responses_api_agents/simple_agent), bring your own, or create a completely new one. As such, it’s possible this loop looks very different depending on your setup. The [MiniSWEAgent](https://github.com/NVIDIA-NeMo/Gym/tree/main/responses_api_agents/mini_swe_agent), for example, delegates the run logic to an external harness running in Docker containers and then converts the output back into the NeMo Gym format.

Existing agents may also come with predefined tools, allowing you to leverage them directly. You can then seamlessly use the resources server to supplement the agent with any additional external tools it may need.

## 2.2 The Resources Server

The Resources Server is the "world" the agent interacts with. In NeMo Gym, this is implemented as a lightweight FastAPI application. It exposes tools as HTTP endpoints (for example, `POST /search_database`) that the model can call via standard OpenAI-compatible tool schemas, as well as reward calculation logic.

Crucially, these servers handle session management. Because an agentic rollout involves multiple steps, the environment must "remember" what happened in previous steps. NeMo Gym uses a `session_id` to maintain isolated state for every parallel rollout.

```
# Conceptual Resources Server Structure

class MyResourceServer(SimpleResourcesServer):
    async def seed_session(self, session_id, initial_data):
        # Initialize the "sandbox" for this specific rollout
        self.state[session_id] = initialize_environment(initial_data)

    async def my_custom_tool(self, session_id, tool_args):
        # Model calls this during the rollout
        result = execute_action(self.state[session_id], tool_args)
        return result
```

## 2.3 Verification Logic

The verifier is one of the most critical parts of environment design. It is often a deterministic function that evaluates the final state of a rollout and returns a reward signal.

Two common ways to design these rewards are:

- **Trajectory matching:** Comparing the agent’s specific tool calls and arguments against a "golden path." This is easier to implement but can be brittle if there are multiple correct ways to solve a problem.
- **State matching:** Checking the final outcome (for example, if the end state of the database matches the ground truth) regardless of how the agent got there. This is more robust and is the approach used for complex environments like the [Workplace Assistant](https://docs.nvidia.com/nemo/gym/latest/training-tutorials/nemo-rl-grpo/about-workplace-assistant.html).

Other major ways to design verification logic include sandboxed execution (running generated code or artifacts against unit tests), using an LLM-as-a-judge (for semantic or open-ended evaluation), and training reward models (to capture human preferences), among others.

```
# Conceptual Verification Logic

async def verify(self, session_id, agent_response, ground_truth):
    # 1. Extract what the agent actually did
    actual_outcome = self.state[session_id].get_final_state()

    # 2. Compare against the "Golden" result
    if actual_outcome == ground_truth:
        return reward(1.0)  # Success!

    return reward(0.0)  # Failure
```

Some best practices for designing verification logic include:

- **Prefer binary rewards:** While it might seem intuitive to award partial credit for intermediate steps, strict binary signals (success/failure) typically yield the most stable and effective optimization targets for algorithms like GRPO.
- **Profile your reward signals:** Before committing to a large-scale training run, evaluate your environment against multiple models of varying capabilities (for example, a small base model versus a large frontier model). If the frontier model cannot consistently outscore the base model, your verifier logic or task definitions likely require recalibration.

## 3\. Model Training

With environments in place, agent training proceeds by generating rollouts through repeated interaction between the policy model(s) and the environment. NeMo Gym orchestrates this process by running environments at scale, managing session state, and producing structured rollout trajectories annotated with rewards from the verification logic.

These rollouts are then consumed by an RL training framework such as Unsloth, NeMo RL, or HuggingFace TRL, which applies an optimization algorithm (for example, GRPO or PPO-style methods) to update model weights. Check out tutorials for [GRPO runs with NeMo RL and NeMo Gym](https://docs.nvidia.com/nemo/gym/latest/training-tutorials/nemo-rl-grpo/index.html) and [RL training with Unsloth and the NeMo Gym Sudoku environment](https://unsloth.ai/docs/models/tutorials/nemotron-3#reinforcement-learning--nemo-gym).

The training framework remains decoupled from environment implementation, allowing teams to swap optimizers, scaling strategies, or hardware backends without modifying environment logic.

Training follows an iterative loop: generate rollouts, verify outcomes, update the policy, and re-evaluate performance. This separation of rollout generation and optimization enables scalable, flexible RL workflows across different domains and infrastructure.

**Deep dive:** For a step-by-step technical walkthrough, including code examples for stateful and multi-step environments, refer to the [supplemental developer guide for building environments](https://docs.nvidia.com/nemo/gym/latest/environment-tutorials/index.html).

## Environment-Driven RL, and the Open Ecosystem

Environment-driven RL workflows are increasingly shaping how agentic systems are trained across research and industry. By separating environment definition, rollout generation, and optimization, teams can iterate faster and scale reinforcement learning without tightly coupling reward logic to a single training framework.

This pattern has already been applied in real-world systems. For example, the [NVIDIA Nemotron 3](https://research.nvidia.com/labs/nemotron/Nemotron-3/) model family was predominantly refined using structured RL across interactive environments, where verification logic prioritized correct trajectories and tool usage over single-step responses. The same environment abstractions used in that work are now available as open libraries and integrate with multiple RL training frameworks.

RL environments are also being developed for applied domains. For example, Edison Scientific integrated NeMo Gym with their [Aviary gym](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/aviary) to train scientific agents that explore hypotheses, run simulations, and receive deterministic feedback from domain-specific environments. See also NVIDIA’s post on [how to train scientific agents with reinforcement learning](https://developer.nvidia.com/blog/how-to-train-scientific-agents-with-reinforcement-learning/).

Today, interactive environments built with NeMo Gym generate verifiable rollout data that can be consumed by libraries such as Unsloth, HuggingFace TRL, NeMo RL, and other PyTorch-native stacks. This interoperability allows practitioners to choose optimizers, memory strategies, and hardware backends independently of environment design, supporting scalable agentic AI from research through production.

## Conclusion & Resources

In the era of agentic AI, the environment defines the contract for intelligence. Here’s how you can get started today:

- [Unsloth + NeMo Gym](https://unsloth.ai/docs/models/tutorials/nemotron-3#reinforcement-learning--nemo-gym): RL notebooks for Sudoku and multi-environment training with Unsloth and NeMo Gym.
- [NeMo Gym environment tutorials](https://docs.nvidia.com/nemo/gym/latest/environment-tutorials/index.html) and [training tutorials](https://docs.nvidia.com/nemo/gym/latest/training-tutorials/index.html) help you build custom RL environments and use them with your preferred RL training framework.
- [NeMo Gym GitHub](https://github.com/NVIDIA-NeMo/Gym): The core library for building and orchestrating verifiable RL environments.

💕 Thank you!

A huge thank you to NVIDIA for authoring this educational blog along with us. Also thank you for reading and using Unsloth - we appreciate it. 🙏  
  
As always, be sure to join our [Reddit page](https://www.reddit.com/r/unsloth/) and [Discord](https://discord.gg/unsloth) server for help or just to show your support! You can also follow us on [Twitter](https://twitter.com/unslothai) and our newsletter on: [Substack](https://unslothai.substack.com/).

Thank you for reading!

Daniel & Michael Han 🦥  
March 12, 2026

## Learn how to RL now!

[Get started for free](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)[Join Our Discord](https://discord.com/invite/unsloth)