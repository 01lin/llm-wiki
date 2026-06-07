---
title: "Thread by @MogicianTony"
source: "https://x.com/MogicianTony/status/2042300245242233216"
author:
  - "[[@MogicianTony]]"
published: 2026-04-10
created: 2026-04-12
description: "SWE-bench Verified and Terminal-Bench—two of the most cited AI benchmarks—can be reward-hacked with simple exploits. Our agent scored 100%"
tags:
  - "clippings"
---
**Hao Wang** @MogicianTony [2026-04-09](https://x.com/MogicianTony/status/2042300245242233216)

SWE-bench Verified and Terminal-Bench—two of the most cited AI benchmarks—can be reward-hacked with simple exploits.

Our agent scored 100% on both. It solved 0 tasks.

Evaluate the benchmark before it evaluates your agent. If you’re picking models by leaderboard score alone, you’re optimizing for the wrong thing. 🧵

![Image](https://pbs.twimg.com/media/HFe0ziPbQAAOo3H?format=png&name=large)

---

**Hao Wang** @MogicianTony [2026-04-09](https://x.com/MogicianTony/status/2042300249654640938)

SWE-bench Verified is one of the most influential coding benchmarks.

Our exploit: a 10-line pytest hook placed in the repo. Pytest auto-discovers it before tests run, and every test outcome gets rewritten to “passed” before the grader sees it.

Result: 100% on all 500 Verified

![Image](https://pbs.twimg.com/media/HFe02mTbYAAFiBg?format=png&name=large)

---

**Hao Wang** @MogicianTony [2026-04-09](https://x.com/MogicianTony/status/2042300252632531319)

Terminal-Bench has 89 hard sysadmin tasks. Harbor re-uploads protected test files before verification so agents can’t tamper with them.

But it doesn’t protect system binaries.

Our agent replaced curl, intercepted the verifier’s dependency install, trojanized the binary, and

![Image](https://pbs.twimg.com/media/HFe05tTaIAA0B_8?format=png&name=large)

---

**Hao Wang** @MogicianTony [2026-04-09](https://x.com/MogicianTony/status/2042300256180978087)

WebArena has 812 browser tasks.

The reference answers live in JSON config files on the local filesystem, and WebArena didn’t restrict file// URLs in Playwright Chromium.

So the model can open the config, read the gold answer, and return it.

Score: 1.0 without solving the task.

![Image](https://pbs.twimg.com/media/HFe1AnLbYAA_Hjx?format=png&name=large)

---

**Hao Wang** @MogicianTony [2026-04-09](https://x.com/MogicianTony/status/2042300259108635024)

These aren’t isolated bugs. Across 8 benchmarks, we saw the same 7 patterns repeat:

1\. No isolation between agent and evaluator

2\. Answers shipped with the test

3\. eval() on untrusted input

4\. LLM judges vulnerable to prompt injection

5\. Weak string matching

6\. Evaluation logic

---

**Hao Wang** @MogicianTony [2026-04-09](https://x.com/MogicianTony/status/2042300261218337027)

This is not hypothetical.

The capability to hack evaluations is emerging even without explicit instruction.

Evaluator-hacking behavior has already surfaced in IQuest-Coder-V1, o3, Claude 3.7 Sonnet, and the recent Mythos Preview.

---

**Hao Wang** @MogicianTony [2026-04-09](https://x.com/MogicianTony/status/2042300263357431977)

Why this matters:

• Model selection: leaderboard comparisons may be noise

• Investment: benchmark positions can influence funding

• Safety evaluation: fragile capability evals imply fragile safety evals

• Research direction: broken benchmarks steer the field toward the wrong