You are a senior AI systems engineer and RL environment designer.

Your task is to build a COMPLETE, production-grade reinforcement learning environment that simulates a real-world intelligent email triage system.

This is NOT a toy project. It must be architected like a real product.

---

## 🎯 SYSTEM GOAL

Design an environment where an AI agent learns to manage an email inbox autonomously by:

* Classifying emails (spam / important / normal)
* Prioritizing emails
* Generating appropriate responses
* Deciding whether to archive

The environment must follow a strict OpenEnv-style API and support realistic decision-making.

---

## 🧱 REQUIRED ARCHITECTURE

You MUST implement the system as modular Python files:

### 1. models.py

Define structured data models for:

* EmailObservation (input to agent)
* AgentAction (agent output)
* StepReward (reward breakdown + compute_total method)
* EpisodeResult (summary per email)
* EnvironmentState (full internal state snapshot)

Use Pydantic if available, otherwise fallback to dataclasses.

---

### 2. tasks.py

Create realistic task datasets and logic:

* Minimum 8 emails
* 3 difficulty tiers:

  * Easy → spam detection
  * Medium → prioritization (weighted scoring)
  * Hard → full workflow (classification + response + archive)

Each task must include:

* Ground truth labels
* Expected behavior
* Scoring logic

---

### 3. environment.py

Implement class: SmartEmailTriageEnv

Must support:

* reset() → initializes episode and returns first observation
* step(action) → processes agent action and returns:
  (next_observation, reward, done, info)
* state() → returns full environment snapshot

Constraints:

* Sequential email processing
* Max steps per email (prevent loops)
* Deterministic transitions
* Clear reward shaping

---

### 4. grader.py

Implement a deterministic grading system:

* Per-email scoring
* Aggregate scoring
* Human-readable feedback
* GradeReport object

---

### 5. baseline.py

CLI runner that:

* Simulates an agent using an LLM
* Uses JSON output format
* Handles parsing errors robustly
* Outputs evaluation results

---

### 6. openenv.yaml

Define:

* Observation space schema
* Action space schema
* Reward definitions
* Task registry
* Metadata

---

### 7. Dockerfile

Production-ready container:

* Minimal image
* Dependency installation
* Health check
* Environment variable support

---

## ⚙️ DESIGN REQUIREMENTS

* The environment must simulate REAL-WORLD constraints (not games)
* Rewards must be decomposed and explainable
* All logic must be deterministic (no randomness in grading)
* Code must be clean, modular, and extensible
* Include meaningful comments explaining design decisions

---

## 🧠 INTELLIGENCE EXPECTATIONS

The system must:

* Handle ambiguous email content
* Score response quality heuristically
* Penalize incorrect prioritization
* Reward correct multi-step decisions

---

## 🚫 WHAT NOT TO DO

* Do NOT oversimplify logic
* Do NOT hardcode answers without reasoning
* Do NOT skip reward design details
* Do NOT produce pseudo-code — write real, runnable code

---

## ✅ OUTPUT FORMAT

Generate ALL required files with:

* Correct imports
* No syntax errors
* Ready-to-run structure

Also include:

* Brief explanation of design decisions
* Example run flow

---

## 🏁 SUCCESS CRITERIA

A judge should look at this and say:
"This is a real AI environment, not a demo."

---

Now begin building the system.
