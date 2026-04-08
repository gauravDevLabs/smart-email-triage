# Smart Email Triage RL Environment

A production-grade reinforcement learning environment for intelligent email triage. This system simulates a real-world inbox where an AI agent must classify, prioritize, and respond to emails autonomously.

## System Goal

The goal is to develop an agent that can:
1. **Classify** emails into `spam`, `normal`, or `important`.
2. **Prioritize** emails into `low`, `normal`, or `high`.
3. **Generate** appropriate responses based on the email content.
4. **Decide** whether to archive the email after processing.

## Architecture

The system is designed with a modular architecture:

- `models.py`: Data models using Pydantic for strict validation.
- `tasks.py`: Dataset containing 8 emails across Easy, Medium, and Hard difficulty tiers.
- `environment.py`: The `SmartEmailTriageEnv` implementing the OpenEnv API.
- `grader.py`: A deterministic grading system providing detailed performance metrics.
- `inference.py`: A CLI runner with an LLM-powered agent to demonstrate the environment.
- `server.py`: FastAPI HTTP server exposing `reset`, `step`, `state` endpoints for HF Spaces.
- `openenv.yaml`: Configuration file defining observation/action spaces and rewards.
- `Dockerfile`: Production-ready container for easy deployment and testing.

## Observation Space

Each observation is an `EmailObservation` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique email identifier |
| `sender` | string | Sender email address |
| `subject` | string | Email subject line |
| `body` | string | Email body text |
| `timestamp` | string | ISO 8601 timestamp |

## Action Space

Each action is an `AgentAction` with the following fields:

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `category` | enum | `spam`, `normal`, `important` | Email classification |
| `priority` | enum | `low`, `normal`, `high` | Triage priority level |
| `should_archive` | boolean | `true`, `false` | Whether to archive the email |
| `response_draft` | string (optional) | any | Draft reply to the email |
| `reasoning` | string (optional) | any | Agent's reasoning |

## Reward Function

Rewards are decomposed per email and normalized to `[0.0, 1.0]`:

| Component | Max | Description |
|-----------|-----|-------------|
| Classification | 1.0 | Correct `category` match |
| Priority | 0.5 | Correct `priority` match |
| Archiving | 0.3 | Correct `should_archive` decision |
| Response quality | 1.0 | Key keywords present in `response_draft` |
| Spam response penalty | -0.5 | Penalizes responding to spam |

Max raw score per email: **2.8** → normalized to **1.0**.

## Tasks

| ID | Difficulty | Description |
|----|-----------|-------------|
| E001 | Easy | Obvious spam with malicious link and prize offer |
| E002 | Easy | Promotional marketing email |
| E003 | Easy | Unsolicited newsletter from unknown sender |
| M001 | Medium | HR benefits update requiring acknowledgement |
| M002 | Medium | System alert for high database latency |
| M003 | Medium | Personal email (dinner invitation) |
| M004 | Medium | Scheduled maintenance notice — informational, archive after reading |
| H001 | Hard | Urgent CEO request with tight deadline and archive requirement |
| H002 | Hard | Overdue invoice with service interruption threat |
| H003 | Hard | Customer support login issue requiring helpful response |
| H004 | Hard | Phishing email disguised as IT security alert |
| H005 | Hard | CEO gift card scam (social engineering) |
| H006 | Hard | Legal litigation notice requiring immediate escalation |
| H007 | Hard | Production deployment approval buried in email thread |
| H008 | Hard | Critical security vulnerability (CVSS 9.8) from GitHub Dependabot |

## Baseline Scores

Baseline agent: `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace Inference API.

| Metric | Score |
|--------|-------|
| Total Score | 3.11 / 15.0 |
| Accuracy | 33.3% (5/15 correct) |
| Emails correct | E001, M001, M002, H002, H003 |
| Hard task accuracy | 0% — adversarial tasks (H004–H008) fooled the baseline model |

## Getting Started

### Prerequisites

- Python 3.11+
- Docker (optional)

### Installation

```bash
pip install -r requirements.txt
```

### Running the Inference Script

Set your API credentials:

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
```

Run inference:

```bash
python inference.py
```

To save results to a JSON file:

```bash
python inference.py --output results.json
```

### Running the API Server

```bash
python server.py
```

API available at `http://localhost:7860` with endpoints:
- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`

### Using Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

## Design Decisions

- **Pydantic Models**: Used for production-grade data validation and easy serialization to JSON.
- **Deterministic Rewards**: Rewards are explicitly broken down into classification, priority, archiving, and response quality to provide clear feedback to the agent.
- **Sequential Processing**: The environment handles emails one by one, simulating a real-world workflow where an agent must make a decision before moving to the next item.
- **Heuristic Response Grading**: Responses are graded based on the presence of key contextually relevant keywords, balancing simplicity with effectiveness.
- **Modular Tasks**: Tasks are separated from the environment logic, allowing for easy expansion of the dataset without modifying the core simulation code.
