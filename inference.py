import os
import sys
import json
from typing import List, Optional
from openai import OpenAI

from models import (
    EmailObservation, AgentAction, EmailCategory, EmailPriority
)
from environment import SmartEmailTriageEnv
from grader import EmailTriageGrader


# ── Structured logger ────────────────────────────────────────────────────────
def _emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def log_start(task: str, env: str, model: str) -> None:
    _emit(f"[START] task={task} env={env} model={model}")


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    _emit(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}")


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    _emit(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}")


# ── OpenAI client ────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
API_KEY      = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN") or ""
HAS_API_KEY  = bool(API_KEY)
BENCHMARK    = "smart-email-triage"

client = OpenAI(api_key=API_KEY or "dummy-key", base_url=API_BASE_URL)


# ── Agent ────────────────────────────────────────────────────────────────────
class BaselineAgent:
    """LLM-powered email triage agent using the OpenAI client."""

    SYSTEM_PROMPT = (
        "You are an expert email triage assistant. "
        "Given an email, classify it and decide how to handle it. "
        "Respond ONLY with a valid JSON object — no markdown, no extra text — "
        "with these exact keys:\n"
        "  category       : one of 'spam', 'normal', 'important'\n"
        "  priority       : one of 'low', 'normal', 'high'\n"
        "  should_archive : true or false\n"
        "  response_draft : a short professional reply string, or null\n"
        "  reasoning      : one sentence explaining your decision"
    )

    def _heuristic(self, observation: EmailObservation, reason: str = "") -> AgentAction:
        body    = observation.body.lower()
        sender  = observation.sender.lower()
        subject = observation.subject.lower()
        if "win" in subject or "prize" in body or "malicious" in body or ".biz" in sender:
            return AgentAction(category=EmailCategory.SPAM,     priority=EmailPriority.LOW,  should_archive=True,  reasoning=f"Heuristic: spam {reason}")
        if "ceo" in sender or "urgent" in subject or "latency" in body or "overdue" in subject:
            return AgentAction(category=EmailCategory.IMPORTANT, priority=EmailPriority.HIGH, should_archive=False, reasoning=f"Heuristic: important {reason}")
        return AgentAction(category=EmailCategory.NORMAL, priority=EmailPriority.NORMAL, should_archive=False, reasoning=f"Heuristic: default {reason}")

    def decide(self, observation: EmailObservation) -> AgentAction:
        if not HAS_API_KEY:
            return self._heuristic(observation, "(no API key)")
        user_msg = (
            f"From: {observation.sender}\n"
            f"Subject: {observation.subject}\n"
            f"Body: {observation.body}"
        )
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=300,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            data = json.loads(raw)
            return AgentAction(
                category=EmailCategory(data.get("category", "normal")),
                priority=EmailPriority(data.get("priority", "normal")),
                should_archive=bool(data.get("should_archive", False)),
                response_draft=data.get("response_draft"),
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            return self._heuristic(observation, f"(API error: {e})")


# ── Main simulation ──────────────────────────────────────────────────────────
def run_simulation():
    env   = SmartEmailTriageEnv()
    agent = BaselineAgent()

    obs  = env.reset()
    done = False
    step_num = 0
    all_rewards: List[float] = []

    while not done:
        step_num += 1
        task_id = obs.id

        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        action = agent.decide(obs)
        next_obs, reward, done, _info = env.step(action)
        reward = round(reward, 2)
        all_rewards.append(reward)

        action_str = f"classify(category={action.category.value},priority={action.priority.value},archive={str(action.should_archive).lower()})"

        log_step(step=1, action=action_str, reward=reward, done=True, error=None)

        success = reward >= 0.40
        log_end(success=success, steps=1, score=reward, rewards=[reward])

        obs = next_obs


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_simulation()
