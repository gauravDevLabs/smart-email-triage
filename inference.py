import os
import sys
import json
from datetime import datetime
from typing import Optional
from openai import OpenAI

from models import (
    EmailObservation, AgentAction, EmailCategory, EmailPriority
)
from environment import SmartEmailTriageEnv
from grader import EmailTriageGrader


# ── Structured logger ────────────────────────────────────────────────────────
def _emit(line: str) -> None:
    """Write a line directly to stdout and flush — bypasses any print buffering."""
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


# ── OpenAI client (reads env vars set in Space secrets) ──────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
API_KEY      = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN") or ""
HAS_API_KEY  = bool(API_KEY)

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
def run_simulation(output_file: Optional[str] = None):
    env   = SmartEmailTriageEnv()
    agent = BaselineAgent()
    grader = EmailTriageGrader()

    obs  = env.reset()
    done = False
    step = 0
    total_reward = 0.0

    _emit(f"[START] task=email_triage model={MODEL_NAME} timestamp={datetime.utcnow().isoformat()}Z")

    try:
        while not done:
            step += 1
            action = agent.decide(obs)
            next_obs, reward, done, _info = env.step(action)
            total_reward += reward

            _emit(
                f"[STEP] step={step} reward={round(reward, 4)} "
                f"email_id={obs.id} category={action.category.value} "
                f"priority={action.priority.value} archive={action.should_archive} done={done}"
            )

            obs = next_obs
    except Exception as e:
        _emit(f"[STEP] step={step} reward=0.0 error={type(e).__name__}:{e}")

    # Try to produce a final grade report — but always emit [END] no matter what.
    total_score = round(total_reward, 4)
    max_score   = float(step) if step else 0.0
    accuracy    = 0.0
    try:
        state = env.state()
        report = grader.generate_report(state)
        total_score = report.total_score
        max_score   = report.max_possible_score
        accuracy    = report.accuracy_percentage
        report_dict = {
            "summary": {
                "total_score":  report.total_score,
                "max_score":    report.max_possible_score,
                "accuracy":     report.accuracy_percentage,
                "feedback":     report.summary_feedback,
            },
            "results": report.detailed_results,
        }
    except Exception as e:
        report_dict = {"error": f"{type(e).__name__}:{e}"}

    _emit(
        f"[END] task=email_triage total_steps={step} "
        f"total_score={total_score} max_score={max_score} "
        f"accuracy={accuracy} timestamp={datetime.utcnow().isoformat()}Z"
    )

    if output_file:
        try:
            with open(output_file, "w") as f:
                json.dump(report_dict, f, indent=4)
        except Exception:
            pass


# ── Entry point ──────────────────────────────────────────────────────────────
# Run unconditionally so the validator gets output whether it executes
# `python inference.py` or imports the module.
run_simulation()
