import multiprocessing
from typing import List
from environment import SmartEmailTriageEnv
from models import AgentAction, EmailObservation
from inference import BaselineAgent


class ModelOrchestrator:
    """Registry to manage which model runs in which process."""

    @staticmethod
    def get_agent(model_name: str):
        if model_name == "baseline":
            return BaselineAgent()
        else:
            raise ValueError(f"Unknown model: {model_name}")

    @classmethod
    def run_triage_task(cls, task_data: dict) -> dict:
        """The target function for multiprocessing. Returns a serializable dict."""
        model_name = task_data["model_name"]
        observation = EmailObservation(**task_data["email"])

        agent = cls.get_agent(model_name)
        action: AgentAction = agent.decide(observation)

        return {
            "email_id": observation.id,
            "category": action.category.value,
            "priority": action.priority.value,
            "should_archive": action.should_archive,
            "reasoning": action.reasoning,
        }

    @classmethod
    def run_parallel(cls, task_list: List[dict]) -> List[dict]:
        """Run triage tasks across multiple processes."""
        with multiprocessing.Pool() as pool:
            results = pool.map(cls.run_triage_task, task_list)
        return results
