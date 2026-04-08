import copy
from typing import Tuple, List, Optional, Dict, Any
from openenv.env.env import Env
from models import (
    EmailObservation, AgentAction, StepReward,
    EpisodeResult, EnvironmentState, EmailCategory, EmailPriority
)
from tasks import get_tasks, EmailTask

class SmartEmailTriageEnv(Env):
    """
    A reinforcement learning environment for intelligent email triage.
    Simulates real-world constraints and provides deterministic grading.
    """
    
    def __init__(self):
        super().__init__(
            name="SmartEmailTriageEnv",
            state_space=["id", "sender", "subject", "body", "timestamp"],
            action_space=["category", "priority", "should_archive", "response_draft", "reasoning"],
            episode_max_length=len(get_tasks()),
        )
        self.tasks: List[EmailTask] = get_tasks()
        self.current_index = 0
        self.history: List[EpisodeResult] = []
        self.is_done = False
        self.max_steps_per_email = 1  # Standard triage is a single decision per email
        self.current_step_count = 0

    def reset(self) -> EmailObservation:
        """Initialize the environment and return the first observation."""
        self.current_index = 0
        self.history = []
        self.is_done = False
        self.current_step_count = 0
        
        if not self.tasks:
            raise ValueError("No tasks available in the environment.")
            
        return self._get_current_observation()

    def step(self, action: AgentAction) -> Tuple[Optional[EmailObservation], float, bool, Dict[str, Any]]:
        """
        Processes the agent's action for the current email.
        Returns: (next_observation, reward, done, info)
        """
        if self.is_done:
            raise RuntimeError("Cannot step in a finished environment. Call reset() first.")

        current_task = self.tasks[self.current_index]
        
        # 1. Compute Reward
        reward_breakdown = self._compute_reward(action, current_task)
        total_reward = reward_breakdown.compute_total()
        
        # Normalize reward to [0.0, 1.0] (max possible per email is 2.8)
        normalized_reward = max(0.0, min(1.0, total_reward / 2.8))

        # 2. Record Result
        is_correct = self._check_correctness(action, current_task)
        result = EpisodeResult(
            email_id=current_task.id,
            action=action,
            reward=reward_breakdown,
            is_correct=is_correct,
            feedback=self._generate_feedback(action, current_task)
        )
        self.history.append(result)

        # 3. Move to next email
        self.current_index += 1
        if self.current_index >= len(self.tasks):
            self.is_done = True
            next_obs = None
        else:
            next_obs = self._get_current_observation()

        info = {
            "email_id": current_task.id,
            "step_reward": reward_breakdown,
            "is_correct": is_correct
        }

        return next_obs, normalized_reward, self.is_done, info

    def state(self) -> EnvironmentState:
        """Returns a snapshot of the current environment state."""
        return EnvironmentState(
            current_email_index=self.current_index,
            total_emails=len(self.tasks),
            processed_emails=[res.email_id for res in self.history],
            current_episode_rewards=[res.reward.compute_total() for res in self.history],
            is_done=self.is_done,
            history=copy.deepcopy(self.history)
        )

    def _get_current_observation(self) -> EmailObservation:
        """Helper to create an observation from the current task."""
        task = self.tasks[self.current_index]
        return EmailObservation(
            id=task.id,
            sender=task.sender,
            subject=task.subject,
            body=task.body,
            timestamp=task.timestamp
        )

    def _compute_reward(self, action: AgentAction, task: EmailTask) -> StepReward:
        """Logic for reward shaping based on ground truth."""
        reward = StepReward()
        
        # Category reward
        if action.category == task.ground_truth_category:
            reward.classification_score = 1.0
        else:
            reward.classification_score = -0.5
            
        # Priority reward
        if action.priority == task.ground_truth_priority:
            reward.priority_score = 0.5
        else:
            # Partial credit for being close (e.g., NORMAL vs HIGH)
            reward.priority_score = -0.2
            
        # Archiving reward
        if action.should_archive == task.ground_truth_should_archive:
            reward.archiving_score = 0.3
        else:
            reward.archiving_score = -0.1
            
        # Response quality (heuristic)
        if task.keywords_in_response:
            if action.response_draft:
                matches = sum(1 for word in task.keywords_in_response if word.lower() in action.response_draft.lower())
                reward.response_quality_score = (matches / len(task.keywords_in_response)) * 1.0
            else:
                reward.response_quality_score = -0.5 # Penalty for missing required response
        elif action.category == EmailCategory.SPAM:
            # Spam shouldn't have responses
            if action.response_draft:
                reward.penalty = 0.5
                
        return reward

    def _check_correctness(self, action: AgentAction, task: EmailTask) -> bool:
        """Binary check for whether the core triage was successful."""
        return (
            action.category == task.ground_truth_category and
            action.priority == task.ground_truth_priority and
            action.should_archive == task.ground_truth_should_archive
        )

    def _generate_feedback(self, action: AgentAction, task: EmailTask) -> List[str]:
        """Human-readable feedback for the agent's action."""
        feedback = []
        if action.category != task.ground_truth_category:
            feedback.append(f"Incorrect category. Expected {task.ground_truth_category}, got {action.category}.")
        if action.priority != task.ground_truth_priority:
            feedback.append(f"Incorrect priority. Expected {task.ground_truth_priority}, got {action.priority}.")
        if action.should_archive != task.ground_truth_should_archive:
            feedback.append(f"Archiving error. Should have been {task.ground_truth_should_archive}.")
        
        if not feedback:
            feedback.append("Perfect triage decisions!")
            
        return feedback


