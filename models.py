from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class EmailPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

class EmailCategory(str, Enum):
    SPAM = "spam"
    NORMAL = "normal"
    IMPORTANT = "important"

class EmailObservation(BaseModel):
    """Input to the agent representing an email to triage."""
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    priority_hint: Optional[str] = None

class AgentAction(BaseModel):
    """Agent output for triaging an email."""
    category: EmailCategory
    priority: EmailPriority
    response_draft: Optional[str] = None
    should_archive: bool = False
    reasoning: Optional[str] = None

class StepReward(BaseModel):
    """Detailed breakdown of the reward for a single step."""
    classification_score: float = 0.0
    priority_score: float = 0.0
    response_quality_score: float = 0.0
    archiving_score: float = 0.0
    penalty: float = 0.0

    def compute_total(self) -> float:
        """Calculate the aggregate reward."""
        return (
            self.classification_score +
            self.priority_score +
            self.response_quality_score +
            self.archiving_score -
            self.penalty
        )

class EpisodeResult(BaseModel):
    """Summary of the agent's performance on a single email."""
    email_id: str
    action: AgentAction
    reward: StepReward
    is_correct: bool
    feedback: List[str] = Field(default_factory=list)

class EnvironmentState(BaseModel):
    """Full internal state snapshot of the environment."""
    current_email_index: int
    total_emails: int
    processed_emails: List[str]
    current_episode_rewards: List[float]
    is_done: bool
    history: List[EpisodeResult] = Field(default_factory=list)
