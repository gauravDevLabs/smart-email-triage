from pydantic import BaseModel
from typing import List, Dict, Any
from models import EpisodeResult, EnvironmentState


class GradeReport(BaseModel):
    """Detailed summary of the performance across an entire episode."""
    total_score: float
    max_possible_score: float
    accuracy_percentage: float
    category_breakdown: Dict[str, float]
    summary_feedback: str
    detailed_results: List[Dict[str, Any]]


class EmailTriageGrader:
    """
    Deterministic grading system for the SmartEmailTriageEnv.
    Provides human-readable feedback and performance metrics.
    """

    def generate_report(self, state: EnvironmentState) -> GradeReport:
        """Analyze the environment state and generate a comprehensive GradeReport."""
        results = state.history
        total_score = sum(max(0.0, min(1.0, res.reward.compute_total() / 2.8)) for res in results)
        max_possible_score = float(len(results))  # Each email scores 0.0–1.0

        correct_count = sum(1 for res in results if res.is_correct)
        accuracy = (correct_count / len(results)) * 100 if results else 0

        category_breakdown = self._compute_category_breakdown(results)

        detailed_results = []
        for res in results:
            detailed_results.append({
                "email_id": res.email_id,
                "reward": round(max(0.0, min(1.0, res.reward.compute_total() / 2.8)), 4),
                "is_correct": res.is_correct,
                "feedback": res.feedback,
                "action": {
                    "category": res.action.category,
                    "priority": res.action.priority,
                    "should_archive": res.action.should_archive,
                }
            })

        summary = self._generate_summary(accuracy)

        return GradeReport(
            total_score=round(total_score, 2),
            max_possible_score=round(max_possible_score, 2),
            accuracy_percentage=round(accuracy, 1),
            category_breakdown=category_breakdown,
            summary_feedback=summary,
            detailed_results=detailed_results,
        )

    def _compute_category_breakdown(self, results: List[EpisodeResult]) -> Dict[str, float]:
        """Calculates accuracy per predicted email category."""
        categories = ["spam", "normal", "important"]
        correct_counts = {cat: 0 for cat in categories}
        total_counts = {cat: 0 for cat in categories}

        for res in results:
            predicted = res.action.category.value
            if predicted in total_counts:
                total_counts[predicted] += 1
                if res.is_correct:
                    correct_counts[predicted] += 1

        return {
            cat: round((correct_counts[cat] / total_counts[cat]) * 100, 1) if total_counts[cat] > 0 else 0.0
            for cat in categories
        }

    def _generate_summary(self, accuracy: float) -> str:
        """Generates a high-level performance summary."""
        if accuracy >= 90:
            return "Excellent performance! The agent demonstrated superior triage capability."
        elif accuracy >= 70:
            return "Good performance. The agent correctly handled most emails but has room for improvement."
        elif accuracy >= 50:
            return "Average performance. Significant errors were observed in categorization or prioritization."
        else:
            return "Poor performance. The agent failed to reliably triage emails according to the ground truth."
