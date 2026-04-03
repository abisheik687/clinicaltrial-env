"""Grader for Task 3."""

from server.graders.base_grader import BaseGrader


class Task3Grader(BaseGrader):
    """Grader for ambiguous criteria and amendment handling."""

    def grade(self, truth: dict[str, str], evaluated: dict[str, str], final_action: str, context: dict[str, object]) -> dict[str, object]:
        criteria_accuracy = sum(1 for key, value in truth.items() if evaluated.get(key) == value) / len(truth)
        score = criteria_accuracy * 0.4
        final_correct = int(((final_action == "enroll") and context["final_eligible"]) or ((final_action == "exclude") and not context["final_eligible"]))
        score += 0.2 * final_correct
        if context.get("amendment_detected"):
            score += 0.15
        if context.get("ambiguity_handled"):
            score += 0.15
        if int(context.get("steps_used", 999)) <= 15:
            score += 0.10
        penalty = 0.0
        if final_action == "defer":
            penalty += 0.2
        if context.get("ignored_amendment"):
            penalty += 0.15
        score = min(max(score - penalty, 0.0), 1.0)
        return {
            "score": round(score, 4),
            "partial_credit": {criterion: round(0.4 / len(truth), 4) if evaluated.get(criterion) == truth[criterion] else 0.0 for criterion in truth},
            "feedback": f"Criteria accuracy {criteria_accuracy:.2f}; amendment bonus {context.get('amendment_detected', False)}; penalties {penalty:.2f}.",
        }
