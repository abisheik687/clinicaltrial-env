"""Grader for Task 1."""

from server.graders.base_grader import BaseGrader


class Task1Grader(BaseGrader):
    """Easy task grader with capped score on wrong final decision."""

    def grade(self, truth: dict[str, str], evaluated: dict[str, str], final_action: str, context: dict[str, object]) -> dict[str, object]:
        criteria = list(truth.keys())
        correct = sum(1 for criterion in criteria if evaluated.get(criterion) == truth[criterion])
        criteria_score = (correct / 5.0) * 0.6
        final_correct = int(((final_action == "enroll") and context["final_eligible"]) or ((final_action == "exclude") and not context["final_eligible"]))
        score = criteria_score + (0.4 * final_correct)
        if not final_correct:
            score = min(score, 0.5)
        partial = {criterion: 0.12 if evaluated.get(criterion) == truth[criterion] else 0.0 for criterion in criteria}
        return {
            "score": round(self.clamp_open_unit_interval(score), 4),
            "partial_credit": partial,
            "feedback": f"Correct evaluations: {correct}/5. Final decision correct: {bool(final_correct)}.",
        }
