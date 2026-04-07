"""Grader for Task 2."""

from server.graders.base_grader import BaseGrader


class Task2Grader(BaseGrader):
    """Weighted grader for oncology screening."""

    def grade(self, truth: dict[str, str], evaluated: dict[str, str], final_action: str, context: dict[str, object]) -> dict[str, object]:
        inclusion = [f"INC-00{i}" for i in range(1, 6)]
        exclusion = [f"EXC-00{i}" for i in range(1, 5)]
        inc_weight = 0.5 / len(inclusion)
        exc_weight = 0.35 / len(exclusion)
        partial: dict[str, float] = {}
        score = 0.0
        for criterion in inclusion:
            partial_score = 0.1 if criterion == "INC-004" else inc_weight
            earned = partial_score if evaluated.get(criterion) == truth[criterion] else 0.0
            partial[criterion] = earned
            score += earned
        for criterion in exclusion:
            earned = exc_weight if evaluated.get(criterion) == truth[criterion] else 0.0
            partial[criterion] = earned
            score += earned
        final_correct = int(((final_action == "enroll") and context["final_eligible"]) or ((final_action == "exclude") and not context["final_eligible"]))
        score += 0.15 * final_correct
        penalty = 0.0
        if context.get("drug_interaction_miss"):
            penalty += 0.1
        penalty += 0.05 * int(context.get("unnecessary_clarifications", 0))
        score = self.clamp_open_unit_interval(score - penalty)
        return {
            "score": round(score, 4),
            "partial_credit": partial,
            "feedback": f"Weighted score with penalty {penalty:.2f}; final decision correct: {bool(final_correct)}.",
        }
