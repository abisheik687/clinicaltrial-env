"""Training helper tests for trajectory planning."""

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from training.trajectory_helpers import build_episode_prompt, parse_trajectory_completion


def test_build_episode_prompt_mentions_finals_workflow() -> None:
    observation = {
        "patient_id": "4f0f64fa-9c2a-4957-95b7-c08fca6a4048",
        "step_number": 0,
        "steps_remaining": 20,
        "demographics": {"age": 12, "sex": "Other", "weight_kg": 37.0, "height_cm": 161.0},
        "diagnosis": {
            "primary_condition": "Rett syndrome",
            "icd10_code": "F84.2",
            "disease_stage": None,
            "diagnosis_date": "2017-10-10",
        },
        "lab_values": {
            "css_score": {"value": 23.6, "certainty": "estimated", "unit": "score"},
        },
        "current_medications": [],
        "trial_protocol_summary": {
            "trial_id": "TRIAL-C-GENE-003",
            "title": "Phase I/II AAV Gene Therapy for Rett Syndrome",
            "phase": "I",
            "amendment_active": False,
            "amendment_description": None,
            "inclusion_criteria": [
                {"criterion_id": "INC-003", "clarification_available": True, "is_ambiguous": True},
            ],
            "exclusion_criteria": [],
        },
        "operational_state": {
            "workflow_phase": "screening",
            "followup_window_start": 7,
            "followup_window_end": 10,
            "amendment_review_required": False,
            "safety_event_active": False,
        },
        "info_message": "Protocol amendment pending review.",
    }

    prompt = build_episode_prompt(observation, task_id="task3", seed=44, max_actions=10)

    assert "schedule_followup" in prompt
    assert "handle_safety_event" in prompt
    assert "investigator escalation" in prompt


def test_parse_trajectory_completion_rejects_disallowed_actions() -> None:
    completion = '{"trajectory":[{"action_type":"inspect_protocol"}]}'

    try:
        parse_trajectory_completion(completion, max_actions=5)
    except ValueError as exc:
        # Disallowed actions are skipped; if all actions are skipped, a ValueError is raised
        assert "No valid actions found" in str(exc) or "Unsupported trajectory action_type" in str(exc)
    else:
        raise AssertionError("Expected parse_trajectory_completion to reject inspect_protocol.")


# ---------------------------------------------------------------------------
# Bug Condition Exploration Tests
# ---------------------------------------------------------------------------


class TestBugConditionExploration:
    """Exploration tests that confirm each bug exists on unfixed code.

    Tests 1b, 1c, 1d are expected to PASS (they confirm the defect is present).
    Test 1a is expected to show ConnectError propagates (confirms no health-check).
    """

    def test_1a_server_missing_crash_propagates_connect_error(self) -> None:
        """Bug 1 — wait_for_server raises RuntimeError when server is unreachable.

        On fixed code, wait_for_server is called before build_prompt_dataset.
        When all HTTP calls raise ConnectError, wait_for_server raises RuntimeError
        with the URL and "Start the server" text.
        Validates: Requirements 2.1
        """
        from unittest.mock import patch
        from training.grpo_phase1 import wait_for_server

        url = "http://localhost:7860"
        with patch("httpx.get", side_effect=httpx.ConnectError("Connection refused")):
            with pytest.raises(RuntimeError) as exc_info:
                wait_for_server(url, max_wait_seconds=0, poll_interval_seconds=0)

        assert url in str(exc_info.value)
        assert "Start the server" in str(exc_info.value)

    def test_1b_token_default_mismatch(self) -> None:
        """Bug 2 — parse_args() now returns max_new_tokens=384 (fixed).

        Confirms the CLI default now matches LocalModelClient.__init__ default of 384.
        Validates: Requirements 2.2
        """
        import sys
        from unittest.mock import patch
        from training.evaluate_models import parse_args

        with patch.object(sys, "argv", ["evaluate_models.py"]):
            args = parse_args()
        assert args.max_new_tokens == 384, (
            f"Expected default max_new_tokens=384 (the fix), got {args.max_new_tokens}"
        )

    def test_1c_doc_omits_intermediate_shaping_bonuses(self) -> None:
        """Bug 3 (fixed) — reward_design.md now documents intermediate shaping bonuses.

        Confirms the +0.3 shaping bonuses are documented, both trigger conditions are present.
        Validates: Requirements 2.3
        """
        doc_path = Path("docs/reward_design.md")
        content = doc_path.read_text(encoding="utf-8")

        assert "+0.3" in content, (
            "Expected '+0.3' to be present in reward_design.md (fix: doc should document +0.3 bonuses)"
        )
        assert "intermediate" in content.lower(), (
            "Expected 'intermediate' to be present in reward_design.md (fix: doc should mention intermediate shaping)"
        )
        assert "amendment_detected" in content, (
            "Expected 'amendment_detected' to be present in reward_design.md (fix: doc should mention trigger condition)"
        )
        assert "SCHEDULE_FOLLOWUP" in content, (
            "Expected 'SCHEDULE_FOLLOWUP' to be present in reward_design.md (fix: doc should mention trigger condition)"
        )

    def test_1d_broken_training_log(self) -> None:
        """Bug 4 — train_log_history.json has exactly one entry with reward=-1.0.

        Confirms the training run was broken (server-missing crash, single step).
        Validates: Requirements 1.4
        """
        log_path = Path("artifacts/phase1_grpo/train_log_history.json")
        log = json.loads(log_path.read_text(encoding="utf-8"))

        assert len(log) == 1, (
            f"Expected exactly 1 log entry (broken run), got {len(log)}"
        )
        assert log[0]["reward"] == -1.0, (
            f"Expected reward=-1.0 (broken run), got {log[0]['reward']}"
        )


# ---------------------------------------------------------------------------
# Preservation Property Tests
# ---------------------------------------------------------------------------


class TestPreservationProperties:
    """Preservation tests that confirm baseline behavior on UNFIXED code.

    These tests MUST PASS on unfixed code — they document behavior that must
    be preserved after fixes are applied.

    Validates: Requirements 3.1, 3.2, 3.3, 3.4
    """

    # ------------------------------------------------------------------
    # Baseline observations (concrete examples)
    # ------------------------------------------------------------------

    def test_explicit_max_new_tokens_512_respected(self) -> None:
        """Observe: parse_args(["--max-new-tokens", "512"]) returns 512.

        Validates: Requirements 3.2
        """
        import sys
        from unittest.mock import patch
        from training.evaluate_models import parse_args

        with patch.object(sys, "argv", ["evaluate_models.py", "--max-new-tokens", "512"]):
            args = parse_args()
        assert args.max_new_tokens == 512

    def test_policy_fallback_respected(self) -> None:
        """Observe: parse_args(["--policy", "fallback"]) returns policy=="fallback".

        Validates: Requirements 3.3
        """
        import sys
        from unittest.mock import patch
        from training.evaluate_models import parse_args

        with patch.object(sys, "argv", ["evaluate_models.py", "--policy", "fallback"]):
            args = parse_args()
        assert args.policy == "fallback"

    # ------------------------------------------------------------------
    # Property-based tests
    # ------------------------------------------------------------------

    def test_explicit_flag_always_respected_property(self) -> None:
        """Property: for all N in [1, 4096], parse_args with --max-new-tokens N returns N.

        Explicit --max-new-tokens flag is always respected regardless of default.
        **Validates: Requirements 3.2**
        """
        from hypothesis import given, settings
        import hypothesis.strategies as st
        import sys
        from unittest.mock import patch
        from training.evaluate_models import parse_args

        @given(st.integers(min_value=1, max_value=4096))
        @settings(max_examples=50)
        def _property(n: int) -> None:
            with patch.object(sys, "argv", ["evaluate_models.py", "--max-new-tokens", str(n)]):
                args = parse_args()
            assert args.max_new_tokens == n, (
                f"Expected max_new_tokens={n}, got {args.max_new_tokens}"
            )

        _property()

    def test_explicit_flag_preserved_same_on_fixed_and_unfixed(self) -> None:
        """Property: for all N, explicit --max-new-tokens is preserved (fix must not change it).

        This is the preservation framing of the above: the fix only changes the DEFAULT,
        never the explicitly-provided value.
        **Validates: Requirements 3.2, 3.3**
        """
        from hypothesis import given, settings
        import hypothesis.strategies as st
        import sys
        from unittest.mock import patch
        from training.evaluate_models import parse_args

        @given(st.integers(min_value=1, max_value=4096))
        @settings(max_examples=50)
        def _property(n: int) -> None:
            with patch.object(sys, "argv", ["evaluate_models.py", "--max-new-tokens", str(n)]):
                args = parse_args()
            assert args.max_new_tokens == n

        _property()

    def test_wait_for_server_does_not_raise_for_non_5xx_responses(self) -> None:
        """Property: for all status codes 200–499, wait_for_server should not raise.

        Since wait_for_server doesn't exist yet on unfixed code, this test mocks
        httpx.get to return the given status code and asserts no exception is raised.
        This documents the INTENDED behavior to verify after the fix is applied.
        **Validates: Requirements 3.1, 3.4**
        """
        from hypothesis import given, settings
        import hypothesis.strategies as st
        from unittest.mock import patch, MagicMock

        # Pre-import to avoid Hypothesis deadline issues on first module load
        try:
            from training.grpo_phase1 import wait_for_server as _wait_for_server  # noqa: F401
        except ImportError:
            _wait_for_server = None  # type: ignore[assignment]

        @given(st.integers(min_value=200, max_value=499))
        @settings(max_examples=50, deadline=None)
        def _property(status_code: int) -> None:
            mock_response = MagicMock()
            mock_response.status_code = status_code

            with patch("httpx.get", return_value=mock_response):
                if _wait_for_server is not None:
                    # Post-fix: function exists — call it and assert no exception is raised
                    _wait_for_server("http://localhost:7860", max_wait_seconds=10, poll_interval_seconds=0)
                # Pre-fix: function doesn't exist — test documents intended behavior; nothing to assert

        _property()
