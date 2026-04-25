# Colab Reproduction: Task 3 Policy-Gradient Artifact Run

Use this notebook flow when the Hugging Face/TRL GRPO language-model path is blocked, too slow, or not producing valid JSON trajectories. It trains the compact Task 3 action policy with real ClinicalTrialEnv verifier rewards. This is an action-policy artifact, not a successful LLM-GRPO checkpoint.

## Cell 1: Clone and Install

```python
!git clone https://github.com/abisheik687/clinicaltrial-env.git
%cd clinicaltrial-env
!python -m pip install -U pip
!python -m pip install -r requirements.txt
!python -m pip install torch matplotlib
```

## Cell 2: Run the RL Artifact Trainer

```python
!python training/train_task3_policy_gradient.py \
  --seed-start 200 \
  --train-seed-count 50 \
  --eval-seed-start 200 \
  --eval-num-seeds 50 \
  --train-steps 15 \
  --batch-size 50 \
  --learning-rate 0.3 \
  --log-every 1
```

The script fails with a non-zero exit code unless the final held-out evaluation reaches at least 90% success and 0% unsafe actions.

## Cell 3: Inspect the Final Metrics

```python
import json

with open("artifacts/eval/policy_gradient_task3_eval.json", "r", encoding="utf-8") as f:
    trained = json.load(f)

with open("artifacts/phase1_pg/train_log_history.json", "r", encoding="utf-8") as f:
    history = json.load(f)

print(json.dumps(trained["aggregate"], indent=2))
print(history[0])
print(history[-1])
```

Expected aggregate after the provided run settings:

```json
{
  "success_rate": 1.0,
  "unsafe_rate": 0.0,
  "fallback_used_rate": 0.0,
  "mean_final_reward": 1.0,
  "amendment_recovery_rate": 1.0,
  "eligibility_component_score": 1.0,
  "amendment_component_score": 1.0
}
```

## Cell 4: Display and Download the Artifacts

```python
from IPython.display import Image, display

display(Image("artifacts/plots/training_reward_curve.png"))
display(Image("artifacts/plots/heldout_base_vs_trained.png"))
```

```python
from google.colab import files

for path in [
    "artifacts/plots/training_reward_curve.png",
    "artifacts/plots/heldout_base_vs_trained.png",
    "artifacts/eval/policy_gradient_task3_eval.json",
    "artifacts/phase1_pg/train_log_history.json",
    "artifacts/phase1_pg/task3_policy.pt",
]:
    files.download(path)
```
