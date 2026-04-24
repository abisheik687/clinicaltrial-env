# Training And Evaluation

This folder contains the Phase 1 training-first workflow for the finalist version of `clinicaltrial-env`.

## Files

- `grpo_phase1.py`: minimal GRPO training entrypoint for the Task 3 finals workflow.
- `evaluate_models.py`: held-out evaluator for baseline or trained models.
- `plot_results.py`: generates the two required judging plots.
- `phase1_colab.ipynb`: lightweight Colab notebook that installs deps and runs the Task 3 workflow.

## Recommended order

1. Start the environment locally or on a private Hugging Face Space.
2. Run the Task 3 fallback evaluator.
3. Run the Phase 1 GRPO script against `task3`.
4. Evaluate the trained checkpoint on the same held-out Task 3 seeds.
5. Generate plots from the saved logs.

## Example commands

```bash
python training/evaluate_models.py --policy fallback --task-ids task3 --output artifacts/eval/fallback_task3_eval.json
python training/grpo_phase1.py --env-url http://localhost:7860 --output-dir artifacts/phase1_grpo
python training/evaluate_models.py --policy local_model --model-name path/to/checkpoint --task-ids task3 --output artifacts/eval/trained_task3_eval.json
python training/plot_results.py
```

## Notes

- The training script intentionally targets the minimal single-patient Task 3 workflow first.
- The terminal verifier lives inside the environment; the trainer only reads `env.reward`.
- If GRPO does not improve held-out success after two checkpoints, simplify the task distribution before adding any new shaping terms.
