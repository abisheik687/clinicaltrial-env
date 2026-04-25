# Training And Evaluation

This folder contains the training and evidence-generation workflows for `clinicaltrial-env`.

## Files

- `grpo_phase1.py`: LM-GRPO attempt for the Task 3 workflow; current artifacts are not claimed as a passed run.
- `train_task3_policy_gradient.py`: compact verifier-trained action-policy rescue path.
- `evaluate_models.py`: held-out evaluator for baseline or trained models.
- `plot_results.py`: generates the two required judging plots.
- `phase1_colab.ipynb`: lightweight Colab notebook that installs deps and runs the Task 3 workflow.

## Recommended order

1. Start the environment locally or on a private Hugging Face Space.
2. Run the Task 3 fallback evaluator.
3. Run the raw base-model Task 3 evaluator.
4. Run the compact action-policy trainer for the passed verifier-policy artifact.
5. Optionally run the LM-GRPO attempt and validate it separately with `--allow-failed`.
6. Generate plots from the method-specific logs.

## Example commands

```bash
python training/evaluate_models.py --policy fallback --task-ids task3 --output artifacts/eval/fallback_task3_eval.json
python training/evaluate_models.py --policy local_model --model-name Qwen/Qwen2.5-0.5B-Instruct --task-ids task3 --output artifacts/eval/base_model_task3_eval.json
python training/train_task3_policy_gradient.py --seed-start 200 --train-seed-count 50 --eval-seed-start 200 --eval-num-seeds 50 --train-steps 15 --batch-size 50 --learning-rate 0.3 --log-every 1
python training/validate_training_outputs.py --mode action_policy --train-log artifacts/phase1_pg/train_log_history.json --trained-eval artifacts/eval/policy_gradient_task3_eval.json
python training/grpo_phase1.py --env-url http://localhost:7860 --output-dir artifacts/phase1_grpo
python training/validate_training_outputs.py --mode lm_grpo --allow-failed --train-log artifacts/phase1_grpo/train_log_history.json --trained-eval artifacts/eval/lm_grpo_task3_eval_failed.json
```

## Notes

- The compact policy is not an LLM fine-tune; it is a verifier-trained action policy.
- The terminal verifier lives inside the environment; the trainer only reads `env.reward`.
- The plotting script prefers `base_model_task3_eval.json` as the baseline if it exists, and otherwise falls back to the heuristic Task 3 reference.
- If GRPO does not improve held-out success after two checkpoints, simplify the task distribution before adding any new shaping terms.
