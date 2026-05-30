# Hearts AI Training Handoff

Date: 2026-05-26

## Context

This project trains a Hearts AI using `GameV2`, `HeartsLSTM`, `PassingNetwork`, joint DAgger pretraining, and PPO fine-tuning. The user reported that PPO did not help much, the model collapsed when the opponent pool got stronger, and it struggled against three rulebased/Expert opponents.

The main conclusion so far: this was not primarily a model-size issue. The first blockers were training correctness and measurement quality: LSTM sequence padding mismatch, wrong Q-spades feature indexing, stochastic evaluation noise, PPO behavior/logprob mismatch, and checkpoint selection by imitation accuracy instead of game performance.

## Important Code Changes Already Made

- `model.py`
  - `HeartsLSTM.forward()` now accepts `lengths`.
  - Batched training can select the last real sequence timestep instead of accidentally using padding.

- `pretrain_joint.py`, `pretrain.py`, `train_joint.py`, `train.py`
  - Variable-length LSTM batches now pass true sequence lengths.
  - Joint DAgger rollout now chooses AI-vs-Expert execution per game, not per action, reducing broken LSTM history.

- `agent.py`
  - Fixed Q-spades feature id with `Card(Suit.SPADES, 12).to_id()`.
  - Evaluation/non-training play is greedy.
  - Learned pass policy is greedy in evaluation.
  - PPO rollout no longer adds extra 5 percent epsilon-random actions that were not reflected in old log-probs.

- `passing_model.py`
  - Passing action log-probs now match the actual sampling probabilities used by sequential selection.

- `pretrain_joint.py`
  - Pretrained checkpoints are now selected by evaluation vs Expert average score, not just action accuracy.
  - Checkpoint metadata includes `eval_vs_expert`, `eval_vs_random`, `dagger_round`, and `selection_metric`.

- `train_joint.py`
  - `main.py eval` now reports average score, average rank, win rate, and top-2 rate.
  - Incompatible checkpoint loading is summarized in one line instead of flooding output.

## Current Measured Baseline

After the fixes and re-running `python pretrain_joint.py`, eval-best checkpoint selection chose DAgger Round 2:

- Quick eval during pretraining: vs Expert avg score `9.07`, vs Random avg score `1.30`.

Independent fixed-seed evaluation with `python main.py eval` over 200 games:

- vs Expert: Avg Score `9.83`, Avg Rank `2.19`, Win Rate `40.0%`, Top-2 `61.0%`.
- vs Random: Avg Score `2.19`, Avg Rank `1.41`, Win Rate `74.5%`, Top-2 `88.5%`.

Interpretation: the model is not helpless against three Experts anymore, but it is not reliably better than them. Average score is still high for Expert games, likely because losing games include large point disasters.

## Known Issues / Caveats

- `output/simple_fcn_model.pth` is an old incompatible RL checkpoint. `main.py eval` skips it and falls back to `simple_fcn_pretrained.pth`, but the file still exists and can confuse future experiments.
- The current pretraining eval uses 100 games per DAgger round, which is noisy. More stable selection should use more games or multi-seed eval if runtime allows.
- PPO is still risky if both play and pass networks are updated together from the start.
- `ExpertPolicy` is still the imitation target, so DAgger alone should not be expected to consistently beat three Experts.
- `GameV2`/`strategies.py` still have possible environment/Expert logic gaps to revisit later, such as STM-related fields and piggy tracking.

## Recommended Next Step

Do not immediately run large joint PPO. Start with a controlled PPO experiment:

1. Use the eval-best pretrained checkpoints as the starting point.
2. Freeze or bypass learned passing first; use Expert passing for player 0, or freeze the current pass model.
3. Train only the play policy with PPO against a mixed curriculum.
4. Add a small BC anchor / imitation loss to keep the policy near Expert behavior while PPO explores improvements.
5. Evaluate every checkpoint with fixed seed against both Expert and Random, reporting avg score, avg rank, win rate, and top-2 rate.
6. Only after play-only PPO is stable, unfreeze passing or introduce a snapshot opponent pool.

Suggested experiment label: `play_only_ppo_bc_anchor`.

## Verification Commands Used

```powershell
conda activate research
python -m py_compile model.py agent.py pretrain_joint.py train_joint.py pretrain.py train.py passing_model.py
python pretrain_joint.py
python main.py eval
```

## Files To Inspect First

- `train_joint.py`
- `pretrain_joint.py`
- `agent.py`
- `model.py`
- `passing_model.py`
- `strategies.py`
- `game.py`
- `config.py`

## Update (2026-05-28)

### What Was Changed

- `train_joint.py`
  - Added controlled PPO mode defaults: play-only PPO with Expert passing for player 0.
  - Added BC anchor loss in play-policy PPO update using expert actions collected during rollout.
  - Added startup fixed-seed baseline eval before PPO and deterministic periodic eval during training.
  - Added more frequent rollout progress printing and line-buffered stdout to avoid "looks stuck" runs.
  - Fixed eval trigger logic: eval now triggers when episode passes threshold (not only exact divisibility).
  - Added robust checkpoint management:
    - Save a unique eval snapshot on every eval to `output/ppo_snapshots/`.
    - Only overwrite `simple_fcn_model_best.pth` if new eval is better than both in-memory and on-disk best score.
    - Read disk best score at startup so restarts do not accidentally regress best selection logic.

- `agent.py`
  - Added `saved_expert_actions` buffer and stores Expert policy action id for each rollout step (used by BC anchor).

- `config.py`
  - Added PPO control knobs (`PPO_TRAIN_PASSING`, `PPO_USE_EXPERT_PASSING`, `PPO_PROGRESS_INTERVAL`, `PPO_EVAL_INTERVAL`).
  - Added conservative PPO defaults:
    - `LR=5e-5`, `BATCH_SIZE=512`, `PPO_EPOCHS=2`, `CLIP_EPS=0.1`, `ENTROPY_COEF=0.005`, `BC_ANCHOR_COEF=0.2`.
  - Added `BEST_MODEL_PATH`, `PASSING_BEST_MODEL_PATH`, and `PPO_SNAPSHOT_DIR`.

### Main Findings From Long PPO Runs

- Rollout average score can look much better (often 5-8 range) while fixed-seed eval vs Expert does not improve consistently.
- In one long run, best fixed-seed Expert eval was around episode `10112` (`vs_expert_avg_score=9.535`) but later checkpoints regressed.
- Running to `80k+` episodes did not deliver stable improvement over early best eval; practical stopping range for this setup is closer to `10k-30k`.

### Critical Checkpoint Incident

- A later run showed `simple_fcn_model_best.pth` metadata pointing to a worse checkpoint (`episode ~80000`, `vs_expert_avg_score ~10.19`), even though earlier eval logs had better values.
- Immediate mitigation already done:
  - Restored current `simple_fcn_model_best.pth` and `passing_model_best.pth` to pretrained baseline metadata (`episode 0`, `vs_expert_avg_score 9.84`) to avoid default eval loading a clearly worse checkpoint.
- Important caveat:
  - The previously better weights around episode `10112` are not recoverable from logs alone if no snapshot was saved.

### Recommended Next Step (For Next Agent)

1. Run a fresh controlled PPO experiment with the new snapshot logic enabled.
2. Stop early at `10k-30k` unless fixed-seed eval clearly keeps improving.
3. Evaluate candidate snapshots with larger game counts (for example 1000 games) before declaring wins.
4. Prefer selecting deployment checkpoint from `output/ppo_snapshots/` based on fixed-seed Expert eval, not rollout score.

