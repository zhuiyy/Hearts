# Hearts AI

LSTM + PPO agent for the card game Hearts. The project includes a full four-player game engine, an expert rule policy for imitation learning, a play network, and a separate passing network.

## Current Training Setup

The default PPO configuration is intentionally conservative:

- The play network is fine-tuned with PPO.
- Player 0 uses expert passing during PPO by default.
- The passing network is pretrained, but PPO passing updates are disabled unless `PPO_TRAIN_PASSING` is enabled and `PPO_USE_EXPERT_PASSING` is disabled.
- Checkpoint selection uses multi-seed evaluation to reduce overfitting to one fixed deal stream.

## Architecture

### Play Network: `HeartsLSTM`

Input feature size: `375`

- Hand cards: 52
- Current trick table cards: 52
- My played history: 52
- Other players' played history: 52
- Scores: 4
- Current trick info: 6
- Danger card status: 3
- My suit counts: 4
- Void inference: 16
- Passed cards: 52
- Received cards: 52
- Pass direction: 4
- Opponent suit count estimates: 12
- SQ location probability: 4
- Current trick winner prediction: 4
- Game progress: 1
- Remaining cards per suit: 4
- Points at risk: 1

Network:

```text
Input(375) -> Embed(512) -> LSTM(512, 2 layers) -> Policy(52) + Value(1) + AuxHeads
```

### Passing Network

Input feature size: `108`

- Hand cards: 52
- Already selected pass cards: 52
- Pass direction: 4

Network:

```text
Input(108) -> MLP(256, 3 layers) -> CardScores(52) + Value(1)
```

The passing agent selects three cards sequentially so each choice can depend on previous choices.

## Training Flow

### 1. Joint Pretraining

```bash
python main.py pretrain
```

Uses DAgger-style imitation learning from `ExpertPolicy` to initialize both play and passing behavior.

### 2. PPO Fine-Tuning

```bash
python main.py train
```

Runs PPO for the play network. By default, passing remains expert-controlled for player 0 during PPO so the play policy can be tuned without simultaneously moving the passing distribution.

### 3. Evaluation

```bash
python main.py eval
```

Evaluates against expert and random opponents across the configured eval seeds.

## Important Files

- `config.py`: hyperparameters, paths, eval settings
- `game.py`: Hearts game engine
- `strategies.py`: expert and random policies
- `model.py`: `HeartsLSTM`
- `passing_model.py`: passing network
- `agent.py`: play agent and feature preprocessing
- `passing_agent.py`: passing agent
- `pretrain_joint.py`: joint imitation pretraining
- `train_joint.py`: PPO training and evaluation
- `plot_training.py`: training curve visualization
- `output/`: checkpoints and logs

## Useful Metrics

Lower score is better in Hearts. The most important evaluation field is:

```text
vs_expert_avg_score
```

Training logs also include:

- `play_entropy`: low values mean the policy is nearly deterministic
- `play_clip_fraction`: PPO update clipping rate
- `play_explained_variance`: critic quality; near zero means weak value estimates
- `eval.seed_results`: per-seed eval summaries
- `training_config`: copied run settings for reproducibility

## Dependencies

```text
torch >= 2.0
numpy
matplotlib
tqdm
```

## TODO

- Add a Web UI or replay viewer
- Support human-vs-agent games
- Try a Transformer encoder as an alternative to LSTM
