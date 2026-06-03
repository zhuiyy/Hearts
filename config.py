import os

# Configuration for Hearts AI

# Model
HIDDEN_DIM = 512
DROPOUT = 0.1

# Enhanced input feature spec:
# 1. Hand (52)
# 2. Table Cards, current trick (52)
# 3. History, played by me (52)
# 4. History, played by others (52)
# 5. Scores (4)
# 6. Current Trick Info (6):
#    [Has Hearts?, Has SQ?, My Rank High?, Current Winner Strength, Count, Is Lead?]
# 7. Danger Cards Status (3): [SQ Out?, SK Out?, SA Out?]
# 8. Suit Counts, my hand (4)
# 9. Void Inference, all players by suit (4 players * 4 suits = 16)
# 10. Passed Cards (52)
# 11. Received Cards (52)
# 12. Pass Direction (4): LEFT/RIGHT/ACROSS/KEEP
# 13. Opponent Suit Counts Estimate (3 opponents * 4 suits = 12)
# 14. SQ Location Probability (4)
# 15. Current Trick Winner Prediction (4)
# 16. Game Progress (1)
# 17. Remaining Cards Per Suit (4)
# 18. Points at Risk (1)
# Total: 349 + 12 + 4 + 4 + 1 + 4 + 1 = 375
INPUT_DIM = 375

# Training
SEED = 42
LR = 5e-5  # Conservative PPO fine-tuning from the pretrained policy
GAMMA = 0.99
GAE_LAMBDA = 0.95
BATCH_SIZE = 512
PPO_EPOCHS = 2
ENTROPY_COEF = 0.005  # Keep exploration modest so PPO does not drift from Expert behavior
CLIP_EPS = 0.1
VALUE_COEF = 0.5
AUX_COEF = 0.3
BC_ANCHOR_COEF = 0.2  # Keep PPO close to Expert play actions during early fine-tuning

# PPO mode. By default, PPO trains only the play network and uses Expert passing for player 0.
PPO_TRAIN_PASSING = False
PPO_USE_EXPERT_PASSING = True

PPO_PROGRESS_INTERVAL = 25
PPO_EVAL_INTERVAL = 2500
TOTAL_EPISODES = 30000
PPO_MAX_MINUTES = 60
PPO_EARLY_STOP_PATIENCE_EVALS = 4

# Evaluation. Training checkpoint selection uses the multi-seed settings below.
EVAL_GAMES = 200
EVAL_SEEDS = [SEED, SEED + 1, SEED + 2]
EVAL_GAMES_PER_SEED = 100

# Paths
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

PPO_SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, "ppo_snapshots")
if not os.path.exists(PPO_SNAPSHOT_DIR):
    os.makedirs(PPO_SNAPSHOT_DIR)

MODEL_PATH = os.path.join(OUTPUT_DIR, "simple_fcn_model.pth")
PRETRAINED_MODEL_PATH = os.path.join(OUTPUT_DIR, "simple_fcn_pretrained.pth")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "simple_fcn_model_best.pth")
PASSING_MODEL_PATH = os.path.join(OUTPUT_DIR, "passing_model.pth")
PASSING_PRETRAINED_PATH = os.path.join(OUTPUT_DIR, "passing_pretrained.pth")
PASSING_BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "passing_model_best.pth")
LOG_FILE = os.path.join(OUTPUT_DIR, "training_log.json")
