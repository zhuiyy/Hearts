import os

# Configuration for Simple FCN Hearts AI

# Model
HIDDEN_DIM = 512
# Old Input: 176
# New Enhanced Input Feature Spec:
# 1. Hand (52)
# 2. Table Cards (Current Trick) (52)
# 3. History (Played by Me) (52) - To know what I spent
# 4. History (Played by Others) (52) - What is gone
# 5. Scores (4)
# 6. Current Trick Info (6): 
#    - [Has Hearts?, Has SQ?, My Rank High?, Others Rank High?, Count, Is Lead?]
# 7. Danger Cards Status (3): [SQ Out?, SK Out?, SA Out?]
# 8. Suit Counts (My Hand) (4)
# 9. Void Inference (Others) (4 players * 4 suits = 16) - Deduced history
# 10. Passed Cards (52) - 我传出去的牌
# 11. Received Cards (52) - 我收到的牌  
# 12. Pass Direction (4) - 传牌方向 (LEFT/RIGHT/ACROSS/KEEP)
# === 新增对手建模特征 ===
# 13. Opponent Suit Counts Estimate (3 opponents * 4 suits = 12) - 估计对手各花色剩余牌数
# 14. SQ Location Probability (4) - SQ在各玩家手中的概率
# 15. Current Trick Winner Prediction (4) - 当前trick谁会赢
# 16. Game Progress (1) - 第几轮/13
# 17. Remaining Cards Per Suit (4) - 各花色剩余未出牌数
# 18. Points at Risk (1) - 当前trick有多少分
# Total: 349 + 12 + 4 + 4 + 1 + 4 + 1 = 375
INPUT_DIM = 375
DROPOUT = 0.1

# Training
SEED = 42
LR = 5e-5  # Conservative PPO fine-tuning from the pretrained policy
GAMMA = 0.99
GAE_LAMBDA = 0.95
BATCH_SIZE = 512
PPO_EPOCHS = 2
ENTROPY_COEF = 0.005  # Keep exploration modest so PPO does not drift from Expert behavior
CLIP_EPS = 0.1
VALUE_COEF = 0.5  # Value loss coefficient
AUX_COEF = 0.3  # Auxiliary task coefficient
BC_ANCHOR_COEF = 0.2  # Keep PPO close to Expert play actions during early fine-tuning
PPO_TRAIN_PASSING = False
PPO_USE_EXPERT_PASSING = True
PPO_PROGRESS_INTERVAL = 25
PPO_EVAL_INTERVAL = 2500
TOTAL_EPISODES = 30000  # Controlled PPO run; stop/extend based on fixed-seed eval trend
EVAL_GAMES = 200
PPO_MAX_MINUTES = 150  # Hard wall-clock guard to avoid accidental all-day runs
PPO_MAX_MINUTES = 60  # Hard wall-clock guard to avoid long stuck runs
PPO_EARLY_STOP_PATIENCE_EVALS = 4  # Stop if N evals do not beat best

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
