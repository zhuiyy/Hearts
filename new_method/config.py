import os

# Configuration for SOTA Hearts AI

# Model
HIDDEN_DIM = 512
LSTM_HIDDEN = 128
DROPOUT = 0.1

# Training
LR = 5e-5
GAMMA = 0.99
BATCH_SIZE = 128
PPO_EPOCHS = 4
PRETRAIN_EPOCHS = 4
CLIP_EPS = 0.2
TOTAL_EPISODES = 100000

# Paths
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

MODEL_PATH = os.path.join(OUTPUT_DIR, "sota_hearts_model.pth")
PRETRAINED_MODEL_PATH = os.path.join(OUTPUT_DIR, "sota_hearts_pretrained.pth")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "sota_hearts_best.pth")
LOG_FILE = os.path.join(OUTPUT_DIR, "training_log.json")
PRETRAIN_LOG_FILE = os.path.join(OUTPUT_DIR, "pretrain_log.json")

# PIMC
PIMC_SIMULATIONS = 20
PIMC_TOP_K = 3
