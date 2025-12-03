# Hyperparameters & Configuration

# --- PPO Training ---
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPISODES = 5000 
BATCH_SIZE = 32 
PPO_EPOCHS = 10 
CLIP_EPS = 0.2
ENTROPY_COEF_START = 0.03 
ENTROPY_COEF_END = 0.005 
VALUE_LOSS_COEF = 0.5 
MAX_GRAD_NORM = 0.5 

# --- Resource Management ---
THROTTLE_TIME = 0.05 

# --- Model Architecture ---
HIDDEN_DIM = 256 
DROPOUT = 0.1 
WEIGHT_DECAY = 1e-5 

# --- Supervised Pretraining ---
PRETRAIN_LR = 1e-4 
PRETRAIN_EPISODES = 5000
PRETRAIN_BATCH_SIZE = 32 
PRETRAIN_EPOCHS = 10 
LABEL_SMOOTHING = 0.0 

# --- DAgger (Dataset Aggregation) ---
DAGGER_BETA_START = 1.0 
DAGGER_BETA_DECAY = 0.9998 
DAGGER_BETA_MIN = 0.3 

# --- Curriculum Learning ---
CURRICULUM_SCORE_THRESHOLD = 6.0 
CURRICULUM_STABILITY_WINDOW = 500 
MAX_STAGE_EPISODES = 3000 
POOL_STAGE_DURATION = 2000

# --- Paths ---
import os
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

MODEL_PATH = os.path.join('saved_models', 'hearts_model.pth')
PRETRAINED_MODEL_PATH = os.path.join('saved_models', 'hearts_model_pretrained.pth')
BEST_MODEL_PATH = os.path.join('saved_models', 'hearts_model_best.pth')
LOG_FILE = 'training_log.json'
