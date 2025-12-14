import os

BASE_DIR = "D:/coding/data/birds_common/"

XENO_RUSSIA_PATH = os.path.join(BASE_DIR, 'input', 'xeno_russia')
BIRDCLEF_RUSSIA_PATH = os.path.join(BASE_DIR, 'input', 'birdclef_russia')
MERGED_DATASET_PATH = os.path.join(BASE_DIR, 'input', 'russian_dataset')

DATA_ROOT = os.path.join(BASE_DIR, 'data_russian')
RESAMPLED_DIR = os.path.join(DATA_ROOT, 'resampled')
MEL_DIR = os.path.join(DATA_ROOT, 'mel')
SPLITS_DIR = os.path.join(DATA_ROOT, 'splits')
EMBEDDINGS_DIR = os.path.join(DATA_ROOT, 'embeddings')
CHECKPOINTS_DIR = os.path.join(DATA_ROOT, 'checkpoints')
RESULTS_DIR = os.path.join(DATA_ROOT, 'results')

MANIFEST_PATH = os.path.join(DATA_ROOT, 'manifest.csv')

TARGET_SR = 32000
SEGMENT_DURATION = 5.0
HOP_DURATION = 2.5

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 50
FMAX = 14000

MIN_SNR_DB = 10.0

TOP_N_KNOWN = 30
MIN_SEGMENTS_KNOWN = 20

UNSEEN_COUNT = 10
MIN_SEGMENTS_UNSEEN = 10
MAX_SEGMENTS_UNSEEN = 20

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

LOGREG_C = 1.0
LOGREG_MAX_ITER = 1000

MLP_HIDDEN_DIM = 256
MLP_LR = 1e-3
MLP_EPOCHS = 50
MLP_BATCH_SIZE = 64

CNN_LR = 1e-4
CNN_EPOCHS = 30
CNN_BATCH_SIZE = 32
