import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

DATA_PATH = os.path.join(DATA_DIR, "positive_negative_feedback.csv")

# TF-IDF params
TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "max_features": 20000,
    "sublinear_tf": True,
    "min_df": 2,
}

# Train/Test split
TEST_SIZE = 0.2
RANDOM_STATE = 42