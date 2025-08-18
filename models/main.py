# from models.train_logreg import main as logreg_main
# from models.train_lightgbm import main as lgbm_main
from models.train_catboost import main as cat_main
# from models.train_lstm import main as lstm_main
# from models.train_tabtransformer import main as tabtr_main
# from models.train_tft import main as tft_main

DATA_DIR = "/Users/nikita/Documents/final_project_data/sentiment_features_cleaned"
TEST_TICKERS = ["TSLA", "APPL"]  # or None
TARGET_COL = "y_5d"

# classical
# logreg_main(DATA_DIR, test_tickers=TEST_TICKERS, target_col=TARGET_COL)
# lgbm_main(DATA_DIR, test_tickers=TEST_TICKERS, target_col=TARGET_COL)
cat_main(DATA_DIR,  test_tickers=TEST_TICKERS, target_col=TARGET_COL)

# deep (sequence)
# lstm_main(DATA_DIR, test_tickers=TEST_TICKERS, target_col=TARGET_COL, window=30, epochs=10, bs=256)
# tabtr_main(DATA_DIR, test_tickers=TEST_TICKERS, target_col=TARGET_COL, window=30,
#            d_model=64, layers=4, heads=4, dropout=0.2, epochs=12, bs=256)
# tft_main(DATA_DIR,  test_tickers=TEST_TICKERS, target_col=TARGET_COL, window=30, epochs=10, bs=256)
