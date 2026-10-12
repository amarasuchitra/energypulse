"""
Test prediction pipeline components
"""
import pytest
import numpy as np
import pandas as pd
import pickle
import os

MODEL_DIR = "models"
DATA_DIR = "data"

def test_xgb_model_loads():
    """Test XGBoost model loads and can predict"""
    path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    assert os.path.exists(path), "XGBoost model file missing"
    
    with open(path, "rb") as f:
        model = pickle.load(f)
    
    assert hasattr(model, "predict"), "Model missing predict method"
    
    # Test prediction with dummy features (9 features)
    dummy_features = pd.DataFrame([{
        "hour": 12,
        "day_of_week": 2,
        "is_weekend": 0,
        "Global_reactive_power": 0.1,
        "Voltage": 235.0,
        "Global_intensity": 5.0,
        "Sub_metering_1": 0.0,
        "Sub_metering_2": 0.0,
        "Sub_metering_3": 0.0,
    }])
    
    pred = model.predict(dummy_features)
    assert len(pred) == 1
    assert np.isfinite(pred[0])
    assert pred[0] >= 0  # Power should be non-negative

def test_scaler_loads():
    """Test scaler loads and transforms correctly"""
    path = os.path.join(MODEL_DIR, "lstm_scaler.pkl")
    assert os.path.exists(path), "Scaler file missing"
    
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    
    assert hasattr(scaler, "transform"), "Scaler missing transform method"
    
    # Test with 10 features (the scaler was fit on 10 features)
    dummy_data = np.random.rand(10, 10).astype("float32")
    scaled = scaler.transform(dummy_data)
    assert scaled.shape == (10, 10)

def test_meta_loads():
    """Test meta file loads with expected keys"""
    path = os.path.join(MODEL_DIR, "model_meta.pkl")
    assert os.path.exists(path), "Meta file missing"
    
    with open(path, "rb") as f:
        meta = pickle.load(f)
    
    required_keys = ["xgb_mae", "lstm_mae", "hybrid_mae", "hybrid_weights", 
                     "feature_cols", "target", "window_size"]
    for key in required_keys:
        assert key in meta, f"Missing key in meta: {key}"
    
    # Check weights sum to ~1
    w_xgb, w_lstm = meta["hybrid_weights"]
    assert abs(w_xgb + w_lstm - 1.0) < 0.01

def test_sample_data_exists():
    """Test sample data files exist"""
    assert os.path.exists(os.path.join(DATA_DIR, "cleaned_energy_data.csv"))
    assert os.path.exists(os.path.join(DATA_DIR, "sample_meta.json"))

def test_predict_next_period_runs():
    """Test predict_next_period function runs without error"""
    from model import predict_next_period, FEATURE_COLS, TARGET, WINDOW_SIZE
    
    # Load models
    with open(os.path.join(MODEL_DIR, "xgboost_model.pkl"), "rb") as f:
        xgb_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "lstm_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "model_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    
    # Create dummy recent data (window_size + 10 rows)
    recent_df = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=WINDOW_SIZE + 10, freq="h"),
        "Global_active_power": np.random.uniform(0.5, 3.0, WINDOW_SIZE + 10),
        "Global_reactive_power": np.random.uniform(0.05, 0.5, WINDOW_SIZE + 10),
        "Voltage": np.random.uniform(220, 250, WINDOW_SIZE + 10),
        "Global_intensity": np.random.uniform(2, 10, WINDOW_SIZE + 10),
        "Sub_metering_1": np.random.uniform(0, 1, WINDOW_SIZE + 10),
        "Sub_metering_2": np.random.uniform(0, 1, WINDOW_SIZE + 10),
        "Sub_metering_3": np.random.uniform(0, 1, WINDOW_SIZE + 10),
    })
    recent_df["hour"] = recent_df["datetime"].dt.hour
    recent_df["day_of_week"] = recent_df["datetime"].dt.dayofweek
    recent_df["is_weekend"] = (recent_df["day_of_week"] >= 5).astype(int)
    
    live_row = recent_df.iloc[-1].to_dict()
    
    # Should not raise
    hybrid, xgb_p, lstm_p, info = predict_next_period(
        live_row, xgb_model, None, scaler, recent_df,
        return_info=True, weights=meta.get("hybrid_weights")
    )
    
    assert np.isfinite(hybrid)
    assert hybrid >= 0
    assert info["lstm_used"] == False  # LSTM is None
    assert info["w_xgb"] == 1.0

def test_hybrid_weights_are_sensible():
    """Test that hybrid weights are reasonable"""
    with open(os.path.join(MODEL_DIR, "model_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    
    w_xgb, w_lstm = meta["hybrid_weights"]
    # XGBoost should have higher weight (it's the working model)
    assert w_xgb > w_lstm
    assert w_xgb > 0.5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])