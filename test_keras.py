import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

print("Loading Keras model...")
try:
    model = tf.keras.models.load_model("Best_TBC_Model_Competition.keras")
    
    # Empty tensor (all zeros)
    empty = np.zeros((1, 224, 224, 3), dtype=np.float32)
    # The TFJS normalization does (pixels / 127.5) - 1, so black (0) is -1.0
    empty_tfjs_black = np.full((1, 224, 224, 3), -1.0, dtype=np.float32)
    
    pred_empty = model.predict(empty_tfjs_black, verbose=0)
    print(f"Prediction for solid black image (-1.0): {pred_empty[0][0]:.6f}")
    
    # White tensor (+1.0)
    white = np.full((1, 224, 224, 3), 1.0, dtype=np.float32)
    pred_white = model.predict(white, verbose=0)
    print(f"Prediction for solid white image (+1.0): {pred_white[0][0]:.6f}")

    # Random noise
    np.random.seed(42)
    noise = np.random.uniform(-1.0, 1.0, (1, 224, 224, 3)).astype(np.float32)
    pred_noise = model.predict(noise, verbose=0)
    print(f"Prediction for random noise: {pred_noise[0][0]:.6f}")

except Exception as e:
    print("Error:", e)
