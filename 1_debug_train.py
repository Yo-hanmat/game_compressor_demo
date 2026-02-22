import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --- CONFIGURATION ---
MODEL_SAVE_PATH = "compressor_ai.h5"
IMG_SIZE = 128

print("--- STARTING AI TRAINER ---")

def create_fake_data():
    """ Creates simple dummy images if internet fails """
    print("⚠️ Internet download skipped. Generating synthetic training data...")
    data = []
    labels = []
    
    # Create 20 fake images (10 simple black squares, 10 random noise)
    for i in range(10):
        # 1. Simple Image (Black/White) -> Low Complexity
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        cv2.rectangle(img, (10, 10), (50, 50), (255, 255, 255), -1)
        data.append(img / 255.0)
        labels.append(10) # Low score
        
        # 2. Complex Image (Random Noise) -> High Complexity
        noise = np.random.randint(0, 256, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        data.append(noise / 255.0)
        labels.append(90) # High score
        
    return np.array(data), np.array(labels)

def train():
    # 1. Get Data
    print("📊 Preparing Data...")
    X, y = create_fake_data()
    print(f"✅ Data Ready: {len(X)} images created.")

    # 2. Build Brain
    print("🧠 Building Neural Network...")
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # 3. Train
    print("🚀 Training starting... (This takes 5 seconds)")
    model.fit(X, y, epochs=5, verbose=1)
    
    # 4. Save
    model.save(MODEL_SAVE_PATH)
    print("-" * 30)
    print(f"🎉 SUCCESS! The file '{MODEL_SAVE_PATH}' has been created.")
    print(f"📂 Look in this folder: {os.getcwd()}")
    print("-" * 30)

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
    
    # KEEP WINDOW OPEN
    input("\nPRESS ENTER TO EXIT...")