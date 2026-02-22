import os
import requests
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --- CONFIGURATION ---
DATASET_DIR = "training_data"
MODEL_SAVE_PATH = "compressor_ai.h5"
IMG_SIZE = 128

# --- 1. DOWNLOAD SAMPLE IMAGES (AUTOMATIC) ---
def download_samples():
    """
    Downloads a few images from the internet to use as examples.
    We need a mix of simple (cartoons) and complex (photos) images
    so the AI can learn the difference.
    """
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        print("⬇️ Downloading sample images for training...")
        
        # We use direct links to Wikimedia Commons images
        urls = [
            "https://upload.wikimedia.org/wikipedia/commons/e/e0/Synthese_.svg", # Simple Shape
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg", # Complex Painting
            "https://upload.wikimedia.org/wikipedia/commons/8/89/Tomato_je.jpg", # Medium Detail
            "https://upload.wikimedia.org/wikipedia/commons/9/9a/Gull_portrait_ca_usa.jpg", # High Detail Bird
            "https://upload.wikimedia.org/wikipedia/commons/b/b4/The_Sun_by_the_Atmospheric_Imaging_Assembly_of_NASA%27s_Solar_Dynamics_Observatory_-_20100819.jpg" # Complex Texture
        ]
        
        for i, url in enumerate(urls):
            try:
                img_data = requests.get(url).content
                with open(f"{DATASET_DIR}/sample_{i}.jpg", 'wb') as handler:
                    handler.write(img_data)
                print(f"   - Downloaded sample_{i}.jpg")
            except:
                print(f"   - Failed to download {url}")
                
    print("✅ Training Data Ready.")

# --- 2. PREPARE DATA ---
def create_dataset():
    """
    Reads the images and assigns them a 'Score' based on complexity.
    This creates the answer key for the AI to study.
    """
    data = []
    labels = []
    
    print("🧠 Analyzing image complexity...")
    for file in os.listdir(DATASET_DIR):
        if file.endswith(".jpg"):
            path = os.path.join(DATASET_DIR, file)
            img = cv2.imread(path)
            if img is None: continue
            
            # Resize image to 128x128 so the AI can read it easily
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
            data.append(img_resized)
            
            # MATH TRICK: Edge Detection
            # We use Canny Edge Detection to count how many 'lines' are in the picture.
            # More lines = More Complex = Needs Higher Quality.
            edges = cv2.Canny(img, 100, 200)
            score = np.mean(edges) # Gives a number (e.g., 20 for simple, 150 for complex)
            
            # Normalize score to be between 10 and 95 (for JPEG quality)
            normalized_score = min(95, max(10, (score / 50) * 100))
            labels.append(normalized_score)

    return np.array(data), np.array(labels)

# --- 3. TRAIN MODEL ---
def train():
    download_samples()
    X, y = create_dataset()
    
    if len(X) == 0:
        print("❌ No data found. Check internet connection.")
        return

    print("🚀 Training Neural Network...")
    # This is the 'Brain' structure
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1) # The output is just ONE number: The Quality Score
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train for 20 cycles (epochs)
    model.fit(X, y, epochs=20, verbose=1)
    
    # Save the learned brain to a file
    model.save(MODEL_SAVE_PATH)
    print("-" * 30)
    print(f"🎉 SUCCESS: AI Brain saved to '{MODEL_SAVE_PATH}'")
    print("You can now run Script 2 to use this brain!")

if __name__ == "__main__":
    train()