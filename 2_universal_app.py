import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import cv2
import numpy as np
import tensorflow as tf
import threading
import shutil
import open3d as o3d

# --- CONFIGURATION ---
MODEL_PATH = "compressor_ai.h5"

class UniversalCompressor:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuralCompress: Batch Optimizer")
        self.root.geometry("800x600")
        self.root.configure(bg="#2d3436")

        # --- HEADER ---
        header = tk.Frame(root, bg="#2d3436")
        header.pack(pady=20)
        tk.Label(header, text="NEURAL COMPRESSOR", font=("Impact", 28), bg="#2d3436", fg="#00cec9").pack()
        tk.Label(header, text="Automated AI Asset Optimization Pipeline", font=("Arial", 10), bg="#2d3436", fg="#dfe6e9").pack()

        # --- SELECTION AREA ---
        self.frame_select = tk.Frame(root, bg="#353b48", bd=1, relief=tk.RAISED)
        self.frame_select.pack(pady=20, padx=40, fill=tk.X)

        tk.Label(self.frame_select, text="TARGET DIRECTORY", font=("Arial", 9, "bold"), bg="#353b48", fg="#f1c40f").pack(pady=5)
        
        self.btn_select = tk.Button(self.frame_select, text="📂 Select Input Folder", command=self.select_folder, font=("Arial", 12), bg="#0984e3", fg="white", width=25)
        self.btn_select.pack(pady=5)
        
        self.lbl_path = tk.Label(self.frame_select, text="No folder selected", fg="#b2bec3", bg="#353b48")
        self.lbl_path.pack(pady=10)

        # --- RUN BUTTON ---
        self.btn_run = tk.Button(root, text="🚀 START OPTIMIZATION", command=self.start_thread, font=("Arial", 14, "bold"), bg="#00b894", fg="white", state=tk.DISABLED, width=30)
        self.btn_run.pack(pady=10)

        # --- LOG WINDOW ---
        self.log = scrolledtext.ScrolledText(root, width=90, height=15, bg="#000000", fg="#55efc4", font=("Consolas", 10))
        self.log.pack(padx=20, pady=10)

        # Variables
        self.target_folder = None
        self.output_folder = None
        self.model = None

        # Load Model on Startup
        self.load_model()

    def load_model(self):
        self.log_msg("--- SYSTEM INITIALIZATION ---")
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            self.log_msg("✅ AI Model Loaded. Ready to process.")
        except:
            self.log_msg("❌ Error: 'compressor_ai.h5' not found.")
            self.log_msg("Please run Script 1 to train the AI first.")

    def log_msg(self, msg):
        self.log.insert(tk.END, ">> " + msg + "\n")
        self.log.see(tk.END)

    def select_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.target_folder = path
            # Set Output Folder Name (e.g. "TestBox_Optimized")
            parent_dir = os.path.dirname(path)
            folder_name = os.path.basename(path)
            self.output_folder = os.path.join(parent_dir, folder_name + "_Optimized")
            
            self.lbl_path.config(text=f"Input: {path}\nOutput: {self.output_folder}")
            self.btn_run.config(state=tk.NORMAL)
            self.log_msg(f"Selected Target: {folder_name}")

    def start_thread(self):
        t = threading.Thread(target=self.run_process)
        t.start()

    # --- AI ENGINE ---
    def get_ai_score(self, image_input):
        # Resize to 128x128 for the Brain
        if image_input.shape[0] != 128 or image_input.shape[1] != 128:
            image_input = cv2.resize(image_input, (128, 128))
        
        img_array = image_input / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = self.model.predict(img_array, verbose=0)[0][0]
        # Result is a number between 10 (Compress hard) and 95 (Keep quality)
        return max(10, min(75, int(prediction*0.8)))

    # --- SAVE HELPER ---
    def get_output_path(self, full_input_path):
        rel_path = os.path.relpath(full_input_path, self.target_folder)
        save_path = os.path.join(self.output_folder, rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path

    # --- OPTIMIZERS ---
    def optimize_2d(self, path):
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None: return 0

            # Prepare for AI
            if len(img.shape) > 2 and img.shape[2] == 4:
                ai_input = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                ai_input = img
            
            quality = self.get_ai_score(ai_input)
            save_path = self.get_output_path(path)

            # Compress based on File Type
            if path.lower().endswith(('.jpg', '.jpeg')):
                cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif path.lower().endswith('.png'):
                comp = int((100 - quality) / 10)
                comp = max(0, min(9, comp))
                cv2.imwrite(save_path, img, [cv2.IMWRITE_PNG_COMPRESSION, comp])

            return os.path.getsize(path) - os.path.getsize(save_path)
        except: return 0

    def optimize_3d(self, path):
        try:
            # 1. Load the heavy mesh
            self.log_msg(f" Reading heavy 3D data: {os.path.basename(path)}...")
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
            mesh = o3d.io.read_triangle_mesh(path)
            
            if not mesh.has_triangles() or len(mesh.triangles) < 100: return 0
            
            # 2. ULTRA-FAST SHADOW MAPPING (Vectorized NumPy)
            verts = np.asarray(mesh.vertices)
            min_v, max_v = verts.min(0), verts.max(0)
            range_v = max_v - min_v
            range_v[range_v == 0] = 1 
            norm_verts = (verts - min_v) / range_v
            
            # Create a black 128x128 canvas
            shadow = np.zeros((128, 128, 3), dtype=np.uint8)
            
            # FAST MATH: Scale vertices to 0-127 and map them directly to pixels
            # This replaces the slow for-loop and runs instantly.
            scaled_verts = (norm_verts * 127).astype(np.int32)
            shadow[scaled_verts[:, 1], scaled_verts[:, 0]] = 255
                
            # 3. AI Analysis
            ai_score = self.get_ai_score(shadow)
            target_ratio = max(0.1, ai_score / 200.0) # Aggressive demo compression
            target_count = int(len(mesh.triangles) * target_ratio)
            
            # 4. Decimate & Save
            self.log_msg(f"Crunching geometry... (This may take a moment)")
            mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=target_count)
            
            save_path = self.get_output_path(path)
            o3d.io.write_triangle_mesh(save_path, mesh_smp)
            
            # 5. FIX THE MATH BUG (Calculate actual bytes saved)
            original_size = os.path.getsize(path)
            new_size = os.path.getsize(save_path)
            
            self.log_msg(f"Done: {os.path.basename(path)}")
            return original_size - new_size # Returns correct bytes saved!
            
        except Exception as e: 
            return 0

    # --- MAIN LOOP ---
    # --- MAIN LOOP ---
    def run_process(self):
        self.btn_run.config(state=tk.DISABLED)
        self.log_msg("--- STARTING OPERATION ---")
        self.log_msg(f"Source: {os.path.basename(self.target_folder)}")
        
        # 1. Clone Folder Structure
        self.log_msg("Creating optimized directory...")
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder) # Clean previous run
        shutil.copytree(self.target_folder, self.output_folder, ignore=shutil.ignore_patterns('*.jpg', '*.png', '*.obj'), dirs_exist_ok=True)
        
        # 2. Process Files
        count_2d = 0
        count_3d = 0
        total_saved_bytes = 0
        
        for root, _, files in os.walk(self.target_folder):
            for file in files:
                full_path = os.path.join(root, file)
                ext = file.lower()
                
                if ext.endswith(('.jpg', '.jpeg', '.png', '.tga')):
                    saved = self.optimize_2d(full_path)
                    if saved > 0:
                        count_2d += 1
                        total_saved_bytes += saved  # <-- Adds 2D savings
                        if count_2d % 5 == 0: self.log_msg(f"Processed {count_2d} images...")
                
                elif ext.endswith(('.obj', '.ply', '.gltf', '.glb')):
                    saved = self.optimize_3d(full_path)
                    if saved > 0:
                        count_3d += 1
                        total_saved_bytes += saved  # <-- THIS WAS MISSING! Adds 3D savings
                        self.log_msg(f"Optimized Model: {file}")

        # 3. Final Report
        total_mb = total_saved_bytes / (1024 * 1024)
        self.log_msg("=" * 30)
        self.log_msg("🎉 OPERATION COMPLETE")
        self.log_msg(f"Images Optimized: {count_2d}")
        self.log_msg(f"Models Optimized: {count_3d}")
        self.log_msg(f"Total Space Saved: {total_mb:.2f} MB")
        self.log_msg(f"Output Folder: {self.output_folder}")
        
        messagebox.showinfo("Done", f"Success!\nSaved {total_mb:.2f} MB.\n\nGo check the folder: {os.path.basename(self.output_folder)}")
        self.btn_run.config(state=tk.NORMAL)

        # 3. Final Report
        total_mb = total_saved_bytes / (1024 * 1024)
        self.log_msg("=" * 30)
        self.log_msg(" OPERATION COMPLETE")
        self.log_msg(f"Images Optimized: {count_2d}")
        self.log_msg(f"Models Optimized: {count_3d}")
        self.log_msg(f"Total Space Saved: {total_mb:.2f} MB")
        self.log_msg(f"Output Folder: {self.output_folder}")
        
        messagebox.showinfo("Done", f"Success!\nSaved {total_mb:.2f} MB.\n\nGo check the folder: {os.path.basename(self.output_folder)}")
        self.btn_run.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = UniversalCompressor(root)
    root.mainloop()
