import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox
import open3d as o3d
import shutil
from skimage.metrics import structural_similarity as ssim
from openpyxl import Workbook


class UniversalOptimizer:

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.classifier = load_model("classifier_model.h5")
        self.CLASSES = ["character", "environment", "texture", "object"]
        self.total_saved = 0

        # Excel data
        self.report_data = []

    # -----------------------------
    def classify_image(self, img):
        img = cv2.resize(img, (128, 128)) / 255.0
        img = np.expand_dims(img, axis=0)
        pred = self.classifier.predict(img)[0]
        return self.CLASSES[np.argmax(pred)]

    # -----------------------------
    def get_output_path(self, path):
        rel = os.path.relpath(path, self.input_dir)
        save_path = os.path.join(self.output_dir, rel)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path

    # -----------------------------
    # SSIM CALCULATION
    # -----------------------------
    def calculate_ssim(self, original, compressed):
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        compressed_gray = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(original_gray, compressed_gray, full=True)
        return score

    # -----------------------------
    # 2D OPTIMIZATION
    # -----------------------------
    def optimize_2d(self, path, log):

        original_size = os.path.getsize(path)
        original_img = cv2.imread(path)

        if original_img is None:
            return

        cls = self.classify_image(original_img)

        img = original_img.copy()

        if cls == "character":
            quality = 92
        elif cls == "object":
            quality = 50
        else:
            small = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
            img = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            quality = 5

        save_path = self.get_output_path(path)
        ext = os.path.splitext(path)[1].lower()

        if ext in [".jpg", ".jpeg"]:
            cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif ext == ".png":
            comp = int((100 - quality) / 10)
            cv2.imwrite(save_path, img, [cv2.IMWRITE_PNG_COMPRESSION, comp])

        compressed_img = cv2.imread(save_path)
        new_size = os.path.getsize(save_path)

        # Prevent size increase for character
        if cls == "character" and new_size > original_size:
            os.remove(save_path)
            shutil.copy2(path, save_path)
            compressed_img = original_img
            new_size = original_size

        saved = original_size - new_size
        self.total_saved += saved

        # SSIM
        ssim_score = self.calculate_ssim(original_img, compressed_img)

        # Compression ratio
        ratio = original_size / new_size if new_size != 0 else 0

        # Save report data
        self.report_data.append([
            os.path.basename(path),
            original_size,
            new_size,
            ratio,
            ssim_score,
            "-"
        ])

        log(f"2D → {cls}: {os.path.basename(path)} | SSIM: {ssim_score:.3f}")

    # -----------------------------
    # 3D OPTIMIZATION
    # -----------------------------
    def optimize_3d(self, path, log):

        original_size = os.path.getsize(path)
        save_path = self.get_output_path(path)
        ext = os.path.splitext(path)[1].lower()

        vertices_before = 0
        vertices_after = 0

        if ext == ".obj":
            mesh = o3d.io.read_triangle_mesh(path)

            if mesh.has_triangles():
                vertices_before = len(np.asarray(mesh.vertices))

                simplified = mesh.simplify_quadric_decimation(
                    int(len(mesh.triangles) * 0.5)
                )

                simplified.compute_vertex_normals()

                vertices_after = len(np.asarray(simplified.vertices))

                o3d.io.write_triangle_mesh(save_path, simplified)
            else:
                shutil.copy2(path, save_path)

        else:
            shutil.copy2(path, save_path)

        new_size = os.path.getsize(save_path)
        saved = original_size - new_size
        self.total_saved += saved

        # Compression ratio
        ratio = original_size / new_size if new_size != 0 else 0

        vertex_reduction = vertices_before - vertices_after

        # Save report
        self.report_data.append([
            os.path.basename(path),
            original_size,
            new_size,
            ratio,
            "-",
            vertex_reduction
        ])

        log(f"3D → {os.path.basename(path)} | Vertices reduced: {vertex_reduction}")

    # -----------------------------
    # CREATE EXCEL REPORT
    # -----------------------------
    def generate_report(self):

        wb = Workbook()
        ws = wb.active
        ws.title = "Compression Report"

        headers = [
            "File Name",
            "Original Size (bytes)",
            "Compressed Size (bytes)",
            "Compression Ratio",
            "SSIM (2D)",
            "Vertices Reduced (3D)"
        ]

        ws.append(headers)

        for row in self.report_data:
            ws.append(row)

        report_path = os.path.join(self.output_dir, "compression_report.xlsx")
        wb.save(report_path)

        return report_path

    # -----------------------------
    def process(self, log):

        for root, _, files in os.walk(self.input_dir):
            for file in files:
                path = os.path.join(root, file)

                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.optimize_2d(path, log)

                elif file.lower().endswith((".obj", ".fbx", ".glb")):
                    self.optimize_3d(path, log)

        report = self.generate_report()
        return self.total_saved, report


# -----------------------------
# GUI
# -----------------------------
class App:

    def __init__(self, root):
        self.root = root
        self.root.title("AI Asset Optimizer")
        self.root.geometry("650x450")
        self.root.configure(bg="#391e10")

        self.input_folder = ""

        tk.Label(root, text="AI Asset Optimizer",
                 font=("Arial", 18, "bold"),
                 bg="#391e10", fg="white").pack(pady=10)

        tk.Button(root, text="Select Folder",
                  command=self.select_folder,
                  bg="#c7a07a", fg="#391e10").pack(pady=5)

        tk.Button(root, text="Start Optimization",
                  command=self.start,
                  bg="#c7a07a", fg="#391e10").pack(pady=5)

        self.log_box = tk.Text(root, bg="#e2ceb1", fg="#000000")
        self.log_box.pack(fill="both", expand=True, padx=10, pady=10)

        self.result_label = tk.Label(root, text="Total Saved: 0 KB",
                                     bg="#391e10", fg="white",
                                     font=("Arial", 12, "bold"))
        self.result_label.pack(pady=5)

    def log(self, msg):
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)
        self.root.update()

    def select_folder(self):
        self.input_folder = filedialog.askdirectory()
        self.log(f"Selected: {self.input_folder}")

    def start(self):

        if not self.input_folder:
            messagebox.showerror("Error", "Select folder first")
            return

        output_folder = "Test_Box_Optimized"

        optimizer = UniversalOptimizer(self.input_folder, output_folder)

        self.log("Starting...\n")

        total_saved, report_path = optimizer.process(self.log)

        self.result_label.config(
            text=f"Total Saved: {total_saved/1024/1024:.2f} MB"
        )

        self.log(f"\nReport saved: {report_path}")
        self.log("\n Done")

        messagebox.showinfo("Done", "Optimization + Report Complete!")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()