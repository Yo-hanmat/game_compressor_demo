# AI-Based Game Asset Optimization System

## 1. Project Overview

This project is an AI-based Game Asset Optimization System designed to reduce the storage requirements of 2D and 3D game assets while applying different levels of optimization depending on the importance and type of the asset.

The system accepts a folder containing game assets and automatically processes the files. For 2D assets, a trained deep learning image classifier identifies the type of asset and selects an appropriate compression strategy. Environment and texture assets are compressed aggressively, objects receive moderate compression, and character assets receive very light compression to preserve their visual quality.

For 3D assets, the system performs mesh simplification using Quadric Error Metric based decimation. This reduces the geometric complexity of supported 3D models by reducing vertices and triangles while attempting to preserve the overall shape of the model.

The system also calculates evaluation metrics such as SSIM for 2D images, vertex reduction for 3D models, original file size, compressed file size, and compression ratio. These results are automatically stored in an Excel report.

A graphical user interface (GUI) is provided so that the user can select a folder and start the optimization process without manually specifying individual files.

---

## 2. Main Objectives

- Develop an intelligent game asset optimization system.
- Classify 2D game assets using a trained deep learning model.
- Automatically identify asset categories such as:
  - Character
  - Environment
  - Texture
  - Object
- Apply adaptive compression according to the predicted asset category.
- Apply maximum compression and pixelation to environment and texture assets.
- Apply moderate compression to object assets.
- Apply very light compression to character assets while preventing unnecessary file-size increase.
- Reduce the geometric complexity of supported 3D models.
- Reduce vertices and triangles in 3D meshes using mesh decimation.
- Measure 2D image quality using Structural Similarity Index (SSIM).
- Measure 3D optimization using vertex reduction.
- Calculate compression ratio and storage savings.
- Generate an Excel report containing optimization results.
- Provide a simple GUI for selecting and optimizing a complete asset folder.

---

## 3. System Workflow

The overall workflow of the project is:

    User
      |
      v
    GUI
      |
      v
    Select Input Folder
      |
      v
    Scan All Files
      |
      v
    File Type Detection
      |
      +---------------------------+
      |                           |
      v                           v
    2D Image                    3D Model
      |                           |
      v                           v
    Image Loading              Mesh Loading
      |                           |
      v                           v
    Preprocessing              Mesh Analysis
      |                           |
      v                           v
    Deep Learning              Quadric Decimation
    Classification                |
      |                           v
      v                       Vertex/Face Reduction
    Asset Class                    |
      |                           v
      v                       Save Optimized Model
    Adaptive Compression           |
      |                           v
      v                       Vertex Reduction
    SSIM Calculation
      |
      v
    Save Optimized Image
      |
      +-------------+-------------+
                    |
                    v
             Data Analysis
                    |
                    v
        Size & Compression Ratio
                    |
                    v
             Excel Report
                    |
                    v
          Test_Box_Optimized
                    |
                    v
                  GUI
                    |
                    v
                 Results

---

## 4. 2D Asset Optimization

The 2D optimization module uses a trained deep learning classifier.

The classifier predicts one of the following categories:

    character
    environment
    texture
    object

The predicted class determines the compression level.

### Character

Characters are visually important game assets. Therefore, the system applies very light compression.

The objective is to maintain the original visual appearance and avoid increasing the file size.

### Object

Objects receive moderate compression.

Examples include:

- Weapons
- Vehicles
- Furniture
- Props
- Tools
- Containers
- Collectible items

### Environment

Environment assets are suitable for aggressive optimization.

Examples include:

- Grass
- Trees
- Rocks
- Mountains
- Buildings
- Walls
- Roads
- Ground
- Terrain
- Background elements

### Texture

Textures are also optimized aggressively.

Examples include:

- Grass textures
- Ground textures
- Brick textures
- Stone textures
- Wood textures
- Metal textures
- Wall textures
- Terrain textures

For environment and texture assets, the system performs downscaling followed by nearest-neighbor upscaling. This intentionally introduces pixelation and allows stronger reduction of visual detail.

---

## 5. 3D Asset Optimization

The 3D module is independent of the 2D deep learning classifier.

The 3D pipeline analyzes the mesh and applies geometry simplification.

The optimization process includes:

1. Loading the 3D model.
2. Reading the mesh geometry.
3. Counting the original vertices.
4. Counting the original triangles/faces.
5. Applying Quadric Error Metric based mesh decimation.
6. Reducing unnecessary geometric detail.
7. Recalculating vertex normals where required.
8. Saving the optimized model.
9. Counting the optimized vertices.
10. Calculating the number of vertices reduced.
11. Comparing original and optimized file sizes.

The purpose is to reduce the mesh complexity and storage requirements while keeping the overall shape visually acceptable.

The 2D and 3D optimization pipelines operate independently.

---

## 6. Evaluation Metrics

### 6.1 Original Size

The original file size is measured before optimization.

### 6.2 Compressed Size

The optimized file size is measured after processing.

### 6.3 Storage Saved

Storage saved is calculated as:

    Storage Saved = Original Size - Compressed Size

### 6.4 Compression Ratio

The compression ratio is calculated as:

    Compression Ratio = Original Size / Compressed Size

A higher ratio indicates greater size reduction.

### 6.5 SSIM

Structural Similarity Index (SSIM) is used for 2D images to measure the structural similarity between the original and compressed image.

The value is generally interpreted between 0 and 1:

    1.0  -> Very high structural similarity
    Lower value -> Greater visual/structural difference

SSIM is only applicable to the 2D image branch.

### 6.6 Vertex Reduction

For supported 3D meshes:

    Vertex Reduction = Original Vertices - Optimized Vertices

This indicates how many vertices were removed during mesh simplification.

---

## 7. Excel Report

After optimization, the system automatically generates an Excel report.

The report contains:

    File Name
    Original Size
    Compressed Size
    Compression Ratio
    SSIM (2D)
    Vertices Reduced (3D)

For a 2D image, the Vertices Reduced field is not applicable.

For a 3D model, the SSIM field is not applicable.

The report is saved inside:

    Test_Box_Optimized

with the filename:

    compression_report.xlsx

---

## 8. GUI Features

The graphical user interface provides:

- Folder selection
- Optimization start button
- Processing logs
- 2D classification information
- 3D optimization information
- Total storage saved
- Completion notification
- Output folder information
- Excel report generation

The user does not need to manually select each individual asset.

The user selects one folder containing the assets, and the system recursively scans the folder.

---

## 9. Supported 2D Formats

The system processes:

    .png
    .jpg
    .jpeg

---

## 10. Supported 3D Formats

The 3D pipeline is designed around mesh-based optimization.

The current implementation primarily performs actual mesh decimation on:

    .obj

If additional formats such as FBX or GLB are handled by the current implementation, their processing depends on the corresponding 3D conversion/optimization support configured in the project.

---

## 11. Project Structure

A recommended project structure is:

    GameAssetOptimizer/
    |
    +-- 2_universal_app.py
    |
    +-- train_classifier.py
    |
    +-- classifier_model.h5
    |
    +-- dataset/
    |   |
    |   +-- train/
    |   |   +-- character/
    |   |   +-- environment/
    |   |   +-- texture/
    |   |   +-- object/
    |   |
    |   +-- val/
    |       +-- character/
    |       +-- environment/
    |       +-- texture/
    |       +-- object/
    |
    +-- Test_Box/
    |   |
    |   +-- 2D assets
    |   +-- 3D assets
    |
    +-- Test_Box_Optimized/
    |
    +-- compression_report.xlsx
    |
    +-- README.md

The exact dataset directory names may be changed as long as they match the paths used by `train_classifier.py`.

---

## 12. Development Environment

### Programming Language

    Python 3.x

### Development Tools

    Visual Studio Code
    Python
    Tkinter

### Deep Learning

    TensorFlow
    Keras

### Image Processing

    OpenCV
    NumPy

### 3D Processing

    Open3D

### Image Quality Evaluation

    scikit-image

### Excel Report Generation

    openpyxl

---

## 13. Required Python Packages

Install the required packages using:

    pip install tensorflow opencv-python numpy open3d scikit-image openpyxl

Tkinter is normally included with Python on Windows.

If Tkinter is missing on a Linux installation, install the corresponding system package for your Linux distribution.

---

# 14. How to Run the Project

## Step 1: Install Python

Install Python 3.x and make sure Python is available from the terminal.

Check:

    python --version

or:

    py --version

---

## Step 2: Open the Project

Open the project folder in Visual Studio Code.

Example:

    GameAssetOptimizer/

---

## Step 3: Create a Virtual Environment

In the VS Code terminal:

    python -m venv venv

Activate it on Windows:

    venv\Scripts\activate

After activation, the terminal should show something similar to:

    (venv)

---

## Step 4: Install Required Packages

Run:

    pip install tensorflow opencv-python numpy open3d scikit-image openpyxl

Wait until all packages are installed successfully.

---

# 15. Train the 2D Classification Model

The classifier must be trained before running the universal optimizer.

Prepare the dataset in this structure:

    dataset/
    |
    +-- train/
    |   |
    |   +-- character/
    |   +-- environment/
    |   +-- texture/
    |   +-- object/
    |
    +-- val/
        |
        +-- character/
        +-- environment/
        +-- texture/
        +-- object/

The `train` folder contains images used for learning.

The `val` folder contains separate images used to evaluate how well the model generalizes to images it did not directly train on.

The classes must have the same names in both folders.

Example:

    train/
        character/
        environment/
        texture/
        object/

    val/
        character/
        environment/
        texture/
        object/

---

## Step 16: Train the Classifier

Run:

    python train_classifier.py

or, on Windows:

    py train_classifier.py

The trained model should be saved as:

    classifier_model.h5

Make sure `classifier_model.h5` is located where `2_universal_app.py` expects it.

---

# 17. Prepare the Input Assets

Create or use the input folder:

    Test_Box/

Place the game assets inside it.

Example:

    Test_Box/
    |
    +-- grass.png
    +-- tree.jpg
    +-- brick.png
    +-- building.jpg
    +-- character.png
    +-- weapon.jpg
    +-- house.obj

The folder may contain both 2D and supported 3D assets.

---

# 18. Run the Universal Optimizer

After the classifier has been trained, run:

    python 2_universal_app.py

or:

    py 2_universal_app.py

The GUI will open.

---

# 19. Using the GUI

### Step 1

Click:

    Select Folder

### Step 2

Select:

    Test_Box

### Step 3

Click:

    Start Optimization

The system will scan the selected folder.

For 2D images:

    Image
      ↓
    Preprocessing
      ↓
    Deep Learning Classification
      ↓
    Character / Object / Environment / Texture
      ↓
    Adaptive Compression
      ↓
    SSIM Calculation
      ↓
    Save Optimized Image

For 3D models:

    3D Model
      ↓
    Mesh Loading
      ↓
    Mesh Analysis
      ↓
    Quadric Decimation
      ↓
    Vertex/Face Reduction
      ↓
    Save Optimized Model
      ↓
    Vertex Reduction Calculation

---

# 20. Output

The optimized assets are stored in:

    Test_Box_Optimized/

The Excel report is also generated there:

    Test_Box_Optimized/compression_report.xlsx

The GUI displays the total storage saved after processing.

Example:

    Total Saved: 25.42 MB

---

# 21. Example Excel Report

The generated report follows this structure:

    +----------------+---------------+----------------+-------------------+----------+-------------------+
    | File Name      | Original Size | Compressed Size| Compression Ratio | SSIM     | Vertices Reduced  |
    +----------------+---------------+----------------+-------------------+----------+-------------------+
    | grass.png      | 2 MB          | 0.4 MB         | 5.00              | 0.72     | -                 |
    | character.png  | 1.5 MB        | 1.4 MB         | 1.07              | 0.98     | -                 |
    | house.obj      | 8 MB          | 4 MB           | 2.00              | -        | 12500             |
    +----------------+---------------+----------------+-------------------+----------+-------------------+

The actual values depend on the input assets and optimization parameters.

---

# 22. Important Design Principle

The main principle of the system is adaptive optimization rather than applying the same compression level to every asset.

The system follows:

    Character
        ↓
    Very Low Compression
        ↓
    Preserve Visual Quality

    Object
        ↓
    Moderate Compression
        ↓
    Balance Quality and Size

    Environment / Texture
        ↓
    Aggressive Compression + Pixelation
        ↓
    Maximum Storage Reduction

For 3D:

    3D Model
        ↓
    Mesh Analysis
        ↓
    Quadric Decimation
        ↓
    Reduce Vertices and Triangles
        ↓
    Preserve Overall Shape
        ↓
    Smaller Model

---

# 23. Why AI Classification Is Used

Different game assets have different visual importance.

For example, a player character is usually more visually important than a distant grass texture.

Therefore, applying the same compression level to every image is not desirable.

The trained classifier allows the system to identify the type of asset and choose an appropriate optimization strategy.

This makes the compression process content-aware rather than applying a single fixed compression setting to all assets.

---

# 24. Why SSIM Is Used

SSIM is used to evaluate the structural similarity between the original and optimized 2D image.

It provides an objective numerical measurement of image quality instead of relying only on visual inspection.

This allows the project to report both:

    Storage Efficiency
    +
    Image Quality

---

# 25. Why Vertex Reduction Is Used for 3D

File size alone does not show the complete effect of 3D optimization.

A 3D model can become smaller because of file encoding or compression without actually reducing its geometric complexity.

Therefore, this project also measures the number of vertices reduced.

This provides a direct indication of mesh simplification.

The main 3D metrics are:

    Original Vertices
    Optimized Vertices
    Vertices Reduced
    Original File Size
    Optimized File Size
    Compression Ratio

---

# 26. Expected Result

The expected result is a reduction in overall game asset storage while applying different optimization levels according to asset importance.

The system is particularly designed to:

- Aggressively optimize environment assets.
- Aggressively optimize texture assets.
- Moderately optimize object assets.
- Apply very light optimization to character assets.
- Reduce the mesh complexity of supported 3D models.
- Measure image quality using SSIM.
- Measure 3D geometric reduction using vertex counts.
- Provide quantitative storage-saving results.
- Generate an Excel report automatically.

---

# 27. Project Technologies

    Python
    TensorFlow / Keras
    OpenCV
    NumPy
    Open3D
    scikit-image
    openpyxl
    Tkinter

---

# 28. Project Summary

The AI-Based Game Asset Optimization System combines deep learning, image compression, image quality assessment, and 3D mesh simplification into a unified game asset optimization pipeline.

The system first identifies the type of asset and then selects an appropriate optimization strategy. Important assets such as characters receive minimal compression, while less visually critical assets such as environments and textures receive stronger compression. For 3D assets, mesh decimation reduces geometric complexity by reducing vertices and triangles.

Finally, the system compares the original and optimized assets and generates an Excel report containing file size, compression ratio, SSIM, and vertex reduction measurements.

The final goal is to reduce storage requirements while maintaining an acceptable level of visual quality and 3D model structure.
