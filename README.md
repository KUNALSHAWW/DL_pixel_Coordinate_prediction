# Deep Learning Pixel Coordinate Prediction

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Deep Learning solution for predicting the coordinates of a single white pixel in grayscale images using Convolutional Neural Networks (CNNs).

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Approach & Rationale](#approach--rationale)
- [Architecture](#architecture)
- [Dataset Design](#dataset-design)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Code Quality](#code-quality)
- [Project Structure](#project-structure)
- [License](#license)

## 🎯 Problem Statement

Given a **50×50 grayscale image** where:
- Exactly **1 pixel** has a value of **255** (white)
- All other **2,499 pixels** have a value of **0** (black)
- The white pixel is **randomly positioned**

**Goal:** Build a Deep Learning model to predict the **(x, y)** coordinates of the white pixel.

## 🧠 Approach & Rationale

### Why Deep Learning?

This problem is ideal for CNNs because:
1. **Spatial Hierarchy:** CNNs excel at learning spatial patterns through convolutional layers
2. **Translation Invariance:** The white pixel can appear anywhere in the image
3. **Feature Extraction:** Automatic learning of relevant features without manual engineering
4. **Regression Task:** Direct prediction of continuous coordinate values

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **CNN Architecture** | Preserves spatial information through convolution and pooling layers |
| **Flatten Layer** | Maintains positional context (vs. Global Pooling which loses spatial info) |
| **Sigmoid Activation** | Bounds predictions to [0, 1] range for normalized coordinates |
| **MSE Loss** | Standard loss function for regression tasks measuring pixel distance |
| **Early Stopping** | Prevents overfitting and optimizes training time |

## 🏗️ Architecture

The model uses a lightweight CNN with the following structure:

```
Input (50×50×1)
    ↓
Conv2D(32) + ReLU + MaxPooling(2×2)
    ↓
Conv2D(64) + ReLU + MaxPooling(2×2)
    ↓
Conv2D(128) + ReLU + MaxPooling(2×2)
    ↓
Flatten
    ↓
Dense(128) + ReLU + Dropout(0.3)
    ↓
Dense(64) + ReLU
    ↓
Dense(2) + Sigmoid → (x, y) predictions
```

**Key Features:**
- **3 Convolutional Blocks:** Progressive feature extraction (32 → 64 → 128 filters)
- **Spatial Downsampling:** MaxPooling reduces dimensionality while retaining important features
- **Dropout Layer:** Regularization to prevent overfitting (30% dropout rate)
- **Fully Connected Layers:** Transform spatial features into coordinate predictions

## 📊 Dataset Design

### Generation Strategy

```python
# Dataset Configuration
- Training Samples: 3,200
- Test Samples: 800
- Image Size: 50×50 pixels
- Color Space: Grayscale (single channel)
- Total Possible Positions: 2,500 (50×50)
```

### Rationale

1. **Sample Size (4,000 total):**
   - Provides **1.6× coverage** of the entire search space (2,500 positions)
   - Ensures statistical diversity while remaining computationally efficient
   - Balances between overfitting (too few samples) and training time (too many samples)

2. **80-20 Train-Test Split:**
   - Standard practice for model validation
   - 800 test samples provide reliable performance metrics

3. **Random Sampling:**
   - Uniform distribution across all possible positions
   - Prevents spatial bias in the model

4. **Normalization:**
   - Coordinates normalized to [0, 1] range
   - Improves training stability and convergence speed

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/KUNALSHAWW/DL-Pixel-Coordinate-Prediction.git
cd DL-Pixel-Coordinate-Prediction
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Dependencies

```
tensorflow>=2.10.0
numpy>=1.23.0
matplotlib>=3.5.0
seaborn>=0.12.0
```

## 💻 Usage

### Running the Notebook

1. **Launch Jupyter Notebook:**
```bash
jupyter notebook DL_pixel_Coordinate_prediction.ipynb
```

2. **Execute cells sequentially** to:
   - Generate synthetic dataset
   - Build and train the CNN model
   - Visualize training progress
   - Evaluate performance metrics
   - View prediction results

### Quick Start (Python Script)

```python
import numpy as np
from tensorflow.keras import models

# Load the trained model
model = models.load_model('pixel_locator_model.h5')

# Create a test image (50×50 with white pixel at position [25, 30])
test_image = np.zeros((1, 50, 50, 1))
test_image[0, 25, 30, 0] = 1.0

# Predict coordinates (normalized to [0, 1])
predicted_coords = model.predict(test_image)

# Convert to pixel coordinates
x, y = predicted_coords[0] * 49  # Denormalize
print(f"Predicted coordinates: ({x:.2f}, {y:.2f})")
```

## 📈 Results

### Performance Metrics

The model achieves excellent localization accuracy:

| Metric | Value |
|--------|-------|
| **Mean Error** | ~0.5 pixels |
| **Max Error** | ~3.0 pixels |
| **Sub-pixel Accuracy (<1px)** | ~95%+ |
| **Training Time** | <10 epochs (with early stopping) |

### Visualization Examples

The notebook includes:
- **Training/Validation Loss Curves:** Monitor model convergence
- **Error Distribution Histograms:** Analyze prediction accuracy
- **Sample Predictions:** Visual comparison of ground truth vs. predicted positions
- **Error Heatmaps:** Spatial distribution of prediction errors

### Key Insights

1. ✅ Model converges rapidly (typically within 5-8 epochs)
2. ✅ Predictions are consistently sub-pixel accurate
3. ✅ No spatial bias detected (errors uniformly distributed)
4. ✅ Robust to positional variance across the entire image

## 🎨 Code Quality

This project adheres to professional coding standards:

### PEP 8 Compliance
- ✓ Consistent 4-space indentation
- ✓ Descriptive variable and function names
- ✓ Maximum line length of 79 characters
- ✓ Proper spacing around operators and commas

### Best Practices
- **Comprehensive Documentation:** All functions include docstrings
- **Type Hints:** Clear parameter and return types
- **Error Handling:** Robust validation and exception management
- **Code Comments:** Inline explanations for complex logic
- **Modularity:** Reusable functions for dataset generation and model building
- **Reproducibility:** Fixed random seeds for consistent results

### Code Organization
```
├── Imports & Configuration    # Setup and dependencies
├── Dataset Generation         # Synthetic data creation
├── Model Architecture         # CNN design and compilation
├── Training                   # Model training with callbacks
├── Evaluation                 # Performance metrics
└── Visualization             # Results and analysis
```

## 📁 Project Structure

```
DL-Pixel-Coordinate-Prediction/
│
├── DL_pixel_Coordinate_prediction.ipynb  # Main implementation notebook
├── requirements.txt                       # Python dependencies
├── README.md                              # Project documentation
├── LICENSE                                # MIT License
```

### Ideas for Enhancement
- [ ] Implement multi-pixel localization
- [ ] Add support for colored images
- [ ] Experiment with attention mechanisms
- [ ] Deploy as web API using FastAPI
- [ ] Create interactive demo with Gradio/Streamlit

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Kunal Shaw**

- GitHub: [@KUNALSHAWW](https://github.com/KUNALSHAWW)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/kunal-kumar-shaw-443999205/)

## 🙏 Acknowledgments

- TensorFlow/Keras team for the excellent deep learning framework
- The computer vision community for inspiration and best practices
- Open-source contributors who make projects like this possible

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ and Python

</div>
