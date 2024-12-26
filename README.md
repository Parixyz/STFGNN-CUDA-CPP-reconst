# Reimplementing Spatial-Temporal Fusion Graph Neural Network (STFGNN) for Traffic Flow Forecasting

This repository contains an implementation of STFGNN using two approaches:
1. **C++/CUDA Implementation**: Optimized for GPU acceleration using CUDA kernels for preprocessing, graph construction, and spatial-temporal graph convolutions.
2. **TensorFlow Implementation**: Python-based model for comparison and benchmarking.

---

## **Features**

### C++/CUDA Implementation
- Z-score normalization with CUDA.
- Construction of Spatial (ASG), Temporal (ATG), and Fusion (ASTFG) graphs.
- Efficient graph convolutions and gated convolutions.
- Fully customizable model parameters via `main.cpp`.
- High performance with CUDA kernels.

### TensorFlow Implementation
- Python-based implementation for rapid prototyping.
- Focus on simplicity and comparison.
- Utilizes TensorFlow's high-level API for graph convolutions and temporal modeling.

---

## **Setup Instructions**

### Prerequisites
- **For C++/CUDA Implementation**:
  - Visual Studio 2022.
  - CUDA Toolkit 12.1.
  - CMake 3.20+.
  - Eigen library.
- **For TensorFlow Implementation**:
  - Python 3.8+.
  - TensorFlow 2.8+.

### Building the C++/CUDA Project

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Configure the project using CMake:
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_C_COMPILER=nvcc -DCMAKE_CXX_COMPILER=g++
   make
   ```

3. Run the program:
   ```bash
   ./stfgnn
   ```

### Running the TensorFlow Implementation

1. Navigate to the TensorFlow folder:
   ```bash
   cd tensorflow_implementation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the program:
   ```bash
   python tensorflow_comparison.py
   ```

---

## **File Structure**

### C++/CUDA Files
| **File**               | **Purpose**                                                |
|------------------------|------------------------------------------------------------|
| `dataloader.h/cpp`     | Loads and preprocesses the dataset.                        |
| `preprocessing.cu`     | CUDA kernel for Z-score normalization.                     |
| `graph_construction.h/cpp` | Constructs ASG, ATG, and ASTFG graphs.                 |
| `stfgnn_layer.h/cpp`   | Implements spatial-temporal graph convolutions.            |
| `gated_conv.h/cpp`     | Implements temporal gated convolutions.                    |
| `fusion_module.h/cpp`  | Combines spatial and temporal features.                    |
| `level1_layers.h/cpp`  | Combines multiple layers (STFGNN and Gated Conv).          |
| `level0_model.h/cpp`   | Full model pipeline including all components.              |
| `main.cpp`             | Orchestrates the pipeline and runs training.               |

### TensorFlow Files
| **File**                | **Purpose**                                                |
|-------------------------|------------------------------------------------------------|
| `tensorflow_comparison.py` | Implements the STFGNN model using TensorFlow.            |

---

## **Comparison of Implementations**

| **Feature**            | **C++/CUDA Implementation**         | **TensorFlow Implementation** |
|------------------------|--------------------------------------|--------------------------------|
| Performance            | High (GPU-optimized).               | Moderate.                     |
| Flexibility            | Full control over CUDA kernels.      | High-level API.               |
| Ease of Use            | Moderate (requires setup).           | Easy (Python-based).          |
| Purpose                | Optimized for production use cases.  | Prototyping and benchmarking. |

---

## **Future Work**
- Add dynamic graph updates for real-time data.
- Optimize CUDA kernels for sparse matrix operations.
- Enhance TensorFlow implementation for larger datasets.

---

## **References**
- [STFGNN Original Paper](https://arxiv.org/abs/2012.09641)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
