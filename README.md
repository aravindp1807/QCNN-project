# Quantum Machine Learning & QCNN Project

An end-to-end repository featuring **Quantum Convolutional Neural Networks (QCNN)**, **Quantum Support Vector Machines (QSVM)**, **Data Re-Uploading**, and classical **Convolutional Neural Networks (CNN)** applied to image recognition, high-energy physics, disease detection, and astronomy.

---

## 📄 Documentation Quick Links

- 📋 **[Technical Specifications (SPECS.md)](SPECS.md)**: Hardware, simulator, QCNN topology, qubit count, data encoding, and hyperparameter specifications.
- 📘 **[Project Explanation (PROJECT_EXPLANATION.md)](PROJECT_EXPLANATION.md)**: Theoretical background of QML, QCNN quantum convolution & pooling mechanics, and domain application breakdowns.
- 📊 **[Evaluation Benchmark Table (EVALUATION_TABLE.md)](EVALUATION_TABLE.md)**: Quantitative comparative analysis of Classical CNN vs. QCNN, QSVM, and QNN variants across accuracy, parameter efficiency, and loss functions.

---

## 📊 Summary Evaluation Benchmark

| Model Architecture | Target Dataset | Framework / Backend | Qubits | Trainable Params | Test Acc (%) | Loss Function |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Classical CNN** | MNIST (10-class) | TensorFlow / Keras | N/A | ~218,000 | **98.54%** | Sparse Categorical Cross-Entropy |
| **QCNN (Cirq + TFQ)** | MNIST (3 vs 6) | TFQ / Cirq | 8 Qubits | 16 Params | **96.80%** | Binary Cross-Entropy / Hinge |
| **QCNN (PyTorch + Qiskit)** | MNIST Binary | PyTorch / Qiskit Aer | 4 Qubits | 12 Params | **94.20%** | Cross-Entropy / NLL |
| **QCNN Pneumonia Detector** | Chest X-Ray | TensorFlow / Cirq | 4-8 Qubits | 24 Params | **88.50%** | Binary Cross-Entropy |
| **QSVM Classifier** | QCD Physics Dataset | Qiskit ML / Dual SVM | 4 Qubits | Quantum Kernel | **93.40%** | Hinge Loss |

*For full evaluation metrics and insights, see [EVALUATION_TABLE.md](EVALUATION_TABLE.md).*

---

## 📁 Repository Structure

```
qnn project qml/
├── SPECS.md                                # Technical & system specifications
├── PROJECT_EXPLANATION.md                  # Comprehensive QML & QCNN theory & architecture
├── EVALUATION_TABLE.md                     # Quantitative evaluation & benchmark matrix
├── Quantum Convolutional Neural Networks/
│   ├── MNIST_using_CNN.ipynb              # Baseline CNN for MNIST digit classification
│   ├── QCNN_with_cirq_and_tf.ipynb         # QCNN using TensorFlow Quantum & Cirq
│   ├── QCNN_with_Pytorch_and_Qiskit.ipynb  # QCNN using PyTorch & Qiskit
│   ├── QCNN_with_tf_and_cirq_adv.ipynb     # Advanced QCNN with TFQ and Cirq
│   ├── Quantum_CNN.ipynb                   # QCNN architecture notebook
│   └── Convolutional_Neural_Networks.ipynb # CNN walkthrough notebook
├── QCNN for Disease Detection/
│   └── QCNN_for_Pneumonia_Detection.ipynb  # QCNN applied to chest X-ray pneumonia detection
├── Higgs_Classification/
│   └── higgs_classification.ipynb          # Quantum classification for high-energy physics
├── Galaxy Detection using Quantum Machine Learning/
│   └── jupyter notebooks/                  # QNN for astronomical image classification
├── Quantum Support Vector Machine on QCD/
│   └── code.ipynb                          # QSVM model implementation
├── train_mnist_cnn.py                      # Standalone Python script for MNIST CNN
├── requirements.txt                        # Python dependencies
└── Quantum_Machine_Learning.pdf            # Reference paper and study guide
```

---

## ⚡ Quick Start

### 1. Environment Setup

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Standalone MNIST CNN Model

```bash
python train_mnist_cnn.py --epochs 5 --batch-size 32
```

### 3. Launch Jupyter Notebooks

```bash
jupyter notebook
```
