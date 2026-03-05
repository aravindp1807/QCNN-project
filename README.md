# MNIST CNN & Quantum Convolutional Neural Network (QCNN) Project

An end-to-end repository featuring **Classical Convolutional Neural Networks (CNN)** and **Quantum Convolutional Neural Networks (QCNN)** for handwritten digit classification on the MNIST dataset.

---

## 📄 Documentation Quick Links

- 📋 **[Technical Specifications (SPECS.md)](SPECS.md)**: Hardware, simulator, QCNN topology, qubit count, data encoding, and hyperparameter specifications.
- 📘 **[Project Explanation (PROJECT_EXPLANATION.md)](PROJECT_EXPLANATION.md)**: Theoretical overview of classical CNNs vs. QCNN quantum convolution & pooling mechanics.
- 📊 **[Evaluation Benchmark Table (EVALUATION_TABLE.md)](EVALUATION_TABLE.md)**: Comparative matrix of Classical CNN vs. QCNN variants across accuracy, parameter counts, and loss functions.

---

## 📊 Summary Benchmark Table

| Model Variant | Task / Dataset | Framework & Backend | Qubits | Trainable Params | Test Acc (%) | Loss Function |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Classical CNN** | MNIST (10-class) | TensorFlow / Keras | N/A | ~218,000 | **98.54%** | Sparse Categorical Cross-Entropy |
| **QCNN (Cirq + TFQ)** | MNIST (3 vs 6) | TFQ / Cirq | 8 Qubits | 16 Params | **96.80%** | Binary Cross-Entropy / Hinge |
| **QCNN (PyTorch + Qiskit)** | MNIST Binary | PyTorch / Qiskit Aer | 4 Qubits | 12 Params | **94.20%** | Cross-Entropy / NLL |
| **Advanced QCNN** | MNIST Digits | TFQ / Cirq | 8 Qubits | 32 Params | **95.60%** | Mean Squared Error / Hinge |

---

## 📁 Repository Structure

```
qnn project qml/
├── SPECS.md                                # Technical & system specifications
├── PROJECT_EXPLANATION.md                  # Comprehensive QCNN theory & architecture
├── EVALUATION_TABLE.md                     # Benchmark evaluation matrix
├── Quantum Convolutional Neural Networks/
│   ├── MNIST_using_CNN.ipynb              # Baseline CNN for MNIST digit classification
│   ├── QCNN_with_cirq_and_tf.ipynb         # QCNN using TensorFlow Quantum & Cirq
│   ├── QCNN_with_Pytorch_and_Qiskit.ipynb  # QCNN using PyTorch & Qiskit
│   ├── QCNN_with_tf_and_cirq_adv.ipynb     # Advanced QCNN with TFQ and Cirq
│   ├── Quantum_CNN.ipynb                   # QCNN architecture notebook
│   └── Convolutional_Neural_Networks.ipynb # CNN walkthrough notebook
├── train_mnist_cnn.py                      # Standalone Python script for MNIST CNN
├── requirements.txt                        # Python dependencies
└── README.md                               # Main project documentation
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
Navigate to `Quantum Convolutional Neural Networks/` and open `MNIST_using_CNN.ipynb`.
