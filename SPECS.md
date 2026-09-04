# Quantum Machine Learning (QCNN) Technical Specifications

This document outlines the detailed system, architectural, data encoding, and framework specifications for the Quantum Convolutional Neural Network (QCNN) and Quantum Machine Learning (QML) projects in this repository.

---

## 1. System & Framework Specifications

| Specification Component | Details |
| :--- | :--- |
| **Primary Programming Language** | Python 3.8+ |
| **Quantum Frameworks** | Google TensorFlow Quantum (TFQ) `v0.7.2+`, Google Cirq `v0.14.0+`, IBM Qiskit `v0.34.0+`, Qiskit Machine Learning `v0.4.0+` |
| **Classical ML Frameworks** | TensorFlow `v2.8.0+`, PyTorch `v1.11.0+`, Scikit-Learn `v1.0.0+` |
| **Quantum Simulators** | `cirq.Simulator`, Qiskit Aer `qasm_simulator`, `statevector_simulator`, TFQ `tfq.layers.PQC` |
| **Data Processing & Viz** | NumPy, Pandas, Matplotlib, Seaborn, SymPy |

---

## 2. Quantum Architecture Specifications (QCNN)

### 2.1 QCNN Topology (Cong et al. Architecture)
- **Qubit Allocation**: 4 to 8 qubits depending on input feature dimensionality.
- **Quantum Convolutional Layer**:
  - Operates on adjacent qubit pairs $(q_i, q_{i+1})$.
  - Applies 2-qubit parameterized unitary gates $U(\vec{\theta})$ composed of single-qubit rotations ($R_x, R_y, R_z$) and two-qubit entangling gates ($CNOT$, $CZ$).
  - Implements translational invariance by sharing parameter vectors $\vec{\theta}$ across spatial convolutions.
- **Quantum Pooling Layer**:
  - Reduces spatial dimensionality by measuring or tracing out half of the active qubits.
  - Applies controlled unitary operations $V(\vec{\phi})$ from measured qubits to remaining target qubits.
- **Quantum Readout & Measurement**:
  - Pauli-Z expectation value measurements $\langle Z_i \rangle$ on remaining target qubits.
  - Output mapped directly to continuous values $[-1, 1]$ or processed through classical Softmax/Dense layers for multi-class classification.

### 2.2 Classical CNN Baseline Architecture
- **Input Shape**: $28 \times 28 \times 1$ grayscale image tensor.
- **Conv2D Layer**: 32 filters, $3 \times 3$ kernel size, ReLU activation.
- **Flatten Layer**: Converts 2D spatial feature maps to 1D vector ($21,632$ units).
- **Dense Layer 1**: 128 units, ReLU activation.
- **Output Dense Layer**: 10 units (logits for digits 0-9).

---

## 3. Data Preprocessing & Quantum Encoding Specifications

| Data Processing Pipeline | Specification |
| :--- | :--- |
| **MNIST Standard Normalization** | Pixel intensity rescaling $x \in [0, 255] \rightarrow x' \in [0.0, 1.0]$. |
| **Dimensionality Reduction** | Downsampling $28 \times 28 \rightarrow 8 \times 8$ or PCA reduction to $N$-qubit feature vectors. |
| **Binary Sub-selection** | Class filtering for binary QCNN benchmarks (e.g. Digits 3 vs 6). |
| **Quantum State Encoding** | **Angle Encoding**: Maps feature $x_i$ to rotation angle $\theta_i = \pi \cdot x_i$ for $R_x(\theta_i)$ / $R_y(\theta_i)$.<br>**Amplitude Encoding**: Prepares $2^N$-dimensional state vector $|\psi(x)\rangle = \sum x_i |i\rangle$. |
| **Data Re-Uploading** | Interleaves single-qubit data encoding gates $U(x)$ between trainable unitary layers $W(\theta)$ to boost expressivity. |

---

## 4. Hyperparameter & Training Specifications

| Parameter | Value / Setting |
| :--- | :--- |
| **Optimizer** | Adam (`learning_rate=0.01` for QCNN, `learning_rate=0.001` for CNN) |
| **Loss Functions** | Binary Cross-Entropy / Mean Squared Error / Hinge Loss (Quantum), Sparse Categorical Cross-Entropy (Classical) |
| **Batch Size** | 32 (MNIST CNN / QCNN), 5 (Pneumonia Detection QCNN) |
| **Epochs** | 5 to 50 epochs depending on model convergence |
| **Evaluation Metrics** | Accuracy (%), Training Loss, Test Loss |
