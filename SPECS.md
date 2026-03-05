# MNIST CNN & Quantum Convolutional Neural Network (QCNN) Technical Specifications

This document outlines the system, architectural, data encoding, and framework specifications for the **MNIST Convolutional Neural Network (CNN)** and **Quantum Convolutional Neural Network (QCNN)** models in this project.

---

## 1. System & Framework Specifications

| Specification Component | Details |
| :--- | :--- |
| **Primary Language** | Python 3.8+ |
| **Classical Frameworks** | TensorFlow `v2.8.0+` (Keras API), PyTorch `v1.11.0+` |
| **Quantum Frameworks** | Google TensorFlow Quantum (TFQ) `v0.7.2+`, Google Cirq `v0.14.0+`, IBM Qiskit `v0.34.0+` |
| **Quantum Simulators** | `cirq.Simulator`, Qiskit Aer `qasm_simulator`, `statevector_simulator`, TFQ `tfq.layers.PQC` |
| **Data Processing** | NumPy, Scikit-Learn, Matplotlib, Seaborn, SymPy |

---

## 2. Model Architecture Specifications

### 2.1 Classical MNIST CNN Baseline Architecture
- **Input Tensor**: $28 \times 28 \times 1$ grayscale image tensor.
- **Convolutional Layer**: Conv2D ($32$ filters, $3 \times 3$ kernel size, ReLU activation).
- **Flatten Layer**: Reshapes 2D spatial feature maps into a 1D vector ($21,632$ units).
- **Dense Layer**: Dense ($128$ units, ReLU activation).
- **Output Layer**: Dense ($10$ units, linear logits for digits 0-9).
- **Parameter Count**: ~218,000 trainable parameters.

### 2.2 Quantum Convolutional Neural Network (QCNN) Architecture
- **Qubit Topology**: 4 to 8 qubits depending on downsampled feature space.
- **Quantum Convolutional Layer ($U_{conv}$)**:
  - 2-qubit parameterized unitaries $U(\vec{\theta})$ applied to adjacent qubit pairs $(q_i, q_{i+1})$.
  - Composed of single-qubit rotations ($R_x, R_y, R_z$) and $CNOT$ entangling gates.
  - Parameter sharing across spatial convolutions to enforce translational invariance.
- **Quantum Pooling Layer ($V_{pool}$)**:
  - Sub-system measurement or partial trace out to downsample active qubits ($8 \rightarrow 4 \rightarrow 2$).
  - Controlled unitaries $V(\vec{\phi})$ condition remaining qubits on measured states.
- **Quantum Measurement & Readout**:
  - Pauli-Z expectation value measurements $\langle Z_i \rangle$ on output qubits.
  - Linear mapping to class predictions.

---

## 3. Data Preprocessing & Quantum State Encoding

| Preprocessing Stage | Specification |
| :--- | :--- |
| **Normalization** | Pixel intensities scaled from $[0, 255] \rightarrow [0.0, 1.0]$. |
| **Dimension Reduction** | Crop & downsample $28 \times 28 \rightarrow 8 \times 8$ or PCA feature reduction for $N$-qubit inputs. |
| **Binary Sub-selection** | Class filtering for binary QCNN benchmarks (Digits 3 vs 6). |
| **Quantum State Preparation** | **Angle Encoding**: Maps pixel $x_i$ to rotation angle $\theta_i = \pi \cdot x_i$.<br>**Data Re-Uploading**: Interleaves encoding gates $U(x)$ between trainable unitary blocks $W(\theta)$. |

---

## 4. Hyperparameter Specifications

| Parameter | Value |
| :--- | :--- |
| **Optimizer** | Adam (`learning_rate=0.01` for QCNN, `learning_rate=0.001` for CNN) |
| **Loss Functions** | Sparse Categorical Cross-Entropy (Classical), Binary Cross-Entropy / Hinge Loss (QCNN) |
| **Batch Size** | 32 |
| **Epochs** | 5 to 50 |
