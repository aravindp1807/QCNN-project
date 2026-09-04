# Quantum Machine Learning (QCNN) Project Explanation

This document provides a comprehensive theoretical and architectural explanation of the **Quantum Convolutional Neural Networks (QCNN)** and hybrid Quantum Machine Learning (QML) models implemented in this repository.

---

## 1. Introduction to Quantum Machine Learning (QML)

Near-Term Quantum Devices (NISQ — Noisy Intermediate-Scale Quantum) operate with a limited number of qubits (tens to hundreds) without full fault-tolerant error correction. Hybrid Quantum-Classical Algorithms leverage classical optimizers to adjust parameters of Quantum Circuits, minimizing noise overhead while capitalizing on quantum mechanics (superposition and entanglement).

Quantum Convolutional Neural Networks (QCNN), first proposed by Cong, Choi, and Lukin (2019), translate the structural principles of classical Convolutional Neural Networks (CNNs) into parametrized quantum circuits (PQCs).

---

## 2. Core QCNN Architecture & Mechanics

### 2.1 The Analogy between Classical CNN and QCNN

```
 Classical CNN: [Input Image] -> [Conv2D + Act] -> [Pooling] -> [Conv2D + Act] -> [Dense] -> [Output]
 QCNN Circuit:  [Quantum State |ψ(x)⟩] -> [U_conv(θ)] -> [V_pool(ϕ)] -> [U_conv(θ')] -> [Meas ⟨Z⟩] -> [Output]
```

1. **State Preparation / Feature Encoding**:
   Classical input features (such as image pixel values) are encoded into quantum state vectors $|\psi(x)\rangle$. Data re-uploading techniques introduce non-linear mapping by interleaving encoding gates $U(x)$ throughout the circuit depth.

2. **Quantum Convolutional Layer ($U_{conv}$)**:
   - Applies 2-qubit parameterized unitary operations to spatially adjacent qubits.
   - Entangles neighboring qubits to extract localized spatial correlations.
   - Employs parameter sharing across all local unitaries in the layer to enforce translational invariance and reduce total parameter count.

3. **Quantum Pooling Layer ($V_{pool}$)**:
   - Reduces the number of active qubits in the circuit (analogous to classical max/average pooling downsampling).
   - Achieved by measuring a subset of qubits and using the measurement outcome to condition unitary rotations on adjacent unmeasured qubits, or by tracing out partial quantum states.

4. **Quantum Measurement & Readout**:
   - Expectation values of Pauli-Z operators $\langle Z_i \rangle$ are measured on the final remaining qubits.
   - The expectation values are passed into a classical loss function or linear classifier layer to produce predicted class labels.

---

## 3. Implemented Projects & Framework Breakdown

### 3.1 MNIST Digit Classification (Classical CNN & QCNN)
- **Baseline Classical CNN**: Implemented via Keras Model subclassing API (`MyModel`). Trains a $32$-filter $3\times 3$ convolutional layer followed by Dense layers on the $28\times 28$ MNIST dataset, achieving $>98\%$ test accuracy.
- **QCNN with TensorFlow Quantum & Cirq**: Employs `tfq.layers.PQC` to construct an 8-qubit QCNN circuit with parametrized Cirq gates. Filters digits (e.g. 3 vs 6), downsamples input images, and trains parameter vectors using gradient descent.
- **QCNN with PyTorch & Qiskit**: Utilizes Qiskit Aer quantum simulator combined with PyTorch `autograd` via custom `torch.autograd.Function` bridge.

### 3.2 QCNN for Disease Detection (Pneumonia Detection)
- Adapts QCNN feature extraction to medical image processing.
- Downsamples chest X-ray images, encodes normalized pixel intensities into quantum states, and trains a QCNN to distinguish between normal and pneumonia cases.

### 3.3 Higgs Boson Classification (Particle Physics)
- Classifies high-energy particle collision events from accelerator detectors.
- Encodes kinematic feature vectors into multi-qubit states and uses a variational quantum classifier (VQC) with entangling layers.

### 3.4 Galaxy Image Classification
- Applies Quantum Neural Network (QNN) architectures to astronomical image datasets to detect galaxy morphology.

### 3.5 Quantum Support Vector Machines (QSVM) on QCD Dataset
- Constructs a non-linear Quantum Kernel Feature Map $K(x_i, x_j) = |\langle \phi(x_i) | \phi(x_j) \rangle|^2$ using Qiskit.
- Computes the quantum kernel matrix and solves the dual quadratic programming problem via classical SVM.

---

## 4. Key Advantages of QCNNs on NISQ Hardware

1. **Parameter Efficiency**: QCNNs require exponentially fewer trainable parameters ($O(\log N)$ parameters) compared to classical fully connected or deep CNN networks ($O(N)$ parameters).
2. **Absence of Barren Plateaus**: Traditional deep random Quantum Neural Networks suffer from exponentially vanishing gradients (barren plateaus). The logarithmic circuit depth and structured pooling in QCNNs mitigate barren plateaus, enabling stable gradient-based optimization.
3. **Quantum Data Feature Extraction**: Exploits quantum entanglement and non-local correlations that are computationally intractable for classical models.
