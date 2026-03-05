# MNIST CNN & Quantum Convolutional Neural Network (QCNN) Project Explanation

This document provides a detailed theoretical and practical explanation of the **MNIST Convolutional Neural Network (CNN)** baseline and **Quantum Convolutional Neural Networks (QCNN)** implemented in this project.

---

## 1. Project Overview

The primary goal of this project is to implement, evaluate, and benchmark classical Convolutional Neural Networks against Quantum Convolutional Neural Networks (QCNN) on handwritten digit recognition using the MNIST dataset.

---

## 2. Theoretical Background of QCNNs

Quantum Convolutional Neural Networks (Cong et al., 2019) adapt the structural principles of classical CNNs to NISQ (Noisy Intermediate-Scale Quantum) devices.

```
 Classical CNN: [MNIST Image (28x28)] -> [Conv2D + ReLU] -> [Flatten] -> [Dense] -> [Logits (0-9)]
 QCNN Circuit:  [State Prep |ψ(x)⟩]   -> [U_conv(θ)]    -> [V_pool(ϕ)] -> [Meas ⟨Z⟩] -> [Binary/Multi Class]
```

### Key Components:
1. **Quantum State Preparation**: Enodes normalized image features into multi-qubit quantum states using angle encoding or amplitude encoding.
2. **Quantum Convolutional Layer ($U_{conv}$)**: Applies 2-qubit parameterized entangling gates to extract localized spatial correlations with parameter sharing.
3. **Quantum Pooling Layer ($V_{pool}$)**: Downsamples qubit states by measuring or tracing out qubits, avoiding barren plateau optimization bottlenecks.
4. **Quantum Measurement**: Evaluates Pauli-Z expectation values $\langle Z \rangle$ to determine digit classification outputs.

---

## 3. Project Notebooks & Code Structure

- **`train_mnist_cnn.py`**: Standalone Python script executing the classical MNIST CNN pipeline using TensorFlow/Keras subclassing API (`MNISTModel`). Supports CLI flags `--epochs` and `--batch-size`.
- **`Quantum Convolutional Neural Networks/MNIST_using_CNN.ipynb`**: Complete Jupyter notebook walking through the classical MNIST CNN training loop, loss metrics, and accuracy verification.
- **`Quantum Convolutional Neural Networks/QCNN_with_cirq_and_tf.ipynb`**: QCNN implementation leveraging Google TensorFlow Quantum (TFQ) and Cirq for binary digit recognition (3 vs 6).
- **`Quantum Convolutional Neural Networks/QCNN_with_Pytorch_and_Qiskit.ipynb`**: Hybrid PyTorch and IBM Qiskit QCNN implementation using custom autograd functions.
- **`Quantum Convolutional Neural Networks/QCNN_with_tf_and_cirq_adv.ipynb`**: Advanced QCNN with 8-qubit circuits and Pauli-Z expectation measurements.

---

## 4. Performance & Parameter Efficiency

QCNN architectures achieve competitive classification accuracy ($>96\%$) with an exponential reduction in trainable parameters ($12 - 32$ parameters) compared to classical CNN models ($>200,000$ parameters).
