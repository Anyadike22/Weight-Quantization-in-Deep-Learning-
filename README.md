# Weight-Quantization-in-Deep-Learning

**Weight quantization** is a technique used in machine learning—especially in deep learning—to reduce the size and computational cost of a model by representing weights with lower precision data types.

### 🔍 In simple terms:
Instead of storing weights as 32-bit floating point numbers (`float32`), you store them as smaller types like:
- **8-bit integers (`int8`)**
- **16-bit floats (`float16`)**, etc.

---

### 💡 Why use weight quantization?
1. **Smaller model size** → great for mobile, embedded, or edge devices.
2. **Faster inference** → smaller numbers are quicker to process.
3. **Lower memory usage** → helpful for memory-constrained environments.
4. **Less power consumption** → efficient for devices like smartphones or IoT devices.

---

### 🧠 Types of quantization:
- **Dynamic quantization**: Weights are quantized once before inference and activations are quantized just-in-time.
- **Post-training quantization**: Apply quantization *after* training a model.
- **Quantization-aware training (QAT)**: Simulates quantization *during* training to maintain accuracy.

---

### ⚖️ Trade-off:
- **Pros**: Speed, size, deployment ease.
- **Cons**: Slight accuracy drop (depends on how aggressive the quantization is).

---

### 🧪 Example:
Original model:
```python
weight = 0.823574  # stored as float32
```

Quantized model (int8 with scaling factor):
```python
quantized_weight = int8(105)
scale = 0.00784
real_weight = quantized_weight * scale ≈ 0.8232
```

---











```python
quantized_model = torch.quantization.quantize_dynamic(model_fp32, {nn.Linear}, dtype=torch.qint8)
```

---

### ✅  **Dynamic Quantization**

#### 📌 Key characteristics of dynamic quantization:
- Only **`nn.Linear`** layers are quantized (as specified in `{nn.Linear}`).
- It uses **`torch.qint8`** (8-bit integers) to store weights.
- **Activations** are quantized **dynamically at runtime**, not during training or ahead of time.
- **Model remains in inference mode** (no retraining required).
- Weights are quantized once before inference and activations are quantized just-in-time.

---

### Summary:

| Aspect                  | Description                                       |
|-------------------------|---------------------------------------------------|
| Quantization Type       | **Dynamic Quantization**                          |
| Applied to              | `nn.Linear` layers only                           |
| Weight dtype            | `torch.qint8` (8-bit integers)                    |
| Activations quantized?  | Yes, dynamically at runtime                       |
| Training aware?         | ❌ No, applied post-training                      |
| Benefit                 | Reduced model size and faster inference           |
| Trade-off               | Minor accuracy loss (compared to full precision)  |

