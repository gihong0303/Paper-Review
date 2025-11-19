# SqueezeLLM: Dense-and-Sparse Quantization

**논문 발표**: 2023년 (ICML 2024)
**저자**: Sehoon Kim, Coleman Hooper, Amir Gholami, Zhen Dong, Xiuyu Li, Sheng Shen, Michael W. Mahoney, Kurt Keutzer
**소속**: UC Berkeley
**논문 링크**: [arXiv:2306.07629](https://arxiv.org/abs/2306.07629)
**공식 구현**: [GitHub](https://github.com/SqueezeAILab/SqueezeLLM)

---

## 한 줄 요약
> 민감도가 높은 가중치 값들을 Sparse 행렬로 별도 저장하고, 나머지는 Non-uniform quantization으로 Dense하게 압축하는 하이브리드 방식으로, 3bit에서도 높은 정확도 달성

---

## 1. 핵심 아이디어

### 1.1 두 가지 문제점

기존 양자화의 한계:
1. **Uniform quantization**: 값 분포를 무시하고 균일 간격으로 양자화
2. **Outlier 처리**: 극단값들이 전체 양자화 정확도를 저하

### 1.2 SqueezeLLM의 해결책

```
가중치 = Sparse (민감한 outlier) + Dense (나머지)

Sparse: 원본 정밀도 유지 (FP16)
Dense:  Non-uniform quantization (3-4bit)
```

---

## 2. Dense-and-Sparse Decomposition

### 2.1 분해 방식

$$W = W_{\text{sparse}} + W_{\text{dense}}$$

```
원본 가중치:
[0.1, 50.0, -0.3, 0.5, -40.0, 0.2, ...]
     ↑              ↑
   Outlier       Outlier

분해 후:
Sparse: [0, 50.0, 0, 0, -40.0, 0, ...] (0.45% 비영)
Dense:  [0.1, 0, -0.3, 0.5, 0, 0.2, ...] (Non-uniform)
```

### 2.2 Outlier 선택 기준

민감도 기반 선택:
$$\text{sensitivity}(w_i) = \left|\frac{\partial \mathcal{L}}{\partial w_i}\right| \cdot |w_i|^2$$

상위 0.45%를 Sparse로 분류

### 2.3 추론 방식

```python
def forward(x):
    # Dense 부분: 양자화된 행렬 곱
    y_dense = quantized_matmul(x, W_dense_quant)

    # Sparse 부분: Sparse 행렬 곱 (효율적 커널)
    y_sparse = sparse_matmul(x, W_sparse)

    return y_dense + y_sparse
```

---

## 3. Non-uniform Quantization

### 3.1 vs Uniform Quantization

```
Uniform:     [--|--|--|--|--|--|--|--]
              0  1  2  3  4  5  6  7
              균일 간격

Non-uniform: [--|-|-|---|---|-------|---]
              0 1 2 3   4   5       6   7
              값 분포에 맞춤
```

### 3.2 Sensitivity-based k-means

가중치의 민감도를 고려한 클러스터링:

```python
def sensitivity_weighted_kmeans(weights, sensitivities, k):
    """
    민감도 가중 k-means로 최적 중심점 찾기
    """
    # 민감도를 가중치로 사용
    weighted_kmeans = KMeans(k, sample_weight=sensitivities)
    centroids = weighted_kmeans.fit(weights)

    return centroids
```

### 3.3 Lookup Table (LUT)

양자화된 인덱스 → 실제 값:

```python
# 3bit = 8개 중심점
LUT = [-0.8, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 1.2]

# 양자화된 가중치 (인덱스)
W_quant = [3, 7, 1, 4, 0, ...]

# 역양자화
W_dequant = LUT[W_quant]
          = [0.0, 1.2, -0.3, 0.1, -0.8, ...]
```

---

## 4. 실험 결과

### 4.1 Perplexity 비교

| 모델 | Bits | RTN | GPTQ | **SqueezeLLM** |
|------|------|-----|------|----------------|
| LLaMA-7B | 3 | 28.3 | 7.89 | **7.75** |
| LLaMA-7B | 4 | 6.29 | 6.09 | **5.97** |
| LLaMA-13B | 3 | 11.4 | 6.61 | **6.56** |
| LLaMA-30B | 3 | 7.56 | 5.10 | **5.08** |

### 4.2 Zero-shot 정확도

| 모델 | 정밀도 | WinoGrande | ARC | PIQA |
|------|--------|------------|-----|------|
| LLaMA-7B | FP16 | 69.9 | 51.0 | 78.7 |
| LLaMA-7B | SqueezeLLM 3bit | **68.4** | **49.0** | **77.6** |

### 4.3 속도 향상

A6000 GPU에서:

| 배치 | FP16 | SqueezeLLM 3bit |
|------|------|-----------------|
| 1 | 기준 | 1.9× |
| 8 | 기준 | 2.1× |

---

## 5. 구현 및 사용

```python
from squeezellm import SqueezeLLMForCausalLM

# 양자화된 모델 로드
model = SqueezeLLMForCausalLM.from_pretrained(
    "squeeze-ai-lab/squeezellm-llama-7b-3bit",
    device_map="auto"
)

# 추론
output = model.generate(input_ids, max_length=100)
```

---

## 6. 핵심 요약

### 기억해야 할 것들

1. **Dense-and-Sparse**: 민감한 값은 Sparse로 보존
2. **Non-uniform quantization**: k-means로 최적 중심점
3. **효과**: 3bit에서도 우수한 정확도

### vs 다른 방법들

| 방법 | Outlier 처리 | 양자화 방식 |
|------|-------------|-------------|
| GPTQ | 오류 보상 | Uniform |
| AWQ | 스케일링 | Uniform |
| **SqueezeLLM** | **Sparse 분리** | **Non-uniform** |

---

## 참고 자료

1. [SqueezeLLM 논문](https://arxiv.org/abs/2306.07629)
2. [GitHub](https://github.com/SqueezeAILab/SqueezeLLM)

---

*이전 리뷰: [SmoothQuant](./004_SmoothQuant.md)*
*다음 리뷰: [BitNet](./006_BitNet.md)*
