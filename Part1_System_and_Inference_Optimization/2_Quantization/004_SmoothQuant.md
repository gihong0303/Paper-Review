# SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models

**논문 발표**: 2022년 (ICML 2023)
**저자**: Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han
**소속**: MIT, NVIDIA
**논문 링크**: [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
**공식 구현**: [GitHub](https://github.com/mit-han-lab/smoothquant)

---

## 한 줄 요약
> 활성화의 양자화 어려움을 가중치로 이전(migration)하여, LLM의 활성화와 가중치 모두를 INT8로 양자화하고 1.56배 속도 향상과 2배 메모리 절약을 달성

---

## 1. 문제 정의: 활성화 양자화의 어려움

### 1.1 가중치 vs 활성화 양자화

| 구분 | 가중치 | 활성화 |
|------|--------|--------|
| 특성 | 정적 (고정) | 동적 (입력마다 변함) |
| 분포 | 균일 | **Outlier 많음** |
| 양자화 | 쉬움 | **어려움** |

### 1.2 활성화의 Outlier 문제

LLM의 활성화는 **극단적인 outlier**를 포함:

```
일반적인 값: [-1, 0.5, -0.3, 0.8, ...]
Outlier:     [..., 60, ..., -50, ...]

분포 범위: 정상 값의 100배 이상
```

### 1.3 Outlier가 양자화에 미치는 영향

```python
# 예: [-0.5, 0.3, 60.0, -0.1]
# max = 60.0
# scale = 127 / 60.0 = 2.12

# INT8 양자화
X_int8 = round(X * 2.12)
       = [-1, 1, 127, 0]

# 대부분의 값이 0이나 1로!
# 심각한 정밀도 손실
```

---

## 2. 핵심 통찰: 양자화 어려움의 이전

### 2.1 관찰

```
활성화 X:  양자화 어려움 (outlier 있음)
가중치 W:  양자화 쉬움 (균일 분포)
```

### 2.2 아이디어

> "활성화의 양자화 어려움을 가중치로 옮기자!"

수학적으로:
$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X} \cdot \hat{W}$$

- $\hat{X} = X \cdot s^{-1}$: 스케일 다운 → 양자화 쉬워짐
- $\hat{W} = s \cdot W$: 스케일 업 → 약간 어려워지지만 OK

### 2.3 균형 잡기

```
변환 전:
X: [outlier가 많음, 양자화 어려움]
W: [균일, 양자화 쉬움]

변환 후:
X̂: [outlier 줄어듦, 양자화 쉬워짐]
Ŵ: [약간 불균일, 여전히 양자화 가능]
```

---

## 3. SmoothQuant 알고리즘

### 3.1 스케일 s 계산

채널별 스케일:

$$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$$

여기서:
- $\max(|X_j|)$: j번째 채널의 활성화 최댓값
- $\max(|W_j|)$: j번째 채널의 가중치 최댓값
- $\alpha$: migration strength (0.5가 기본)

### 3.2 α의 의미

```
α = 0:    W만 변환 (활성화 양자화 여전히 어려움)
α = 0.5:  균형 (권장)
α = 1:    X만 변환 (가중치 양자화 어려워짐)
```

### 3.3 시각화

```
원본:
X: ████████████████░░░░  (range: 60)
W: ████░░░░░░░░░░░░░░░░  (range: 2)
→ X 양자화 어려움

α = 0.5 적용:
X̂: ██████████░░░░░░░░░░  (range: 11)
Ŵ: ██████████░░░░░░░░░░  (range: 11)
→ 둘 다 양자화 가능!
```

---

## 4. 수학적 상세

### 4.1 양자화 오류 분석

양자화 오류:
$$\epsilon = Q(X) \cdot Q(W) - X \cdot W$$

스케일링 후:
$$\epsilon' = Q(\hat{X}) \cdot Q(\hat{W}) - X \cdot W$$

적절한 s를 선택하면 $\epsilon' < \epsilon$

### 4.2 등가 변환 증명

$$Y = X \cdot W = (X \cdot s^{-1}) \cdot (s \cdot W) = \hat{X} \cdot \hat{W}$$

스케일 s를 앞 레이어에 흡수:

```python
# Linear: Y = X @ W
# 변환 후: Y = (X @ s_inv) @ (s @ W)

# LayerNorm → Linear 구조에서:
# LayerNorm의 gamma에 s_inv 흡수
layernorm.weight = layernorm.weight * s_inv

# Linear의 W에 s 흡수
linear.weight = s.unsqueeze(0) * linear.weight
```

### 4.3 알고리즘 전체

```
알고리즘: SmoothQuant
─────────────────────────
입력: 모델 M, calibration 데이터 D, α
출력: INT8 양자화 모델

1. for each Transformer block:
2.     # 활성화 수집
3.     X = collect_activations(D)
4.
5.     # 채널별 스케일 계산
6.     for j in channels:
7.         s[j] = (max|X_j|)^α / (max|W_j|)^(1-α)
8.
9.     # 스케일 적용 및 흡수
10.    layernorm.weight *= s_inv
11.    linear.weight *= s
12.
13.    # INT8 양자화
14.    linear.weight = INT8(linear.weight)
15.
16. end for

# 추론 시: 활성화도 동적으로 INT8 양자화
```

---

## 5. 실제 구현

### 5.1 스케일 흡수 위치

```
Transformer Block 구조:

LayerNorm1 → Attention → LayerNorm2 → FFN
    │                         │
    ↓                         ↓
  s 흡수                     s 흡수

세부:
- QKV projection 전: LayerNorm1에 흡수
- FFN 전: LayerNorm2에 흡수
```

### 5.2 적용 대상

SmoothQuant를 적용할 연산:
- Attention의 QKV projection
- Attention의 output projection
- FFN의 모든 linear 레이어

### 5.3 코드 예시

```python
import torch
from smoothquant import smooth_lm

# 모델 로드
model = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")

# Calibration 데이터
calibration_samples = [
    tokenizer("Example text", return_tensors="pt")
    for _ in range(128)
]

# SmoothQuant 적용
model = smooth_lm(
    model,
    calibration_samples,
    alpha=0.5  # migration strength
)

# INT8 양자화 추론
# (INT8 커널 필요)
```

---

## 6. 쉬운 예시로 이해하기

### 6.1 시소 균형 비유

```
원래 상태:
활성화 쪽: [무거운 짐] ─┬─ [가벼운 짐] :가중치 쪽
                       │
                    불균형!

SmoothQuant:
활성화 쪽: [중간 짐] ─┬─ [중간 짐] :가중치 쪽
                     │
                   균형!
```

### 6.2 물 분배 비유

두 용기에 물을 담아야 함 (각 용기 용량 제한):

**원래 상태**:
- 활성화 용기: 100L (용량 50L → 넘침!)
- 가중치 용기: 10L (용량 50L → 여유)

**SmoothQuant**:
- 활성화 용기: 30L (OK)
- 가중치 용기: 30L (OK)
- 총량은 동일, 분배만 변경

### 6.3 숫자 예시

```python
# 원본
X = [0.5, -0.3, 60.0, 0.2]  # outlier: 60
W = [0.8, 0.5, 0.3, 0.6]    # 균일

# 채널 3의 스케일
# s = (60)^0.5 / (0.3)^0.5 = 7.75 / 0.55 = 14.1

# 변환 후
X_hat = [0.5, -0.3, 60/14.1, 0.2] = [0.5, -0.3, 4.3, 0.2]
W_hat = [0.8, 0.5, 0.3*14.1, 0.6] = [0.8, 0.5, 4.2, 0.6]

# 이제 둘 다 비슷한 범위 → INT8 양자화 가능!
```

---

## 7. W8A8 추론

### 7.1 INT8 GEMM의 장점

```
FP16 GEMM:   메모리 대역폭 2×, 연산 1×
INT8 GEMM:   메모리 대역폭 1×, 연산 2× (Tensor Core)

→ INT8이 더 빠름!
```

### 7.2 Tensor Core 활용

NVIDIA GPU의 INT8 Tensor Core:
- A100: INT8 624 TOPS vs FP16 312 TFLOPS
- INT8이 2배 빠름

### 7.3 구현

```python
# FP16 추론
Y = X @ W  # FP16 × FP16

# SmoothQuant INT8 추론
X_int8 = quantize(X_smooth)
W_int8 = quantize(W_smooth)
Y_int32 = X_int8 @ W_int8  # INT8 × INT8 → INT32
Y = dequantize(Y_int32)
```

---

## 8. 실험 결과

### 8.1 정확도 (OPT 모델)

| 모델 | FP16 | W8A8 (naive) | **W8A8 (SmoothQuant)** |
|------|------|--------------|------------------------|
| OPT-6.7B | 10.86 | 발산 | **10.91** |
| OPT-13B | 10.13 | 발산 | **10.19** |
| OPT-30B | 9.56 | 발산 | **9.59** |
| OPT-66B | 9.34 | 발산 | **9.36** |
| OPT-175B | 8.34 | 발산 | **8.42** |

**핵심**: Naive W8A8은 발산하지만, SmoothQuant는 FP16과 비슷

### 8.2 다양한 모델에서의 결과

| 모델 | FP16 PPL | SmoothQuant PPL | 차이 |
|------|----------|-----------------|------|
| BLOOM-176B | 8.11 | 8.24 | +0.13 |
| GLM-130B | 9.02 | 9.15 | +0.13 |
| LLaMA-65B | 3.53 | 3.59 | +0.06 |

### 8.3 속도 향상

A100 GPU에서 OPT-175B:

| 측면 | FP16 | SmoothQuant W8A8 |
|------|------|------------------|
| 지연시간 | 기준 | **0.64×** (1.56× 빠름) |
| 처리량 | 기준 | **1.56×** |

### 8.4 메모리 절약

| 모델 | FP16 | SmoothQuant W8A8 | 절약 |
|------|------|------------------|------|
| OPT-30B | 60GB | 30GB | **2×** |
| OPT-175B | 326GB | 163GB | **2×** |

---

## 9. 다른 방법들과의 비교

### 9.1 vs LLM.int8()

| 측면 | LLM.int8() | SmoothQuant |
|------|------------|-------------|
| 가중치 | INT8 | INT8 |
| 활성화 | FP16 (outlier만) | **INT8 (전체)** |
| 속도 | 느림 (혼합 정밀도) | **빠름** (순수 INT8) |
| 정확도 | 좋음 | 좋음 |

### 9.2 vs ZeroQuant

| 측면 | ZeroQuant | SmoothQuant |
|------|-----------|-------------|
| 방법 | 토큰별 양자화 | 스무딩 + 채널별 양자화 |
| 정확도 | 낮음 | **높음** |
| 속도 | 보통 | **빠름** |

### 9.3 vs AWQ/GPTQ

| 측면 | AWQ/GPTQ | SmoothQuant |
|------|----------|-------------|
| 타겟 | 가중치만 | 가중치 + **활성화** |
| 비트 | 4bit | 8bit |
| 속도 이점 | 메모리 | 메모리 + **연산** |

---

## 10. α 값 선택

### 10.1 모델별 최적 α

| 모델 | 권장 α |
|------|--------|
| OPT | 0.5 |
| BLOOM | 0.5 |
| GLM-130B | 0.75 |
| LLaMA | 0.5-0.65 |

### 10.2 α 결정 방법

```python
def find_optimal_alpha(model, calibration_data):
    """
    그리드 서치로 최적 α 찾기
    """
    best_alpha = 0.5
    best_ppl = float('inf')

    for alpha in [0.25, 0.5, 0.75, 0.9]:
        smoothed_model = smooth_lm(model, calibration_data, alpha)
        quantized_model = int8_quantize(smoothed_model)
        ppl = evaluate_perplexity(quantized_model)

        if ppl < best_ppl:
            best_ppl = ppl
            best_alpha = alpha

    return best_alpha
```

### 10.3 레이어별 α

더 정밀한 접근: 각 레이어마다 다른 α

```python
# 레이어별 최적화
for layer_idx, layer in enumerate(model.layers):
    alpha = find_optimal_alpha_for_layer(layer, calibration_data)
    smooth_layer(layer, alpha)
```

---

## 11. 한계점 및 후속 연구

### 11.1 한계점

1. **8bit 한정**:
   - 4bit 적용 어려움 (AWQ/GPTQ 필요)

2. **레이어별 최적화**:
   - 전역 α가 모든 레이어에 최적 아님

3. **동적 양자화 오버헤드**:
   - 활성화는 런타임에 양자화

### 11.2 후속 연구

- **Atom** (2023): SmoothQuant + 4bit 가중치
- **QLLM** (2023): 적응적 smoothing
- **OmniQuant** (2024): 학습 기반 smoothing

---

## 12. 핵심 요약

### 기억해야 할 것들

1. **핵심 통찰**: 활성화의 양자화 어려움을 가중치로 이전
2. **방법**: Per-channel scaling (migration)
3. **결과**: W8A8 양자화로 1.56× 속도, 2× 메모리 절약
4. **핵심 공식**: $s_j = \frac{\max|X_j|^\alpha}{\max|W_j|^{1-\alpha}}$

### 다른 방법과의 관계

```
LLM.int8(): Outlier를 FP16으로 (혼합 정밀도)
AWQ:        활성화 기반 가중치 스케일링 (가중치만)
SmoothQuant: 양자화 어려움 이전 (가중치 + 활성화)
```

### 실무 적용

```python
# SmoothQuant + FasterTransformer/TensorRT-LLM
from smoothquant import smooth_lm

model = smooth_lm(model, calibration_data, alpha=0.5)
# → FasterTransformer의 INT8 커널로 추론
```

---

## 참고 자료

1. [SmoothQuant 논문](https://arxiv.org/abs/2211.10438)
2. [공식 GitHub](https://github.com/mit-han-lab/smoothquant)
3. [Song Han 발표](https://www.youtube.com/watch?v=U0yvqjhMfr4)
4. [MIT HAN Lab](https://hanlab.mit.edu/)

---

*이전 리뷰: [AWQ](./003_AWQ.md)*
*다음 리뷰: [SqueezeLLM](./005_SqueezeLLM.md)*
