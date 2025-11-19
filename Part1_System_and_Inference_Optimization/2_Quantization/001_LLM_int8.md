# LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale

**논문 발표**: 2022년 (NeurIPS 2022)
**저자**: Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer
**소속**: University of Washington, Meta AI
**논문 링크**: [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
**공식 구현**: [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

---

## 한 줄 요약
> 대규모 Transformer에서 나타나는 극단적인 활성화 값(outlier)을 별도로 처리하는 Mixed-precision 분해를 통해, 175B 파라미터 모델까지 성능 저하 없이 INT8 추론을 가능하게 함

---

## 1. 배경: 양자화(Quantization)란?

### 1.1 기본 개념

양자화: 높은 정밀도 숫자를 낮은 정밀도로 변환

```
FP32 (32비트)  → INT8 (8비트)
메모리 4배 절약, 연산 2-4배 빠름
```

### 1.2 양자화 수식

**Absmax Quantization** (대칭 양자화):
$$X_{\text{int8}} = \text{round}\left(\frac{127}{\max(|X|)} \cdot X\right)$$

**역양자화**:
$$X_{\text{dequant}} = \frac{\max(|X|)}{127} \cdot X_{\text{int8}}$$

### 1.3 예시

```python
# FP16 값: [0.5, -1.2, 3.0, -0.8]
# max(|X|) = 3.0
# scale = 127 / 3.0 = 42.33

# 양자화
X_int8 = round([0.5, -1.2, 3.0, -0.8] * 42.33)
      = [21, -51, 127, -34]

# 역양자화
X_dequant = [21, -51, 127, -34] / 42.33
          = [0.496, -1.205, 3.0, -0.803]

# 양자화 오류: 미미함
```

---

## 2. 문제: 대규모 모델의 Emergent Features

### 2.1 관찰: Outlier의 출현

6.7B 파라미터를 넘어서면 **극단적인 활성화 값(outlier)** 이 나타남:

```
모델 크기별 활성화 분포:

125M: [-5 .......... 0 .......... 5]  정규 분포

6.7B: [-10 ........ 0 ........ 10]   약간 넓어짐

13B:  [-50 ...... 0 ...... 50]       outlier 출현
               ↓
            [..........0.........] 대부분의 값
      [-60]                    [60] 극소수의 outlier
```

### 2.2 Outlier의 특징

1. **체계적 (Systematic)**: 특정 차원에서 일관되게 발생
2. **중요함**: 제거 시 성능 급격히 저하
3. **극단적**: 정상 값의 20배 이상

```python
# 실제 관찰된 예시
normal_values = [-0.5, 0.3, -0.1, 0.2, ...]  # 대부분
outlier_value = 60.0  # 특정 차원에서 모든 토큰에 나타남
```

### 2.3 Outlier가 양자화에 미치는 영향

```python
# 예: [0.5, -0.3, 60.0, 0.1]
# max(|X|) = 60.0
# scale = 127 / 60.0 = 2.12

# 양자화
X_int8 = round([0.5, -0.3, 60.0, 0.1] * 2.12)
      = [1, -1, 127, 0]  # 대부분의 값이 0이나 1로!

# 심각한 정보 손실!
```

**문제**: Outlier로 인해 scale이 커지면, 정상 값들이 모두 0에 가까워짐

---

## 3. LLM.int8()의 해결책: Mixed-precision Decomposition

### 3.1 핵심 아이디어

> "Outlier는 FP16으로, 나머지는 INT8로 처리하자!"

```
행렬 X를 두 부분으로 분해:
X = X_normal + X_outlier

X_normal (99.9%): INT8로 양자화
X_outlier (0.1%): FP16 유지
```

### 3.2 알고리즘 개요

```
┌─────────────────────────────────────────────┐
│        LLM.int8() 행렬 곱셈                  │
├─────────────────────────────────────────────┤
│                                             │
│  Input: X[M, K] @ W[K, N]                   │
│                                             │
│  1. Outlier 차원 식별                        │
│     outlier_dims = {i : max(|X[:, i]|) > τ} │
│                                             │
│  2. 행렬 분해                                │
│     X_fp16 = X[:, outlier_dims]             │
│     W_fp16 = W[outlier_dims, :]             │
│     X_int8 = X[:, ~outlier_dims]            │
│     W_int8 = W[~outlier_dims, :]            │
│                                             │
│  3. 분리 계산                                │
│     Y_fp16 = X_fp16 @ W_fp16  (FP16)        │
│     Y_int8 = X_int8 @ W_int8  (INT8)        │
│                                             │
│  4. 결과 합산                                │
│     Y = Y_fp16 + dequant(Y_int8)            │
│                                             │
└─────────────────────────────────────────────┘
```

### 3.3 시각화

```
원본 행렬:           분해 후:

┌─────────────┐     ┌─────────────┐  ┌───┐
│ . . O . . . │     │ . . . . . . │  │ O │
│ . . O . . . │  →  │ . . . . . . │ +│ O │
│ . . O . . . │     │ . . . . . . │  │ O │
└─────────────┘     └─────────────┘  └───┘
    전체 X            X_int8 (INT8)   X_fp16
                                     (FP16)

O = Outlier 차원 (0.1%)
. = 정상 차원 (99.9%)
```

---

## 4. Vector-wise Quantization

### 4.1 기존 방식의 한계

**Tensor-wise**: 전체 텐서에 하나의 scale
- 문제: 한 행에 큰 값이 있으면 다른 행도 영향받음

**Row-wise/Column-wise**: 행/열마다 다른 scale
- 문제: 행렬 곱 후 dequantization이 복잡

### 4.2 Vector-wise 양자화

행과 열 각각에 scale 적용:

```python
# X @ W 계산

# X는 행별로 양자화
c_x = 127 / max(|X[i, :]|) for each row i
X_int8[i, :] = round(c_x[i] * X[i, :])

# W는 열별로 양자화
c_w = 127 / max(|W[:, j]|) for each column j
W_int8[:, j] = round(c_w[j] * W[:, j])

# INT8 행렬 곱
Y_int32 = X_int8 @ W_int8

# 역양자화: outer product로 scale 적용
Y = Y_int32 / (c_x[:, None] * c_w[None, :])
```

### 4.3 장점

```
정확도:  Tensor-wise < Row-wise < Vector-wise

Vector-wise가 가장 정확하면서도
GPU에서 효율적으로 구현 가능
```

---

## 5. Threshold 결정

### 5.1 Outlier 탐지 기준

$$\text{outlier dimension } i \text{ if } \max_j |X_{j,i}| > \tau$$

기본 threshold τ = 6.0

### 5.2 왜 6.0인가?

```
실험적 관찰:
- τ < 6: 너무 많은 차원이 FP16 → 메모리 이점 감소
- τ > 6: 일부 outlier 누락 → 성능 저하

최적점: τ = 6.0 (0.1% 정도의 차원)
```

### 5.3 동적 탐지

```python
def find_outlier_dims(X, threshold=6.0):
    """
    입력 활성화에서 outlier 차원 탐지
    """
    # 각 차원의 최대 절댓값
    col_max = X.abs().max(dim=0).values

    # Threshold 초과 차원
    outlier_mask = col_max > threshold

    return outlier_mask
```

---

## 6. 쉬운 예시로 이해하기

### 6.1 학생 점수 비유

반 학생들의 점수를 1-10 스케일로 저장하려 함

**기존 양자화**:
- 대부분 학생: 60-90점
- 한 학생: 1000점 (outlier)
- 스케일: 1000점 기준
- 결과: 대부분 학생이 0이나 1점으로 (정보 손실)

**LLM.int8()**:
- 1000점 학생: 별도 기록 (원본 그대로)
- 나머지 학생: 1-10 스케일로 저장
- 결과: 모든 학생 정보 보존

### 6.2 사진 압축 비유

**기존**:
- 전체 사진을 같은 품질로 압축
- 밝은 부분 때문에 어두운 부분 정보 손실

**LLM.int8()**:
- 매우 밝은 픽셀: 원본 유지
- 나머지: 압축
- 결과: 전체 이미지 품질 유지

---

## 7. 수학적 정당성

### 7.1 오차 분석

양자화 오차:
$$\epsilon = X - \text{dequant}(\text{quant}(X))$$

**문제**: Outlier가 있으면 $\epsilon$이 커짐 (dynamic range 문제)

**해결**: Outlier 분리로 각 부분의 dynamic range 축소
$$\epsilon_{\text{int8}} \approx 0 \text{ (정상 값들만)}$$
$$\epsilon_{\text{fp16}} = 0 \text{ (원본 유지)}$$

### 7.2 왜 Outlier가 체계적인가?

**가설**: Outlier 차원이 특별한 의미를 인코딩
- 문장 종료 신호
- 언어/도메인 정보
- 기타 전역 정보

**증거**: 같은 차원이 모든 토큰에서 outlier

---

## 8. 구현 세부사항

### 8.1 CUDA 커널

```cuda
__global__ void int8_mm_with_extraction(
    int8_t* X_int8, float* X_fp16,
    int8_t* W_int8, float* W_fp16,
    float* Y,
    bool* outlier_mask,
    float* scales_x, float* scales_w
) {
    // 1. INT8 행렬 곱
    int32_t acc_int = 0;
    for (int k = 0; k < K; k++) {
        if (!outlier_mask[k]) {
            acc_int += X_int8[row][k] * W_int8[k][col];
        }
    }

    // 2. FP16 부분 계산
    float acc_fp = 0;
    for (int k = 0; k < K; k++) {
        if (outlier_mask[k]) {
            acc_fp += X_fp16[row][k] * W_fp16[k][col];
        }
    }

    // 3. 역양자화 및 합산
    float result = acc_int / (scales_x[row] * scales_w[col]);
    result += acc_fp;

    Y[row][col] = result;
}
```

### 8.2 HuggingFace 통합

```python
from transformers import AutoModelForCausalLM
import torch

# 8bit 양자화 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    load_in_8bit=True,
    device_map="auto"
)

# 추론
output = model.generate(input_ids, max_length=100)
```

### 8.3 메모리 절약

| 모델 | FP16 메모리 | INT8 메모리 | 절약 |
|------|-------------|-------------|------|
| 13B | 26 GB | 13 GB | 50% |
| 66B | 132 GB | 66 GB | 50% |
| 175B | 350 GB | 175 GB | 50% |

---

## 9. 실험 결과

### 9.1 성능 비교 (Perplexity)

| 모델 | FP16 | INT8 (naive) | LLM.int8() |
|------|------|--------------|------------|
| OPT-6.7B | 10.86 | 10.92 | **10.86** |
| OPT-13B | 10.13 | 발산 | **10.13** |
| OPT-66B | 9.34 | 발산 | **9.37** |
| OPT-175B | 8.34 | 발산 | **8.41** |

**관찰**:
- 6.7B 이하: 단순 양자화도 OK
- 13B 이상: LLM.int8() 필수

### 9.2 Zero-shot 태스크

| 태스크 | FP16 | LLM.int8() | 차이 |
|--------|------|------------|------|
| LAMBADA | 76.2% | 76.0% | -0.2% |
| HellaSwag | 78.9% | 78.7% | -0.2% |
| WinoGrande | 70.1% | 70.0% | -0.1% |

**결론**: 사실상 성능 저하 없음

### 9.3 속도 비교

| 작업 | FP16 | LLM.int8() | 차이 |
|------|------|------------|------|
| 행렬 곱 | 기준 | 0.95× | -5% |
| 전체 추론 | 기준 | 1.0× | 동일 |

**참고**: 메모리 이동 오버헤드로 INT8이 살짝 느림
하지만 큰 모델을 로드할 수 있게 됨이 더 중요

---

## 10. 한계점 및 후속 연구

### 10.1 한계점

1. **메모리 절약만 50%**:
   - 4bit 양자화 대비 적은 절약
   - GPTQ, AWQ가 더 공격적 양자화

2. **추론 속도**:
   - FP16 대비 빠르지 않음 (오버헤드)
   - Tensor Core 활용 어려움

3. **학습 미지원**:
   - 추론만 가능
   - QLoRA가 학습 지원

### 10.2 후속 연구

- **GPTQ** (2022): 4bit까지 양자화
- **AWQ** (2023): Activation-aware 양자화
- **QLoRA** (2023): 4bit에서 학습 가능

---

## 11. 핵심 요약

### 기억해야 할 것들

1. **핵심 발견**: 대규모 모델(6B+)에서 outlier 출현
2. **해결책**: Mixed-precision decomposition
3. **결과**: 175B까지 성능 저하 없이 INT8 추론
4. **장점**: 메모리 50% 절약, 즉시 사용 가능

### Outlier 처리의 중요성

```
크기   | Outlier | 단순 INT8 | LLM.int8()
-------|---------|-----------|----------
<6.7B  | 없음    | OK        | OK
13B+   | 있음    | 발산!     | OK
```

### 실무 적용

```python
# bitsandbytes 설치
# pip install bitsandbytes

from transformers import AutoModelForCausalLM

# 8bit 로드
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-7b1",
    load_in_8bit=True,
    device_map="auto"
)

# 일반적으로 사용
outputs = model.generate(inputs)
```

---

## 참고 자료

1. [LLM.int8() 논문](https://arxiv.org/abs/2208.07339)
2. [bitsandbytes GitHub](https://github.com/TimDettmers/bitsandbytes)
3. [Tim Dettmers 블로그](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/)
4. [HuggingFace 문서](https://huggingface.co/docs/transformers/main/en/quantization)

---

*이전 섹션: [Kernel & Attention Optimization](../1_Kernel_and_Attention_Optimization/)*
*다음 리뷰: [GPTQ](./002_GPTQ.md)*
