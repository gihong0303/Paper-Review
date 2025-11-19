# Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients

**논문 발표**: 2024년
**저자**: Zhenyu Zhang, Ajay Jaiswal, Lu Yin, Shiwei Liu, Jiawei Zhao, Yuandong Tian, Zhangyang Wang
**소속**: UT Austin, Meta, University of Liverpool
**논문 링크**: [arXiv:2407.08296](https://arxiv.org/abs/2407.08296)

---

## 한 줄 요약
> GaLore의 투영 행렬과 가중치를 INT4로 양자화하고, 레이어별 적응형 rank를 사용하여, 단일 24GB GPU에서 LLaMA-7B를 Full Precision GaLore보다 50% 적은 메모리로 학습

---

## 1. 배경: GaLore란?

### 1.1 GaLore (Gradient Low-Rank Projection)

메모리 효율적 학습을 위한 기법:
- Gradient를 저차원 부분공간으로 투영
- Optimizer state 크기 대폭 감소

$$G_{\text{proj}} = P_L^\top G P_R$$

여기서 $P_L, P_R$은 투영 행렬

### 1.2 GaLore의 메모리 사용

```
전체 학습 메모리:
= 가중치 + Gradient + Optimizer state

GaLore:
= 가중치 + 투영된 Gradient + 작은 Optimizer state
```

---

## 2. Q-GaLore의 핵심 개선

### 2.1 양자화 적용 대상

```
가중치 W:     FP16 → INT4
투영 행렬 P:  FP16 → INT4
Gradient:    FP16 유지 (정확도)
```

### 2.2 INT4 Quantization

```python
def quantize_weight(W):
    # Group-wise INT4 양자화
    return gptq_quantize(W, bits=4, group_size=128)

def quantize_projection(P):
    # 투영 행렬도 INT4로
    return uniform_quantize(P, bits=4)
```

### 2.3 Layer-Adaptive Rank

각 레이어의 최적 rank를 자동 결정:

```python
def adaptive_rank(layer_idx, total_layers):
    # 하위 레이어: 낮은 rank
    # 상위 레이어: 높은 rank
    if layer_idx < total_layers // 3:
        return 128
    elif layer_idx < 2 * total_layers // 3:
        return 256
    else:
        return 512
```

---

## 3. 알고리즘

### 3.1 학습 과정

```
알고리즘: Q-GaLore Training
───────────────────────────
1. 가중치 W를 INT4로 양자화
2. 투영 행렬 P를 INT4로 초기화

3. for each iteration:
4.     W_fp = dequantize(W_int4)
5.     G = compute_gradient(W_fp)
6.
7.     # 저차원 투영
8.     P_fp = dequantize(P_int4)
9.     G_proj = P_fp.T @ G @ P_fp
10.
11.    # Optimizer 업데이트
12.    update = optimizer.step(G_proj)
13.
14.    # 역투영 및 업데이트
15.    W_fp -= P_fp @ update @ P_fp.T
16.    W_int4 = quantize(W_fp)
17.
18.    # 주기적 투영 갱신
19.    if iter % T == 0:
20.        P_int4 = update_projection(G)
```

---

## 4. 실험 결과

### 4.1 LLaMA Pre-training

C4 데이터셋에서 학습:

| 방법 | 메모리 | PPL |
|------|--------|-----|
| Full Precision | 58 GB | 14.2 |
| GaLore | 22 GB | 14.3 |
| **Q-GaLore** | **11 GB** | **14.5** |

### 4.2 Fine-tuning

| 모델 | 방법 | 메모리 | 정확도 |
|------|------|--------|--------|
| LLaMA-7B | LoRA | 18 GB | 기준 |
| LLaMA-7B | GaLore | 16 GB | +0.5% |
| LLaMA-7B | **Q-GaLore** | **8 GB** | **+0.3%** |

### 4.3 단일 GPU에서 학습

24GB GPU (RTX 3090/4090)에서:

| 모델 | Full Precision | GaLore | Q-GaLore |
|------|----------------|--------|----------|
| 7B | OOM | 가능 | **가능** |
| 13B | OOM | OOM | **가능** |

---

## 5. 핵심 요약

### 기억해야 할 것들

1. **핵심**: GaLore + INT4 양자화
2. **적용**: 가중치 + 투영 행렬 양자화
3. **결과**: 50% 메모리 추가 절감
4. **활용**: 소비자 GPU에서 7B 학습 가능

### vs 다른 방법들

| 방법 | 타겟 | 메모리 절감 |
|------|------|-------------|
| LoRA | Fine-tuning | 중간 |
| QLoRA | Fine-tuning | 높음 |
| GaLore | Pre-training | 중간 |
| **Q-GaLore** | **Pre-training** | **최고** |

---

## 참고 자료

1. [Q-GaLore 논문](https://arxiv.org/abs/2407.08296)
2. [GaLore 논문](https://arxiv.org/abs/2403.03507)

---

*이전 리뷰: [BitNet b1.58](./007_BitNet_b158.md)*
*다음 섹션: [Speculative Decoding](../3_Speculative_Decoding/)*
