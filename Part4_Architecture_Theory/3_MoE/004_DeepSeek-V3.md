# DeepSeek-V3

**논문 발표**: 2024년 12월
**저자**: DeepSeek-AI
**소속**: DeepSeek
**논문 링크**: [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
**공식 모델**: [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3)

---

## 한 줄 요약
> 671B 파라미터에서 37B만 활성화하는 MoE로, $5.5M 학습 비용으로 GPT-4o와 Claude 3.5 Sonnet에 필적하는 오픈소스 최강 모델

---

## 1. 주요 혁신

### 1.1 Auxiliary-Loss-Free Load Balancing

```
기존 문제:
Auxiliary loss → 성능 저하

해결:
Bias term으로 load balancing
Loss 없이도 균형!
```

### 1.2 Multi-Token Prediction (MTP)

```
기존: 다음 1개 토큰 예측
MTP: 다음 여러 토큰 동시 예측

→ 더 효율적인 학습
→ Speculative decoding과 결합 가능
```

---

## 2. 모델 구성

### 2.1 아키텍처

| 항목 | 값 |
|------|-----|
| 총 파라미터 | 671B |
| 활성 파라미터 | 37B |
| Layers | 61 |
| d_model | 7168 |
| Experts | 256 + 1 shared |
| Top-k | 8 |

### 2.2 vs DeepSeek-V2

| 항목 | V2 | V3 |
|------|-----|-----|
| 총 파라미터 | 236B | 671B |
| 활성 파라미터 | 21B | 37B |
| Experts | 160 | 256 |

---

## 3. 학습

### 3.1 데이터

```
Pre-training: 14.8T tokens
Post-training: SFT + RL

Context: 128K
```

### 3.2 비용

```
총 비용: $5.576M
- H800 GPU: 2.788M hours
- 2048 GPUs × 2달

믿기 어려울 정도로 저렴!
(GPT-4 추정: $100M+)
```

### 3.3 FP8 Training

```
FP8 mixed precision:
- 메모리 절약
- 속도 향상
- 품질 유지
```

---

## 4. Auxiliary-Loss-Free Balancing

### 4.1 기존 방식

```python
# Auxiliary loss
L_aux = α * load_balancing_loss()
L_total = L_main + L_aux

문제: α 조정 어려움, 성능 저하
```

### 4.2 DeepSeek-V3 방식

```python
# Bias-based balancing
router_logits = W_r @ x + bias

# bias를 동적으로 조정
if expert_i 과부하:
    bias[i] -= δ
elif expert_i 유휴:
    bias[i] += δ

# Loss 없이 균형!
```

---

## 5. 실험 결과

### 5.1 벤치마크

| 모델 | MMLU | MATH | HumanEval |
|------|------|------|-----------|
| GPT-4o | 88.7 | 76.6 | 90.2 |
| Claude 3.5 | 88.7 | 78.3 | 92.0 |
| **DeepSeek-V3** | **88.5** | **90.2** | **92.7** |

**MATH에서 압도적!**

### 5.2 코딩

```
LiveCodeBench:
- DeepSeek-V3: 40.5%
- GPT-4o: 33.4%
- Claude 3.5: 38.9%

코딩 최강!
```

### 5.3 중국어

```
C-Eval: 86.5%
CMMLU: 88.2%

다국어 성능도 우수
```

---

## 6. 핵심 요약

### 기억해야 할 것들

1. **규모**: 671B total, 37B active
2. **비용**: $5.5M (20배+ 저렴)
3. **성능**: GPT-4o/Claude 3.5급
4. **혁신**: Aux-loss-free balancing, MTP

### 의의

```
오픈소스 LLM의 새 시대:
- 최고 성능
- 최저 비용
- 완전 공개

연구/개발 접근성 혁명!
```

---

## 참고 자료

1. [DeepSeek-V3 논문](https://arxiv.org/abs/2412.19437)
2. [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3)

---

*이전 리뷰: [DeepSeek-V2](./003_DeepSeek-V2.md)*
*다음 섹션: [Post-Transformer](../4_Post_Transformer/)*
