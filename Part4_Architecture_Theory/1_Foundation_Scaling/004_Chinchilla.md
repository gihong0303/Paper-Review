# Training Compute-Optimal Large Language Models (Chinchilla)

**논문 발표**: 2022년 (NeurIPS 2022)
**저자**: Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al.
**소속**: DeepMind
**논문 링크**: [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

---

## 한 줄 요약
> 기존 대형 모델들이 undertrained 상태임을 밝히고, 모델 크기와 데이터를 동일하게 스케일해야 한다는 새로운 최적 비율 제시

---

## 1. 핵심 발견: 기존 모델은 Undertrained

### 1.1 Kaplan vs Chinchilla

| 접근법 | 모델:데이터 비율 |
|--------|------------------|
| Kaplan (2020) | N ∝ C^0.73 |
| **Chinchilla** | **N ∝ C^0.50** |

### 1.2 최적 법칙

$$N_{opt} \approx D_{opt}$$

즉, **파라미터 수 ≈ 토큰 수** (대략 1:20)

---

## 2. Gopher vs Chinchilla

### 2.1 같은 컴퓨팅, 다른 배분

| 모델 | 파라미터 | 토큰 | 성능 |
|------|----------|------|------|
| Gopher | 280B | 300B | 기준 |
| **Chinchilla** | **70B** | **1.4T** | **더 좋음!** |

```
Chinchilla:
- 4배 작은 모델
- 4배 많은 데이터
- 더 좋은 성능!
```

### 2.2 MMLU 결과

| 모델 | MMLU |
|------|------|
| Gopher (280B) | 60.0% |
| **Chinchilla (70B)** | **67.5%** |

---

## 3. 수식

### 3.1 최적 할당

주어진 FLOP 예산 $C$:

$$N_{opt} = G \cdot \left(\frac{C}{6}\right)^a$$
$$D_{opt} = G^{-1} \cdot \left(\frac{C}{6}\right)^b$$

여기서 $a \approx b \approx 0.5$

### 3.2 실용적 법칙

$$D_{opt} \approx 20 \times N_{opt}$$

```python
def chinchilla_optimal(compute_flops):
    """Chinchilla 최적 계산"""
    # 대략적 계산
    N_opt = (compute_flops / 6) ** 0.5 / 20
    D_opt = 20 * N_opt
    return N_opt, D_opt

# 예: 10^24 FLOPs
N, D = chinchilla_optimal(1e24)
# N ≈ 67B, D ≈ 1.4T
```

---

## 4. 실무 영향

### 4.1 기존 모델 분석

| 모델 | 파라미터 | 토큰 | 최적 토큰 | 상태 |
|------|----------|------|-----------|------|
| GPT-3 | 175B | 300B | 3.5T | Undertrained |
| Gopher | 280B | 300B | 5.6T | Undertrained |
| LLaMA | 65B | 1.4T | 1.3T | **최적!** |

### 4.2 이후 모델들

```
Chinchilla 이후:
- LLaMA: 65B, 1.4T tokens
- Falcon: 40B, 1T tokens
- Mistral: 7B, 충분한 토큰

→ 더 작은 모델, 더 많은 데이터
```

---

## 5. 추론 비용 고려

### 5.1 학습 vs 추론

```
Chinchilla 최적 = 학습 비용만 고려

실제로는:
- 추론 비용이 더 클 수 있음
- 작은 모델이 추론에 유리
- 따라서 더 작게 만들고 더 많이 학습도 OK
```

### 5.2 LLaMA의 선택

```
LLaMA 7B: 1T tokens (140x 최적)
→ 추론 효율을 위해 의도적 over-training
```

---

## 6. 쉬운 예시

### 6.1 공부 비유

```
Kaplan: 똑똑한 사람을 짧게 공부시켜라
Chinchilla: 적당한 사람을 충분히 공부시켜라

결과:
- 적당히 똑똑하지만 많이 공부한 사람이 더 잘함
- 아무리 똑똑해도 공부 안 하면...
```

---

## 7. 핵심 요약

### 기억해야 할 것들

1. **핵심**: 기존 모델은 undertrained
2. **비율**: 토큰 ≈ 20 × 파라미터
3. **결과**: 4배 작은 Chinchilla > 4배 큰 Gopher
4. **영향**: LLaMA 등 이후 모델 설계 변경

### 핵심 수식

$$D_{opt} \approx 20 \times N_{opt}$$

### Chinchilla 체크리스트

- [ ] 토큰 수 ≥ 20 × 파라미터
- [ ] 추론 비용 고려 시 더 작게
- [ ] 품질 좋은 데이터 중요

---

## 참고 자료

1. [Chinchilla 논문](https://arxiv.org/abs/2203.15556)

---

*이전 리뷰: [Scaling Laws](./003_Scaling_Laws.md)*
*다음 리뷰: [LLaMA 2](./005_LLaMA_2.md)*
