# DeepSeek-V2: A Strong, Economical, and Efficient MoE Model

**논문 발표**: 2024년
**저자**: DeepSeek-AI
**소속**: DeepSeek
**논문 링크**: [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
**공식 모델**: [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V2)

---

## 한 줄 요약
> MLA(Multi-head Latent Attention)와 DeepSeekMoE를 결합하여, 236B 파라미터에서 21B만 활성화하면서 GPT-4급 성능 달성

---

## 1. 두 가지 혁신

### 1.1 MLA (Multi-head Latent Attention)

```
기존 GQA 문제:
- KV 캐시 여전히 큼
- KV heads 수에 비례

MLA:
- KV를 저차원 latent로 압축
- KV 캐시 획기적 감소
```

### 1.2 DeepSeekMoE

```
Fine-grained experts:
- 더 많고 작은 experts
- 더 세밀한 전문화
```

---

## 2. MLA 상세

### 2.1 핵심 아이디어

$$c_{KV} = W_{DKV} h$$
$$k = W_{UK} c_{KV}, \quad v = W_{UV} c_{KV}$$

```
h → 저차원 c_KV → K, V 복원

KV 캐시: d_model × seq → d_c × seq
d_c << d_model → 큰 절약!
```

### 2.2 KV 캐시 비교

| 방법 | 캐시 크기 |
|------|-----------|
| MHA | 2 × h × d |
| GQA (8 groups) | 2 × 8 × d |
| **MLA** | **d_c** (예: 512) |

93.3% 감소!

---

## 3. DeepSeekMoE

### 3.1 Fine-grained Experts

```
Mixtral: 8 experts (큼)
DeepSeek: 160 experts (작음)

장점:
- 더 세밀한 전문화
- 유연한 조합
```

### 3.2 Shared Experts

```
일부 expert는 항상 활성화:
- 공통 지식 담당
- 나머지는 전문 지식

구조:
- 2 shared experts (항상)
- 160 routed experts (6개 선택)
```

---

## 4. 모델 구성

### 4.1 DeepSeek-V2 236B

| 항목 | 값 |
|------|-----|
| 총 파라미터 | 236B |
| 활성 파라미터 | 21B |
| Layers | 60 |
| d_model | 5120 |
| Experts | 160 + 2 shared |
| Top-k | 6 |

### 4.2 학습

```
8.1T tokens
Context: 128K
```

---

## 5. 실험 결과

### 5.1 벤치마크

| 모델 | 활성 | MMLU | HumanEval |
|------|------|------|-----------|
| Mixtral 8x22B | 39B | 77.8 | 45.1 |
| LLaMA 3 70B | 70B | 79.5 | 81.7 |
| **DeepSeek-V2** | **21B** | **78.5** | **81.1** |

3배 적은 연산으로 동급 성능!

### 5.2 비용 효율

```
학습 비용: $5.76M
API 가격: 100만 토큰당 $0.14

매우 경제적!
```

---

## 6. 핵심 요약

### 기억해야 할 것들

1. **MLA**: KV 캐시 93% 감소
2. **Fine-grained MoE**: 160 작은 experts
3. **효율**: 21B로 70B급 성능
4. **비용**: 매우 경제적

### 수식 요약

$$\text{Attention} = \text{softmax}\left(\frac{Q(W_{UK}c_{KV})^T}{\sqrt{d}}\right)W_{UV}c_{KV}$$

---

## 참고 자료

1. [DeepSeek-V2 논문](https://arxiv.org/abs/2405.04434)

---

*이전 리뷰: [Mixtral](./002_Mixtral.md)*
*다음 리뷰: [DeepSeek-V3](./004_DeepSeek-V3.md)*
