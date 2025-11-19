# Part 4: LLM Architecture & Theory

대규모 언어 모델의 아키텍처와 이론적 기반을 다루는 핵심 논문들입니다.

## 섹션 구성

### 1. [Foundation & Scaling Laws](./1_Foundation_Scaling/)
Transformer의 탄생부터 스케일링 법칙까지

| 날짜 | 논문 | 핵심 내용 |
|------|------|-----------|
| 2025-06-13 | Attention Is All You Need | Transformer 아키텍처 |
| 2025-06-16 | GPT-3 | In-context Learning |
| 2025-06-19 | Scaling Laws | 스케일링 법칙 |
| 2025-06-22 | Chinchilla | Compute-Optimal 학습 |
| 2025-06-25 | LLaMA 2 | 오픈소스 표준 |
| 2025-06-28 | LLaMA 3 | 최신 학습 레시피 |

### 2. [Positional Embeddings & Attention Variants](./2_Positional_Attention/)
위치 인코딩과 어텐션 변형

| 날짜 | 논문 | 핵심 내용 |
|------|------|-----------|
| 2025-07-01 | RoPE (RoFormer) | Rotary Position Embedding |
| 2025-07-04 | ALiBi | Linear Bias 위치 인코딩 |
| 2025-07-07 | GQA | Grouped-Query Attention |
| 2025-07-10 | Sliding Window (Mistral) | 윈도우 어텐션 |
| 2025-07-13 | Ring Attention | 분산 무한 컨텍스트 |

### 3. [Mixture of Experts (MoE)](./3_MoE/)
Sparse 아키텍처의 발전

| 날짜 | 논문 | 핵심 내용 |
|------|------|-----------|
| 2025-07-16 | Switch Transformer | MoE 기초 |
| 2025-07-19 | Mixtral | Sparse MoE 대중화 |
| 2025-07-22 | DeepSeek-V2 | MLA + MoE |
| 2025-07-25 | DeepSeek-V3 | 최신 SOTA MoE |

### 4. [Post-Transformer Architectures](./4_Post_Transformer/)
Transformer를 넘어서

| 날짜 | 논문 | 핵심 내용 |
|------|------|-----------|
| 2025-07-28 | RWKV | RNN + Transformer |
| 2025-07-31 | Mamba | Selective SSM |
| 2025-08-03 | Jamba | Mamba + Transformer 하이브리드 |
| 2025-08-06 | Mamba-2 | SSM 이론 강화 |
| 2025-08-09 | Hyena | Convolution 기반 |

### 5. [Reasoning & Chain of Thought](./5_Reasoning_CoT/)
LLM의 추론 능력

| 날짜 | 논문 | 핵심 내용 |
|------|------|-----------|
| 2025-08-12 | Chain-of-Thought | CoT 프롬프팅 |
| 2025-08-15 | Self-Consistency | 다수결 추론 |
| 2025-08-18 | Tree of Thoughts | 트리 탐색 추론 |
| 2025-08-21 | Quiet-STaR | 내부 추론 학습 |

---

## 학습 로드맵

### 아키텍처 이해 순서

```
1. Transformer 기초
   └── Attention Is All You Need

2. 스케일링 이해
   └── Scaling Laws → Chinchilla

3. 위치 인코딩
   └── RoPE → ALiBi → Ring Attention

4. 효율적 어텐션
   └── GQA → Sliding Window

5. Sparse 아키텍처
   └── MoE → Mixtral → DeepSeek

6. Post-Transformer
   └── Mamba → Jamba

7. 추론 능력
   └── CoT → ToT
```

### 핵심 수식 미리보기

**Scaled Dot-Product Attention**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Chinchilla Optimal**:
$$N_{opt} \approx 20 \times D$$

**RoPE**:
$$f(x, m) = e^{im\theta}x$$

**Mamba Selection**:
$$y = \text{SSM}(A, B, C)(x)$$

---

## 참고

- 총 24개 논문
- 2025년 6월 ~ 8월 작성
- 각 리뷰는 수식, 코드, 직관적 설명 포함

---

*이전: [Part 3 - Domain Adaptation](../Part3_Domain_Adaptation/)*
