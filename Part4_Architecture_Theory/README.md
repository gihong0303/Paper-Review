# Part 4: LLM Architecture & Theory

대규모 언어 모델의 아키텍처와 이론적 기반을 다루는 핵심 논문들입니다.

## 섹션 구성

### 1. [Foundation & Scaling Laws](./1_Foundation_Scaling/)
Transformer의 탄생부터 스케일링 법칙까지

| 논문 | 핵심 내용 |
|------|-----------|
| Attention Is All You Need | Transformer 아키텍처 |
| GPT-3 | In-context Learning |
| Scaling Laws | 스케일링 법칙 |
| Chinchilla | Compute-Optimal 학습 |
| LLaMA 2 | 오픈소스 표준 |
| LLaMA 3 | 최신 학습 레시피 |

### 2. [Positional Embeddings & Attention Variants](./2_Positional_Attention/)
위치 인코딩과 어텐션 변형

| 논문 | 핵심 내용 |
|------|-----------|
| RoPE (RoFormer) | Rotary Position Embedding |
| ALiBi | Linear Bias 위치 인코딩 |
| GQA | Grouped-Query Attention |
| Sliding Window (Mistral) | 윈도우 어텐션 |
| Ring Attention | 분산 무한 컨텍스트 |

### 3. [Mixture of Experts (MoE)](./3_MoE/)
Sparse 아키텍처의 발전

| 논문 | 핵심 내용 |
|------|-----------|
| Switch Transformer | MoE 기초 |
| Mixtral | Sparse MoE 대중화 |
| DeepSeek-V2 | MLA + MoE |
| DeepSeek-V3 | 최신 SOTA MoE |

### 4. [Post-Transformer Architectures](./4_Post_Transformer/)
Transformer를 넘어서

| 논문 | 핵심 내용 |
|------|-----------|
| RWKV | RNN + Transformer |
| Mamba | Selective SSM |
| Jamba | Mamba + Transformer 하이브리드 |
| Mamba-2 | SSM 이론 강화 |
| Hyena | Convolution 기반 |

### 5. [Reasoning & Chain of Thought](./5_Reasoning_CoT/)
LLM의 추론 능력

| 논문 | 핵심 내용 |
|------|-----------|
| Chain-of-Thought | CoT 프롬프팅 |
| Self-Consistency | 다수결 추론 |
| Tree of Thoughts | 트리 탐색 추론 |
| Quiet-STaR | 내부 추론 학습 |

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


*이전: [Part 3 - Domain Adaptation](../Part3_Domain_Adaptation/)*
