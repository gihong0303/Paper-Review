# Flash-Decoding for Long-Context Inference

**발표**: 2023년 10월
**저자**: Tri Dao, Daniel Haziza, Francisco Massa, Grigory Sizov (Together AI, Meta)
**링크**: [PyTorch Blog](https://pytorch.org/blog/flash-decoding/), [Tri Dao Blog](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
**공식 구현**: [flash-attention GitHub](https://github.com/Dao-AILab/flash-attention)

---

## 한 줄 요약
> 긴 컨텍스트에서의 디코딩 병목을 KV 시퀀스 축 병렬화로 해결하여, 시퀀스 길이 64K 이상에서 8배 이상의 속도 향상을 달성

---

## 1. 문제 정의: 디코딩 단계의 병목

### 1.1 LLM 추론의 두 단계

```
┌────────────────────────────────────────────────────┐
│                  LLM 추론 과정                      │
├──────────────────┬─────────────────────────────────┤
│    Prefill       │         Decode                  │
│  (프롬프트 처리)  │      (토큰 생성)                │
├──────────────────┼─────────────────────────────────┤
│ 입력: 전체 프롬프트│ 입력: 새 토큰 1개씩             │
│ Q: [N, d]        │ Q: [1, d]                       │
│ K, V: [N, d]     │ K, V: [context_len, d]          │
│                  │                                 │
│ Compute-bound    │ Memory-bound                    │
│ (연산량이 병목)   │ (메모리 대역폭이 병목)           │
└──────────────────┴─────────────────────────────────┘
```

### 1.2 Prefill vs Decode 특성

| 특성 | Prefill | Decode |
|------|---------|--------|
| 쿼리 수 | N (프롬프트 길이) | 1 |
| 연산 집약도 | 높음 | 낮음 |
| 병목 | Compute-bound | Memory-bound |
| GPU 활용률 | 높음 | **낮음** |

### 1.3 Decode 단계의 문제

```python
# Decode 단계의 Attention
Q = [1, d]           # 새 토큰 하나의 query
K = [seq_len, d]     # 전체 컨텍스트의 key
V = [seq_len, d]     # 전체 컨텍스트의 value

# 연산량
FLOPs = 2 * seq_len * d  # 매우 적음

# 메모리 접근량
Memory = seq_len * d * 2 (K) + seq_len * d * 2 (V)
       = 4 * seq_len * d bytes

# 연산 강도 (Arithmetic Intensity)
AI = FLOPs / Memory = 0.5  # 매우 낮음!
```

**문제**: 연산량에 비해 메모리 접근이 너무 많음 → GPU 활용률 저하

### 1.4 기존 FlashAttention의 한계

FlashAttention은 Prefill에 최적화:
- 여러 query가 있을 때 배치 축으로 병렬화
- Decode에서는 query가 1개뿐 → 병렬화 불가능

```
Prefill (N개 query):
GPU SM 0: Q[0:N/4]
GPU SM 1: Q[N/4:N/2]
GPU SM 2: Q[N/2:3N/4]
GPU SM 3: Q[3N/4:N]
→ 모든 SM 활용

Decode (1개 query):
GPU SM 0: Q[0]  ← 이 SM만 일함
GPU SM 1: idle
GPU SM 2: idle
...
→ 대부분의 SM이 놀고 있음!
```

---

## 2. Flash-Decoding의 핵심 아이디어

### 2.1 핵심 통찰

> "Query가 1개뿐이라면, **Key-Value 축으로 병렬화**하자!"

```
기존 FlashAttention:
    Q축 병렬화
    ┌─────┐
    │ Q_0 │─→ [전체 K, V 처리]
    │ Q_1 │─→ [전체 K, V 처리]
    │ ... │
    └─────┘

Flash-Decoding:
    KV축 병렬화
    ┌─────┐
    │ Q_0 │─┬→ [K[0:L/4], V[0:L/4]]     GPU SM 0
    │     │├→ [K[L/4:L/2], V[L/4:L/2]] GPU SM 1
    │     │├→ [K[L/2:3L/4], V[L/2:3L/4]] GPU SM 2
    │     │└→ [K[3L/4:L], V[3L/4:L]]   GPU SM 3
    └─────┘
           ↓ Reduction (합치기)
         최종 출력
```

### 2.2 병렬화 구조

```
Input:  Q[1, d], K[seq_len, d], V[seq_len, d]

Step 1: KV를 splits로 분할
        K_0[split_len, d], K_1[split_len, d], ...
        V_0[split_len, d], V_1[split_len, d], ...

Step 2: 각 split을 병렬로 처리
        Thread Block 0: Q @ K_0^T → Softmax → @ V_0
        Thread Block 1: Q @ K_1^T → Softmax → @ V_1
        Thread Block 2: Q @ K_2^T → Softmax → @ V_2
        ...

Step 3: 결과들을 합침 (Reduction)
        Final Output = Combine(Output_0, Output_1, ...)
```

---

## 3. 알고리즘 상세

### 3.1 주요 과제: Softmax 합치기

문제: Softmax는 전체 시퀀스에 대해 정규화해야 함

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}$$

각 split은 자신의 부분만 알고 있음 → 어떻게 합칠까?

### 3.2 Online Softmax Reduction

FlashAttention의 online softmax를 확장:

```python
def combine_partial_softmax(outputs, log_sum_exps, maxes):
    """
    여러 split의 partial softmax 결과를 합침

    Args:
        outputs: 각 split의 출력 [num_splits, d]
        log_sum_exps: 각 split의 log(sum(exp)) [num_splits]
        maxes: 각 split의 최댓값 [num_splits]

    Returns:
        combined_output: 최종 출력 [d]
    """
    # 전체 최댓값
    global_max = max(maxes)

    # 각 split의 기여도 계산
    weights = []
    for i in range(num_splits):
        # exp(local_max - global_max) * local_sum
        weight = exp(maxes[i] - global_max + log_sum_exps[i])
        weights.append(weight)

    # 정규화
    total_weight = sum(weights)

    # 가중 평균
    combined_output = 0
    for i in range(num_splits):
        combined_output += (weights[i] / total_weight) * outputs[i]

    return combined_output
```

### 3.3 수학적 유도

Split 1의 결과: $O_1 = \text{softmax}(S_1) V_1$
Split 2의 결과: $O_2 = \text{softmax}(S_2) V_2$

각 split이 저장하는 정보:
- $m_i = \max(S_i)$ (로컬 최댓값)
- $\ell_i = \sum_j e^{S_{i,j} - m_i}$ (로컬 exp 합)
- $O_i = \frac{1}{\ell_i} \sum_j e^{S_{i,j} - m_i} V_{i,j}$ (로컬 출력)

전체 출력 계산:
$$m = \max(m_1, m_2)$$

$$\ell = e^{m_1 - m} \ell_1 + e^{m_2 - m} \ell_2$$

$$O = \frac{1}{\ell} \left( e^{m_1 - m} \ell_1 O_1 + e^{m_2 - m} \ell_2 O_2 \right)$$

### 3.4 Flash-Decoding 알고리즘

```
알고리즘: Flash-Decoding
────────────────────────────────
입력: Q[1, d], K[L, d], V[L, d], num_splits
출력: O[d]

# Phase 1: 병렬 처리 (각 split에서)
parallel for s = 0 to num_splits - 1:
    start = s * (L / num_splits)
    end = (s + 1) * (L / num_splits)

    K_s = K[start:end]
    V_s = V[start:end]

    # FlashAttention 스타일 처리
    S_s = Q @ K_s^T
    m_s = max(S_s)
    P_s = exp(S_s - m_s)
    ℓ_s = sum(P_s)
    O_s = P_s @ V_s / ℓ_s

    # 통계량 저장
    store(O_s, m_s, ℓ_s)

# Phase 2: Reduction
m = max(m_0, m_1, ..., m_{num_splits-1})
ℓ = sum(e^{m_i - m} * ℓ_i for i in range(num_splits))

O = sum(e^{m_i - m} * ℓ_i * O_i for i) / ℓ

return O
```

---

## 4. 구현 세부사항

### 4.1 GPU 커널 구조

```
┌───────────────────────────────────────┐
│       Flash-Decoding 커널 구조         │
├───────────────────────────────────────┤
│                                       │
│  Kernel 1: Parallel Attention         │
│  ┌─────────────────────────────────┐  │
│  │ Thread Block 0: Split 0 처리    │  │
│  │ Thread Block 1: Split 1 처리    │  │
│  │ ...                             │  │
│  │ Thread Block N: Split N 처리    │  │
│  └─────────────────────────────────┘  │
│               │                       │
│               ▼                       │
│  Kernel 2: Reduction                  │
│  ┌─────────────────────────────────┐  │
│  │ 모든 split 결과를 합쳐서         │  │
│  │ 최종 출력 계산                   │  │
│  └─────────────────────────────────┘  │
│                                       │
└───────────────────────────────────────┘
```

### 4.2 Split 수 결정

```python
def determine_num_splits(seq_len, num_sms=108):
    """
    최적의 split 수 결정

    고려 사항:
    1. GPU SM 개수 (A100: 108)
    2. 각 split의 최소 크기 (너무 작으면 오버헤드)
    3. Reduction 비용
    """
    # 경험적 최적값
    min_split_size = 256
    max_splits = min(num_sms, seq_len // min_split_size)

    # 2의 거듭제곱으로 맞춤 (효율성)
    num_splits = 2 ** int(log2(max_splits))

    return num_splits

# 예시
# seq_len = 64K → num_splits = 64 정도
# seq_len = 8K  → num_splits = 16 정도
```

### 4.3 CUDA 커널 (간소화)

```cuda
// Kernel 1: 병렬 Attention
__global__ void flash_decoding_attention(
    float* Q, float* K, float* V,
    float* partial_O, float* partial_m, float* partial_l,
    int seq_len, int d, int num_splits
) {
    int split_idx = blockIdx.x;
    int split_size = seq_len / num_splits;
    int start = split_idx * split_size;
    int end = start + split_size;

    // 이 split의 K, V
    float* K_split = K + start * d;
    float* V_split = V + start * d;

    // FlashAttention 스타일 처리
    float m = -INFINITY;
    float l = 0.0f;
    float O[HEAD_DIM] = {0};

    for (int i = 0; i < split_size; i += BLOCK_SIZE) {
        // ... attention 계산 ...
    }

    // 결과 저장
    store_partial_result(partial_O, partial_m, partial_l,
                        O, m, l, split_idx);
}

// Kernel 2: Reduction
__global__ void flash_decoding_reduce(
    float* partial_O, float* partial_m, float* partial_l,
    float* final_O, int num_splits
) {
    // 전체 최댓값
    float global_m = -INFINITY;
    for (int i = 0; i < num_splits; i++) {
        global_m = max(global_m, partial_m[i]);
    }

    // 가중 합
    float global_l = 0.0f;
    float O[HEAD_DIM] = {0};

    for (int i = 0; i < num_splits; i++) {
        float scale = exp(partial_m[i] - global_m);
        global_l += scale * partial_l[i];
        for (int j = 0; j < HEAD_DIM; j++) {
            O[j] += scale * partial_l[i] * partial_O[i * HEAD_DIM + j];
        }
    }

    // 정규화
    for (int j = 0; j < HEAD_DIM; j++) {
        final_O[j] = O[j] / global_l;
    }
}
```

---

## 5. 쉬운 예시로 이해하기

### 5.1 시험 채점 비유

**기존 방식 (FlashAttention Decode)**:
- 한 명의 채점자가 모든 답안지(KV)를 순서대로 채점
- 다른 채점자들은 대기

**Flash-Decoding**:
- 답안지를 여러 채점자에게 분배
- 각 채점자가 자기 몫을 동시에 채점
- 마지막에 점수를 합산

### 5.2 레스토랑 주방 비유

한 테이블(Query 1개)의 주문을 처리할 때:

**기존 방식**:
- 주방장 한 명이 모든 재료(KV)를 순서대로 처리
- 다른 주방장들은 놀고 있음

**Flash-Decoding**:
- 재료를 여러 주방장에게 분배
- 각자 담당 재료로 요리
- 마지막에 플레이팅에서 합침

### 5.3 숫자 예시

시퀀스 길이 64K, 헤드 차원 128, A100 (108 SMs):

```
기존 FlashAttention:
- 사용 SM: 1
- 총 시간: 10ms

Flash-Decoding:
- Split 수: 64
- 각 split: 1K 토큰
- 병렬 처리 시간: 0.15ms
- Reduction 시간: 0.02ms
- 총 시간: ~0.17ms

속도 향상: 10ms / 0.17ms ≈ 60배!
```

실제로는 오버헤드 등으로 8-16배 정도 향상

---

## 6. 실험 결과

### 6.1 다양한 시퀀스 길이에서의 성능

| 시퀀스 길이 | FlashAttention | Flash-Decoding | 속도 향상 |
|-------------|----------------|----------------|-----------|
| 1K | 기준 | 약간 느림 | 0.9× |
| 8K | 기준 | 2× | 2× |
| 16K | 기준 | 4× | 4× |
| 32K | 기준 | 6× | 6× |
| 64K | 기준 | 8× | **8×** |
| 128K | OOM | 가능 | **∞** |

**관찰**: 시퀀스가 길수록 이득이 큼 (병렬화 효과)

### 6.2 배치 크기별 성능

| 배치 크기 | FlashDecoding 효과 |
|-----------|-------------------|
| 1 | 최대 효과 |
| 4 | 여전히 효과적 |
| 16 | 효과 감소 |
| 64+ | 기존과 비슷 |

**이유**: 배치가 크면 이미 Query 축 병렬화가 가능

### 6.3 실제 모델에서의 개선

LLaMA-2-70B, 시퀀스 길이 32K:

| 메트릭 | 기존 | Flash-Decoding |
|--------|------|----------------|
| 토큰/초 | 22 | 97 |
| 지연시간 | 45ms/token | 10ms/token |

---

## 7. Flash-Decoding++

### 7.1 추가 최적화

Flash-Decoding++ (2023년 말):
- Flat GEMM 최적화
- DataType 최적화
- 더 나은 split 전략

### 7.2 성능 개선

```
Flash-Decoding:    ████████████░░░░  8×
Flash-Decoding++:  ██████████████░░  10-12×
```

---

## 8. 다른 기술과의 조합

### 8.1 PagedAttention + Flash-Decoding

```python
# vLLM에서 Flash-Decoding 사용
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    # Flash-Decoding은 자동 적용
)
```

### 8.2 Tensor Parallelism과의 상호작용

여러 GPU에 분산된 경우:

```
GPU 0: Head 0-15의 Flash-Decoding
GPU 1: Head 16-31의 Flash-Decoding
...

각 GPU 내에서 KV 축 병렬화
```

### 8.3 Speculative Decoding과의 조합

```
작은 모델 (draft): 빠른 디코딩, Flash-Decoding 적용
큰 모델 (verify): Flash-Decoding으로 verification 가속
```

---

## 9. 한계점 및 고려사항

### 9.1 오버헤드

1. **Kernel Launch 오버헤드**:
   - 두 번의 커널 실행 (attention + reduction)

2. **Reduction 비용**:
   - Split이 많을수록 reduction 비용 증가

3. **메모리 사용**:
   - Partial 결과 저장 공간 필요

### 9.2 최적 사용 시나리오

**효과적인 경우**:
- 긴 시퀀스 (8K+)
- 작은 배치 크기 (1-8)
- 디코딩 지연시간이 중요한 경우

**비효과적인 경우**:
- 짧은 시퀀스 (<1K)
- 큰 배치 크기 (64+)
- Prefill 단계

### 9.3 적응적 전략

```python
def select_attention_kernel(batch_size, seq_len):
    if batch_size * seq_len < threshold:
        # Flash-Decoding
        return flash_decoding_attention
    else:
        # Standard FlashAttention
        return flash_attention
```

---

## 10. 핵심 요약

### 기억해야 할 것들

1. **문제**: Decode 단계는 Query가 1개 → GPU 활용률 낮음
2. **해결책**: KV 시퀀스 축으로 병렬화
3. **핵심 기술**: Online Softmax reduction으로 partial 결과 합치기
4. **효과**: 긴 시퀀스에서 8배+ 속도 향상

### 언제 사용해야 하는가?

| 상황 | Flash-Decoding |
|------|----------------|
| 시퀀스 8K+, 배치 1-8 | 강력 추천 |
| 시퀀스 1-8K, 배치 작음 | 추천 |
| 시퀀스 짧음, 배치 큼 | 효과 작음 |

### 실무 적용

대부분의 최신 서빙 엔진에서 자동 적용:
- vLLM
- TensorRT-LLM
- FlashInfer

```python
# flash-attn 라이브러리 사용 시
from flash_attn import flash_attn_with_kvcache

# 자동으로 Flash-Decoding 적용
output = flash_attn_with_kvcache(
    q, k_cache, v_cache,
    cache_seqlens=seq_lens  # 각 시퀀스의 현재 길이
)
```

---

## 참고 자료

1. [PyTorch Blog - Flash-Decoding](https://pytorch.org/blog/flash-decoding/)
2. [Stanford CRFM Blog](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
3. [flash-attention GitHub](https://github.com/Dao-AILab/flash-attention)
4. [Flash-Decoding++ Paper](https://arxiv.org/abs/2311.01282)

---

*이전 리뷰: [PagedAttention (vLLM)](./003_PagedAttention.md)*
*다음 리뷰: [Splitwise](./005_Splitwise.md)*
