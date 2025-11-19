# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

**논문 발표**: 2023년
**저자**: Tri Dao
**소속**: Stanford University
**논문 링크**: [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
**공식 구현**: [GitHub](https://github.com/Dao-AILab/flash-attention)

---

## 한 줄 요약
> FlashAttention의 알고리즘을 개선하여 GPU 점유율을 높이고, 병렬화를 개선하며, 불필요한 연산을 줄여 A100에서 이론적 최대 성능의 50-73%를 달성 (기존 FlashAttention 대비 약 2배 빠름)

---

## 1. FlashAttention v1의 한계점

### 1.1 낮은 GPU 점유율 (Occupancy)

FlashAttention v1의 Forward pass는 이론적 최대 FLOPS의 **30-50%** 만 달성:

| GPU | 이론적 최대 | FlashAttention v1 | 점유율 |
|-----|-------------|-------------------|--------|
| A100 | 312 TFLOPS | 124 TFLOPS | ~40% |

**원인 분석**:
1. 비효율적인 작업 분할 (Work Partitioning)
2. 불필요한 메모리 읽기/쓰기
3. Non-matmul 연산의 오버헤드

### 1.2 구체적인 병목 지점

```
FlashAttention v1의 문제점:
┌─────────────────────────────────────┐
│ 1. K, V를 외부 루프에서 순회        │
│    → Warp 간 통신/동기화 필요       │
│                                     │
│ 2. Output을 매번 rescale            │
│    → 불필요한 non-matmul FLOPs      │
│                                     │
│ 3. 시퀀스 축 병렬화 없음            │
│    → 배치 크기 작을 때 비효율       │
└─────────────────────────────────────┘
```

---

## 2. FlashAttention-2의 핵심 개선사항

### 2.1 개선 요약

| 측면 | FlashAttention v1 | FlashAttention-2 | 개선 |
|------|-------------------|------------------|------|
| 알고리즘 | K,V 외부 루프 | Q 외부 루프 | 통신 감소 |
| Rescaling | 매 블록마다 | 마지막에 한 번 | FLOPs 감소 |
| 병렬화 | Batch × Head | + Sequence | 점유율 향상 |
| 점유율 | 30-50% | 50-73% | ~2배 |

### 2.2 전체 성능 비교

```
A100 80GB 기준 속도 (TFLOPS):

FlashAttention v1:  ████████████░░░░░░░░  ~124 TFLOPS
FlashAttention-2:   ████████████████████  ~220 TFLOPS
                    0        150       312
                                      (이론적 최대)
```

---

## 3. 알고리즘 개선 1: 루프 순서 변경

### 3.1 v1의 문제: K, V를 외부 루프에서 순회

```
FlashAttention v1:
for j in K_blocks:        # 외부 루프: K, V
    for i in Q_blocks:    # 내부 루프: Q
        compute(Q_i, K_j, V_j) → O_i 업데이트
```

**문제점**: O_i가 여러 thread block에 의해 업데이트됨
- Warp 간 통신 필요
- Shared memory에 O를 저장하고 동기화해야 함

### 3.2 v2의 해결책: Q를 외부 루프에서 순회

```
FlashAttention-2:
for i in Q_blocks:        # 외부 루프: Q
    for j in K_blocks:    # 내부 루프: K, V
        compute(Q_i, K_j, V_j) → O_i 업데이트
```

**장점**: 각 O_i 블록이 하나의 thread block에 의해서만 계산됨
- Warp 간 통신 불필요
- Shared memory 접근 패턴 개선

### 3.3 시각적 비교

```
v1: 여러 thread block이 같은 O_i를 업데이트
    Thread Block 1 ─┐
    Thread Block 2 ─┼─→ O_i (동기화 필요!)
    Thread Block 3 ─┘

v2: 하나의 thread block이 O_i를 완전히 계산
    Thread Block i ────→ O_i (동기화 불필요)
```

---

## 4. 알고리즘 개선 2: Rescaling 최적화

### 4.1 v1의 문제: 매 블록마다 rescaling

FlashAttention v1에서는 online softmax를 위해 **매 블록마다** 출력을 rescale:

```python
# v1: 매 K 블록 처리 후
for j in range(Tc):
    # ... 계산 ...

    # rescaling (매번 수행)
    O_i = diag(l_i_new)^(-1) * (
        diag(l_i) * exp(m_i - m_i_new) * O_i +
        exp(m_ij - m_i_new) * P_ij @ V_j
    )
```

**문제점**:
- 행렬-행렬 곱셈(matmul)보다 느린 element-wise 연산
- 매 블록마다 수행 → 오버헤드 누적

### 4.2 v2의 해결책: 마지막에 한 번만 rescaling

```python
# v2: 루프 내에서는 rescaling 없이 누적
for j in range(Tc):
    # ... 계산 ...

    # rescaling 없이 누적
    O_i = exp(m_i - m_i_new) * O_i + exp(m_ij - m_i_new) * P_ij @ V_j
    l_i = exp(m_i - m_i_new) * l_i + exp(m_ij - m_i_new) * l_ij

# 루프 종료 후 한 번만 rescaling
O_i = diag(l_i)^(-1) * O_i
```

### 4.3 수식 비교

**v1의 반복 공식**:
$$O_i^{(j)} = \text{diag}(\ell_i^{(j)})^{-1} \left( \text{diag}(\ell_i^{(j-1)}) e^{m_i^{(j-1)} - m_i^{(j)}} O_i^{(j-1)} + e^{\tilde{m}_{ij} - m_i^{(j)}} \tilde{P}_{ij} V_j \right)$$

**v2의 반복 공식**:
$$\tilde{O}_i^{(j)} = e^{m_i^{(j-1)} - m_i^{(j)}} \tilde{O}_i^{(j-1)} + e^{\tilde{m}_{ij} - m_i^{(j)}} \tilde{P}_{ij} V_j$$

마지막에:
$$O_i = \text{diag}(\ell_i^{(T_c)})^{-1} \tilde{O}_i^{(T_c)}$$

**효과**: Non-matmul FLOPs가 **4배** 감소

---

## 5. 알고리즘 개선 3: 시퀀스 병렬화

### 5.1 v1의 병렬화

```
v1 병렬화 차원: Batch × NumHeads

배치 크기 = 2, 헤드 수 = 8인 경우:
총 병렬 유닛 = 2 × 8 = 16

문제: A100의 SM 개수 = 108
     → 많은 SM이 놀고 있음!
```

### 5.2 v2의 시퀀스 병렬화

```
v2 병렬화 차원: Batch × NumHeads × SeqLength

배치 크기 = 2, 헤드 수 = 8, 시퀀스 블록 = 16인 경우:
총 병렬 유닛 = 2 × 8 × 16 = 256

→ 모든 SM 활용 가능!
```

### 5.3 시퀀스 병렬화 구현

```python
# 각 thread block이 Q의 일부분만 담당
block_idx = blockIdx.x  # 0 to (batch * heads * seq_blocks - 1)

# 인덱스 분해
batch_idx = block_idx // (num_heads * num_seq_blocks)
head_idx = (block_idx // num_seq_blocks) % num_heads
seq_block_idx = block_idx % num_seq_blocks

# 해당 Q 블록만 로드하여 처리
Q_i = load_Q_block(batch_idx, head_idx, seq_block_idx)
```

**주의**: 이 기능은 Backward pass에서 특히 중요
- Forward는 이미 충분히 병렬화
- Backward는 추가 병렬화로 큰 이득

---

## 6. Warp 수준 최적화

### 6.1 GPU Warp 구조

```
GPU Thread 계층:
┌────────────────────────────┐
│ Thread Block               │
│ ┌──────┬──────┬──────┐    │
│ │Warp 0│Warp 1│Warp 2│... │
│ │32개  │32개  │32개  │    │
│ │thread│thread│thread│    │
│ └──────┴──────┴──────┘    │
└────────────────────────────┘
```

- Warp: 32개 thread가 동시에 같은 명령 실행 (SIMT)
- Warp 내 통신: 빠름 (shuffle 명령어)
- Warp 간 통신: 느림 (shared memory + 동기화)

### 6.2 v1의 Warp 분할 문제

```
v1: 4 warps가 K를 분할

    Warp 0: K[0:Bc/4]  ─┐
    Warp 1: K[Bc/4:Bc/2]├─→ 합쳐서 O 계산
    Warp 2: K[Bc/2:3Bc/4]│   (reduction 필요)
    Warp 3: K[3Bc/4:Bc] ─┘

    문제: Warp 간 reduction은 shared memory 통해야 함
```

### 6.3 v2의 Warp 분할 개선

```
v2: 4 warps가 Q를 분할 (또는 Q와 K를 둘 다 분할)

    Warp 0: Q[0:Br/4]      → O[0:Br/4]
    Warp 1: Q[Br/4:Br/2]   → O[Br/4:Br/2]
    Warp 2: Q[Br/2:3Br/4]  → O[Br/2:3Br/4]
    Warp 3: Q[3Br/4:Br]    → O[3Br/4:Br]

    장점: 각 Warp가 독립적으로 O의 일부를 계산
          Warp 간 통신 불필요!
```

### 6.4 최적 Warp 분할 전략

Forward pass:
- Q를 4 warp로 분할 (K는 모든 warp가 공유)

Backward pass:
- dQ 계산: K를 4 warp로 분할
- dK, dV 계산: Q를 4 warp로 분할

---

## 7. 완성된 FlashAttention-2 알고리즘

### 7.1 Forward Pass

```
알고리즘: FlashAttention-2 Forward
────────────────────────────────────
입력: Q, K, V ∈ ℝ^(N×d)
출력: O ∈ ℝ^(N×d)

# 병렬로 실행 (각 Q 블록마다 하나의 thread block)
parallel for i = 1 to Tr:

    1. Q_i를 HBM에서 SRAM으로 로드
    2. Õ_i = (0), ℓ_i = (0), m_i = (-∞) 초기화

    3. for j = 1 to Tc:
        4. K_j, V_j를 HBM에서 SRAM으로 로드

        5. S_ij = Q_i K_j^T (SRAM에서)
        6. m̃_ij = rowmax(S_ij)
        7. P̃_ij = exp(S_ij - m̃_ij)
        8. ℓ̃_ij = rowsum(P̃_ij)

        9. m_i^new = max(m_i, m̃_ij)
        10. # Rescaling 없이 누적
        11. Õ_i = e^(m_i - m_i^new) Õ_i + e^(m̃_ij - m_i^new) P̃_ij V_j
        12. ℓ_i = e^(m_i - m_i^new) ℓ_i + e^(m̃_ij - m_i^new) ℓ̃_ij
        13. m_i = m_i^new

    14. # 마지막에 한 번만 rescaling
    15. O_i = diag(ℓ_i)^(-1) Õ_i
    16. O_i, ℓ_i, m_i를 HBM에 저장

end parallel for
```

### 7.2 Causal Masking 최적화

```python
# v2에서의 효율적 causal masking
for i in range(Tr):
    for j in range(Tc):
        # 완전히 마스킹되는 블록은 스킵
        if j * Bc > (i + 1) * Br - 1:
            break  # 이후 j들도 모두 마스킹됨

        # 부분 마스킹이 필요한 블록
        elif j * Bc >= i * Br:
            S_ij = compute_with_mask(Q_i, K_j)

        # 마스킹 불필요한 블록
        else:
            S_ij = Q_i @ K_j.T
```

---

## 8. 성능 분석

### 8.1 이론적 FLOPS 분석

Attention의 총 FLOPS (causal 아닌 경우):
$$\text{FLOPs} = 4 \cdot N^2 \cdot d$$

A100의 이론적 최대:
$$\text{Max TFLOPS} = 312 \text{ (FP16/BF16 with tensor cores)}$$

### 8.2 실측 성능 비교

| 시퀀스 길이 | Head Dim | FlashAttention | FlashAttention-2 | 속도 향상 |
|-------------|----------|----------------|------------------|-----------|
| 1K | 64 | 124 TFLOPS | 187 TFLOPS | 1.5× |
| 2K | 64 | 136 TFLOPS | 210 TFLOPS | 1.5× |
| 4K | 64 | 141 TFLOPS | 220 TFLOPS | 1.6× |
| 8K | 64 | 144 TFLOPS | 227 TFLOPS | 1.6× |
| 16K | 64 | 145 TFLOPS | 230 TFLOPS | 1.6× |

### 8.3 이론 대비 효율

```
FlashAttention-2 GPU 효율:

시퀀스 1K:  ████████████████░░░░  60% (187/312)
시퀀스 4K:  ██████████████████░░  71% (220/312)
시퀀스 16K: ██████████████████░░  74% (230/312)

           0%        50%      100%
```

### 8.4 Backward Pass 개선

| 측면 | v1 | v2 | 개선 |
|------|----|----|------|
| Forward | ~40% | ~70% | 1.7× |
| Backward | ~25-30% | ~60% | 2× |

Backward가 더 많이 개선된 이유:
- 시퀀스 병렬화의 효과가 큼
- Warp 분할 최적화의 효과가 큼

---

## 9. 다른 구현체들과의 비교

### 9.1 속도 비교 (A100 80GB)

| 구현체 | 시퀀스 2K | 시퀀스 8K |
|--------|-----------|-----------|
| PyTorch | 기준 | 기준 |
| Megatron | 1.7× | 1.8× |
| xFormers | 1.8× | 2.0× |
| FlashAttention | 2.4× | 2.8× |
| **FlashAttention-2** | **4.0×** | **4.6×** |

### 9.2 Triton 구현과 비교

Triton으로 구현한 FlashAttention-2:
- 성능: 직접 CUDA의 ~90%
- 장점: 더 읽기 쉽고 수정하기 쉬움
- 단점: 일부 최적화 적용 어려움

```python
# Triton 구현 예시 (간소화)
@triton.jit
def flash_attn_v2_fwd(Q, K, V, O, ...):
    # 블록 인덱스
    i_block = tl.program_id(0)

    # Q 블록 로드
    q = tl.load(Q + i_block * Br * d)

    # 초기화
    o = tl.zeros([Br, d])
    l = tl.zeros([Br])
    m = tl.full([Br], -float('inf'))

    # K, V 블록 순회
    for j in range(Tc):
        k = tl.load(K + j * Bc * d)
        v = tl.load(V + j * Bc * d)

        # 어텐션 계산
        s = tl.dot(q, tl.trans(k))
        # ... online softmax ...
        o = o * scale + tl.dot(p, v)

    # 최종 rescaling
    o = o / l[:, None]
    tl.store(O + i_block * Br * d, o)
```

---

## 10. 쉬운 예시로 이해하기

### 10.1 레스토랑 주방 비유

**v1**: 여러 요리사가 한 접시를 같이 만듦
- 서로 "내가 지금 뭐 넣었어" 계속 소통 필요
- 한 명이 끝날 때까지 다른 사람 대기

**v2**: 각 요리사가 자기 접시를 처음부터 끝까지 담당
- 소통 불필요
- 병렬로 여러 접시 동시 완성

### 10.2 시험 채점 비유

**v1**: 각 문제를 모든 채점자가 조금씩 채점
- 1번 문제: 채점자 A가 10문항, B가 10문항, C가 10문항...
- 점수 합산을 위해 모든 채점자가 모여야 함

**v2**: 각 채점자가 전체 답안지의 일부를 완전히 채점
- 채점자 A: 1-10번 학생 전체 채점
- 채점자 B: 11-20번 학생 전체 채점
- 각자 독립적으로 작업, 모일 필요 없음

### 10.3 숫자로 보는 개선

시퀀스 길이 4096, 배치 1, 헤드 32인 경우:

```
v1 병렬 유닛: 1 × 32 = 32
A100 SM 개수: 108
→ 76개 SM이 놀고 있음!

v2 (시퀀스 병렬화 적용):
Q 블록 수: 4096 / 64 = 64
병렬 유닛: 1 × 32 × 64 = 2048
→ 모든 SM이 풀가동!
```

---

## 11. 구현 세부사항

### 11.1 메모리 사용량

| 단계 | FlashAttention | FlashAttention-2 |
|------|----------------|------------------|
| Forward | O(N) | O(N) |
| Backward | O(N) | O(N) |

둘 다 O(N) 추가 메모리만 사용 (표준은 O(N²))

### 11.2 지원 기능

FlashAttention-2에서 지원:
- Multi-query attention (MQA)
- Grouped-query attention (GQA)
- ALiBi positional encoding
- Rotary positional encoding (RoPE)
- Causal masking
- 다양한 head dimensions (최대 256)

### 11.3 사용 방법

```python
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

# 기본 사용
out = flash_attn_func(q, k, v, causal=True)

# QKV가 packed된 경우
out = flash_attn_qkvpacked_func(qkv, causal=True)

# MHA, MQA, GQA 자동 지원
# q: [batch, seq, heads, dim]
# k, v: [batch, seq, kv_heads, dim]
# kv_heads < heads이면 자동으로 GQA 처리
```

---

## 12. FlashAttention-3로의 발전

### 12.1 FlashAttention-2의 남은 과제

1. **Hopper 아키텍처 최적화**:
   - H100의 새로운 기능 활용 안 됨
   - TMA (Tensor Memory Accelerator)
   - FP8 지원

2. **더 긴 시퀀스**:
   - 100K+ 시퀀스 지원

### 12.2 FlashAttention-3 미리보기

2024년에 발표된 FlashAttention-3:
- H100에서 ~740 TFLOPS (이론 대비 75%)
- FP8 지원으로 더 빠른 속도
- 비동기 데이터 전송

---

## 13. 핵심 요약

### 기억해야 할 것들

1. **루프 순서 변경**: K→Q에서 Q→K로 (통신 감소)
2. **Rescaling 최적화**: 마지막에 한 번만 (FLOPs 감소)
3. **시퀀스 병렬화**: GPU 점유율 향상
4. **Warp 분할 개선**: Warp 간 통신 제거

### 성능 개선 요약

| 개선 사항 | 기여도 |
|-----------|--------|
| 루프 순서 변경 | ~20% |
| Rescaling 최적화 | ~10% |
| 시퀀스 병렬화 | ~30% |
| Warp 분할 | ~20% |

### 실무 적용

```python
# PyTorch 2.0+에서 자동 활성화
# 또는 명시적 사용:
from flash_attn import flash_attn_func

# 기본값으로 v2 사용
output = flash_attn_func(
    q, k, v,
    causal=True,
    softmax_scale=None,  # 자동 계산
    dropout_p=0.0
)
```

---

## 참고 자료

1. [FlashAttention-2 논문](https://arxiv.org/abs/2307.08691)
2. [공식 GitHub 저장소](https://github.com/Dao-AILab/flash-attention)
3. [Tri Dao의 발표 영상](https://www.youtube.com/watch?v=IoMSGuiwV3g)
4. [Triton 구현](https://github.com/openai/triton)

---

*이전 리뷰: [FlashAttention](./001_FlashAttention.md)*
*다음 리뷰: [PagedAttention (vLLM)](./003_PagedAttention.md)*
