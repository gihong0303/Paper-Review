# FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision

**논문 발표**: 2024년 (NeurIPS 2024 Spotlight)
**저자**: Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao
**소속**: Colfax Research, NVIDIA, Princeton University, Meta
**논문 링크**: [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
**공식 구현**: [GitHub](https://github.com/Dao-AILab/flash-attention)

---

## 한 줄 요약
> NVIDIA Hopper GPU의 새로운 하드웨어 기능(WGMMA, TMA, FP8)을 활용하여 FlashAttention-2 대비 1.5-2배 빠른 속도와 H100에서 이론적 최대 성능의 75%를 달성

---

## 1. 문제 정의

### 1.1 FlashAttention-2의 한계

FlashAttention-2는 A100에서 이론적 최대 성능의 70%를 달성했지만, H100 (Hopper 아키텍처)에서는 **35%**에 그침:

| GPU | 이론적 최대 | FlashAttention-2 | 점유율 |
|-----|-------------|------------------|--------|
| A100 | 312 TFLOPS | ~220 TFLOPS | ~70% |
| H100 | 989 TFLOPS | ~350 TFLOPS | ~35% |

### 1.2 원인 분석

```
H100에서 FlashAttention-2가 느린 이유:
┌─────────────────────────────────────────┐
│ 1. 새로운 하드웨어 기능 미활용           │
│    - WGMMA (새로운 Tensor Core 명령)    │
│    - TMA (Tensor Memory Accelerator)    │
│    - FP8 저정밀도 지원                  │
│                                         │
│ 2. 동기식 실행                          │
│    - 데이터 전송과 연산이 순차적        │
│    - GPU 유휴 시간 발생                 │
│                                         │
│ 3. Non-matmul 연산 오버헤드             │
│    - softmax가 Tensor Core 활용 못함    │
└─────────────────────────────────────────┘
```

### 1.3 Hopper의 새로운 기능

| 기능 | 설명 | 효과 |
|------|------|------|
| **WGMMA** | Warpgroup Matrix Multiply-Accumulate | Tensor Core 처리량 2배 |
| **TMA** | Tensor Memory Accelerator | 비동기 데이터 전송 |
| **FP8** | 8비트 부동소수점 | 처리량 2배, 메모리 절반 |

---

## 2. 배경 지식

### 2.1 Hopper GPU 아키텍처

```
H100 SXM5 스펙:
┌────────────────────────────┐
│ SM 개수: 132               │
│ FP16 Tensor Core: 989 TFLOPS │
│ FP8 Tensor Core: 1,979 TFLOPS │
│ HBM3 대역폭: 3.35 TB/s     │
│ L2 캐시: 50 MB              │
└────────────────────────────┘

vs A100:
┌────────────────────────────┐
│ SM 개수: 108               │
│ FP16 Tensor Core: 312 TFLOPS │
│ HBM2e 대역폭: 2.0 TB/s     │
│ L2 캐시: 40 MB              │
└────────────────────────────┘
```

### 2.2 WGMMA (Warpgroup Matrix Multiply-Accumulate)

Hopper의 새로운 Tensor Core 명령어:

```
Ampere (A100):
- wmma 명령어
- 단일 Warp (32 threads) 사용
- Shared Memory에서 로드

Hopper (H100):
- wgmma 명령어
- Warpgroup (128 threads = 4 warps) 사용
- Shared Memory 또는 Register에서 직접 로드
- 비동기 실행 가능
```

### 2.3 TMA (Tensor Memory Accelerator)

전용 하드웨어 유닛으로 데이터 전송 자동화:

```python
# 기존 방식: 여러 스레드가 협력해서 데이터 로드
for i in range(num_threads):
    shared_mem[i] = global_mem[base + i]
__syncthreads()

# TMA: 하드웨어가 자동으로 전송
tma_load(shared_mem, global_mem, tensor_desc)
# GPU 스레드는 다른 작업 수행 가능!
```

---

## 3. 핵심 아이디어

### 3.1 세 가지 핵심 기법

FlashAttention-3는 세 가지 기법으로 Hopper GPU 활용:

| 기법 | 설명 | 기여도 |
|------|------|--------|
| **Warp Specialization** | Producer/Consumer 분리 | ~30% |
| **Pingpong Scheduling** | Softmax와 GEMM 인터리빙 | ~20% |
| **FP8 with Block Quantization** | 저정밀도 + 정확도 보존 | ~50% 추가 |

### 3.2 전체 아키텍처

```
FlashAttention-3 Warp Specialization:
┌──────────────────────────────────────┐
│         Thread Block                  │
│  ┌─────────────┬─────────────────┐   │
│  │  Producer   │    Consumer     │   │
│  │  Warpgroup  │   Warpgroups    │   │
│  │             │                 │   │
│  │  TMA로      │   WGMMA로       │   │
│  │  K,V 로드   │   Attention     │   │
│  │             │   계산          │   │
│  └─────────────┴─────────────────┘   │
│        ↓ 비동기 파이프라인 ↓          │
└──────────────────────────────────────┘
```

---

## 4. 알고리즘 상세 설명

### 4.1 기법 1: Warp Specialization으로 비동기 오버랩

**문제**: 데이터 전송과 연산이 순차적으로 실행되어 GPU가 놀게 됨

**해결**: Producer/Consumer로 역할 분리

```
기존 FlashAttention-2:
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│Load K│→│GEMM  │→│Load V│→│GEMM  │→ ...
└──────┘ └──────┘ └──────┘ └──────┘
      시간 →  (순차적 실행)

FlashAttention-3:
Producer: │Load K₁│Load V₁│Load K₂│Load V₂│...
          ─────────────────────────────────→
Consumer: │  wait │ GEMM₁ │ GEMM₂ │ GEMM₃ │...
          ─────────────────────────────────→
                (파이프라인 실행)
```

### 4.2 기법 2: Pingpong Scheduling

**문제**: GEMM과 Softmax가 같은 리소스(레지스터, shared memory) 경쟁

**해결**: 두 Warpgroup이 번갈아가며 GEMM과 Softmax 수행

```
WGMMA와 Softmax 인터리빙:

Warpgroup 1: │GEMM₁ │Soft₁ │GEMM₃ │Soft₃ │
Warpgroup 2: │Soft₀ │GEMM₂ │Soft₂ │GEMM₄ │
             ─────────────────────────────→
                    시간

핵심: GEMM이 레지스터 점유 중에
      다른 Warpgroup이 Softmax 수행
```

```python
# Pingpong Scheduling 의사코드
def attention_pingpong(Q, K, V):
    # 두 개의 Warpgroup 할당
    wg1, wg2 = allocate_warpgroups()

    for i in range(num_blocks):
        if i % 2 == 0:
            # Warpgroup 1: GEMM, Warpgroup 2: Softmax
            wg1.wgmma(Q, K[i])  # S = Q @ K^T
            wg2.softmax(S_prev)  # P = softmax(S_prev)
        else:
            # 역할 교체
            wg2.wgmma(Q, K[i])
            wg1.softmax(S_prev)
```

### 4.3 기법 3: FP8 Block Quantization

**문제**: FP8의 좁은 동적 범위로 인한 정확도 손실

| Format | Exponent | Mantissa | 동적 범위 |
|--------|----------|----------|-----------|
| FP16 | 5 bits | 10 bits | ~65,504 |
| BF16 | 8 bits | 7 bits | ~3.4×10³⁸ |
| FP8 E4M3 | 4 bits | 3 bits | ~448 |
| FP8 E5M2 | 5 bits | 2 bits | ~57,344 |

**해결**: Block-wise Quantization + Incoherent Processing

```python
# Block Quantization 개념
def fp8_block_attention(Q, K, V, block_size=256):
    # 각 블록별로 스케일 계산
    for block in blocks:
        # 블록 내 최대값으로 스케일 결정
        scale_q = max(abs(Q[block])) / 448  # FP8 최대값
        scale_k = max(abs(K[block])) / 448

        # 스케일링 후 FP8로 변환
        Q_fp8 = (Q[block] / scale_q).to(fp8)
        K_fp8 = (K[block] / scale_k).to(fp8)

        # FP8 GEMM 수행
        S = wgmma_fp8(Q_fp8, K_fp8)

        # 스케일 복원
        S = S * scale_q * scale_k
```

### 4.4 Incoherent Processing

랜덤 직교 행렬로 값 분포 평탄화:

```python
# Incoherent Processing
# 아이디어: 값들의 분포를 더 균일하게 만들어 양자화 오류 감소

def incoherent_attention(Q, K, V):
    # 랜덤 직교 행렬 생성 (한 번만)
    M = random_orthogonal_matrix(d)

    # 값 분포 평탄화
    Q_flat = Q @ M
    K_flat = K @ M

    # FP8로 양자화 (더 균일한 분포)
    Q_fp8, K_fp8 = quantize_fp8(Q_flat, K_flat)

    # Attention 계산
    S = Q_fp8 @ K_fp8.T
    P = softmax(S)
    O = P @ V

    return O
```

**수학적 배경**:
$$\text{Var}((QM)_i) = \frac{1}{d}\|Q\|_F^2$$

모든 원소의 분산이 동일해져서 양자화 오류가 균등하게 분포됨.

---

## 5. 완성된 알고리즘

### 5.1 FlashAttention-3 Forward Pass

```
알고리즘: FlashAttention-3 Forward (FP16/BF16)
────────────────────────────────────────────────
입력: Q, K, V ∈ ℝ^(N×d), 블록 크기 Br, Bc
출력: O ∈ ℝ^(N×d)

# Thread Block 구성
Producer Warpgroup: 1개 (TMA 담당)
Consumer Warpgroups: 2개 (WGMMA + Softmax)

parallel for i = 1 to Tr:
    1. Q_i를 SMEM으로 로드
    2. 초기화: O_i = 0, ℓ_i = 0, m_i = -∞

    # Producer: 비동기 데이터 프리페치
    producer_warpgroup:
        for j = 1 to Tc:
            TMA_async_load(K_j, V_j)

    # Consumer: Pingpong으로 계산
    consumer_warpgroups[0,1]:
        for j = 1 to Tc:
            wait_for_tma(K_j, V_j)

            # Pingpong: 번갈아가며 GEMM/Softmax
            if j % 2 == 0:
                wg0: S_ij = wgmma(Q_i, K_j^T)
                wg1: P_{i,j-1} = softmax(S_{i,j-1})
            else:
                wg1: S_ij = wgmma(Q_i, K_j^T)
                wg0: P_{i,j-1} = softmax(S_{i,j-1})

            # Online Softmax 업데이트
            m_i_new = max(m_i, rowmax(S_ij))
            P̃_ij = exp(S_ij - m_i_new)
            ℓ_i = exp(m_i - m_i_new) * ℓ_i + rowsum(P̃_ij)
            O_i = exp(m_i - m_i_new) * O_i + wgmma(P̃_ij, V_j)
            m_i = m_i_new

    3. 최종 스케일링: O_i = diag(ℓ_i)^(-1) * O_i
    4. O_i를 HBM에 저장

end parallel for
```

### 5.2 FP8 Forward Pass

```
알고리즘: FlashAttention-3 FP8 Forward
────────────────────────────────────────────────
추가 입력: block_size for quantization

# Incoherent Processing (선택적)
if use_incoherent:
    M = random_orthogonal(d)
    Q, K = Q @ M, K @ M

# Block Quantization
for each block b:
    scale_Q[b] = max(abs(Q[b])) / FP8_MAX
    scale_K[b] = max(abs(K[b])) / FP8_MAX
    scale_V[b] = max(abs(V[b])) / FP8_MAX

    Q_fp8[b] = quantize(Q[b] / scale_Q[b])
    K_fp8[b] = quantize(K[b] / scale_K[b])
    V_fp8[b] = quantize(V[b] / scale_V[b])

# FP8 Attention (WGMMA_fp8 사용)
for i, j in blocks:
    S_ij = wgmma_fp8(Q_fp8_i, K_fp8_j)
    S_ij = S_ij * scale_Q[i] * scale_K[j]  # 스케일 복원

    # 나머지는 FP16/FP32로 수행
    P_ij = softmax(S_ij)
    O_i += P_ij @ V_j  # 또는 FP8 V 사용
```

---

## 6. 쉬운 예시로 이해하기

### 6.1 공장 생산라인 비유

**FlashAttention-2**: 한 명이 모든 작업

```
작업자 1명:
재료 가져오기 → 조립 → 재료 가져오기 → 조립 → ...
     5분          10분       5분          10분
총 30분 (대기 시간 포함)
```

**FlashAttention-3**: 역할 분담 (Warp Specialization)

```
배달원 (Producer): 재료 가져오기 → 재료 가져오기 → ...
조립공 (Consumer):      조립      →      조립      → ...

시간:  │──5분──│──5분──│──5분──│
배달:  │ 재료1 │ 재료2 │ 재료3 │
조립:  │  대기 │ 조립1 │ 조립2 │ 조립3

총 20분 (파이프라인 효과)
```

### 6.2 레스토랑 주방 비유 (Pingpong)

두 명의 셰프가 번갈아가며 요리:

```
셰프 A: │고기 굽기│소스 만들기│고기 굽기│소스 만들기│
셰프 B: │소스 만들기│고기 굽기│소스 만들기│고기 굽기│
        ─────────────────────────────────────────→

같은 오븐(레지스터)을 공유하지만 번갈아 사용하므로 충돌 없음
```

### 6.3 FP8 Block Quantization 비유

**문제**: 카메라로 사진을 찍을 때 밝은 부분과 어두운 부분이 함께 있으면 한쪽이 망가짐

**해결**: 영역별로 노출 조정 (HDR 사진)

```
전체 양자화 (일반 사진):
┌─────────────────┐
│ 밝은 영역: 과노출 │  → 정보 손실!
│ 어두운 영역: OK   │
└─────────────────┘

블록별 양자화 (HDR):
┌────────┬────────┐
│영역 1:  │영역 2:  │
│자체 노출│자체 노출│  → 모든 영역 정보 보존
│ 조정    │ 조정    │
└────────┴────────┘
```

---

## 7. 구현

### 7.1 PyTorch에서 FlashAttention-3 사용

```python
import torch
from flash_attn import flash_attn_func

# H100 GPU 필요 (Hopper 아키텍처)
device = torch.device("cuda")

# 입력 준비
batch_size = 4
seq_len = 8192
num_heads = 32
head_dim = 128

q = torch.randn(batch_size, seq_len, num_heads, head_dim,
                device=device, dtype=torch.bfloat16)
k = torch.randn(batch_size, seq_len, num_heads, head_dim,
                device=device, dtype=torch.bfloat16)
v = torch.randn(batch_size, seq_len, num_heads, head_dim,
                device=device, dtype=torch.bfloat16)

# FlashAttention-3 자동 사용 (H100에서)
output = flash_attn_func(
    q, k, v,
    causal=True,
    softmax_scale=None,  # 자동: 1/sqrt(head_dim)
)

print(f"Output shape: {output.shape}")
# Output shape: torch.Size([4, 8192, 32, 128])
```

### 7.2 FP8 Attention 사용

```python
from flash_attn import flash_attn_func

# FP8 입력 (H100 필요)
q_fp8 = q.to(torch.float8_e4m3fn)
k_fp8 = k.to(torch.float8_e4m3fn)
v_fp8 = v.to(torch.float8_e4m3fn)

# FP8 FlashAttention-3
# 주의: 현재 실험적 기능
output = flash_attn_func(
    q_fp8, k_fp8, v_fp8,
    causal=True,
)
```

### 7.3 Triton 구현 (간소화)

```python
import triton
import triton.language as tl

@triton.jit
def flash_attn_v3_kernel(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    N_CTX, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 블록 인덱스
    start_m = tl.program_id(0)
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)

    # Q 블록 로드
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + \
             offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # 초기화
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # K, V 블록 순회
    for start_n in range(0, N_CTX, BLOCK_N):
        # Hopper에서는 TMA로 비동기 로드
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K + off_b * stride_kb + off_h * stride_kh + \
                 offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
        v_ptrs = V + off_b * stride_vb + off_h * stride_vh + \
                 offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk

        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        # S = Q @ K^T
        s = tl.dot(q, tl.trans(k))

        # Online Softmax
        m_ij = tl.max(s, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(s - m_i_new[:, None])

        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = alpha[:, None] * acc + tl.dot(p.to(q.dtype), v)

        m_i = m_i_new

    # 최종 스케일링
    acc = acc / l_i[:, None]

    # 결과 저장
    o_ptrs = O + off_b * stride_ob + off_h * stride_oh + \
             offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX)
```

---

## 8. 실험 결과

### 8.1 Forward Pass 성능

| Sequence Length | Head Dim | FlashAttention-2 | FlashAttention-3 | 속도 향상 |
|-----------------|----------|------------------|------------------|-----------|
| 1K | 64 | 350 TFLOPS | 580 TFLOPS | 1.66× |
| 2K | 64 | 370 TFLOPS | 640 TFLOPS | 1.73× |
| 4K | 64 | 380 TFLOPS | 690 TFLOPS | 1.82× |
| 8K | 64 | 390 TFLOPS | 720 TFLOPS | 1.85× |
| 16K | 64 | 400 TFLOPS | 740 TFLOPS | 1.85× |

### 8.2 FP8 성능

```
H100 기준 처리량:

FP16 FlashAttention-3:  ████████████████████  740 TFLOPS
FP8 FlashAttention-3:   ████████████████████████████████  1.2 PFLOPS

                        0        500       1000      1500
                                  TFLOPS
```

### 8.3 GPU 효율

| 버전 | GPU | 이론적 최대 | 실측 | 효율 |
|------|-----|-------------|------|------|
| FA-2 | A100 | 312 TFLOPS | 220 TFLOPS | 70% |
| FA-2 | H100 | 989 TFLOPS | 350 TFLOPS | 35% |
| **FA-3** | **H100** | **989 TFLOPS** | **740 TFLOPS** | **75%** |

### 8.4 FP8 수치 정확도

```
기준: FP16 FlashAttention-3

일반 FP8 Attention:    평균 오류 = 2.4 × 10⁻³
FP8 + Incoherent:      평균 오류 = 9.1 × 10⁻⁴  (2.6× 감소!)
```

### 8.5 End-to-End LLM 성능

GPT-style 모델 학습 처리량:

| 모델 | FlashAttention-2 | FlashAttention-3 | 속도 향상 |
|------|------------------|------------------|-----------|
| 1.3B | 기준 | 1.4× | +40% |
| 7B | 기준 | 1.5× | +50% |
| 13B | 기준 | 1.6× | +60% |

---

## 9. 한계점 및 후속 연구

### 9.1 현재 한계점

1. **H100 전용**: Hopper 아키텍처에서만 작동
   - A100, RTX 4090 등에서는 FlashAttention-2 사용

2. **FP8 정확도**: 일부 태스크에서 여전히 정확도 손실
   - 긴 시퀀스에서 누적 오류

3. **복잡한 구현**: CUDA 커널 최적화 어려움
   - Triton으로 간소화 진행 중

4. **Backward Pass**: FP8 backward는 아직 불안정
   - Forward만 FP8, Backward는 FP16 권장

### 9.2 후속 연구 방향

1. **차세대 GPU 지원**:
   - NVIDIA Blackwell (B100) 최적화
   - AMD MI300X 지원

2. **더 긴 시퀀스**:
   - 1M+ 토큰 지원
   - Ring Attention과 결합

3. **추가 최적화**:
   - 동적 희소성 활용
   - Mixture of Experts (MoE) 통합

### 9.3 관련 연구

- **Ring Attention**: 분산 환경에서 긴 시퀀스
- **SGLang RadixAttention**: 서빙 최적화
- **Splitwise**: Prefill/Decode 분리

---

## 10. 핵심 요약

### 기억해야 할 것들

1. **Warp Specialization**: Producer/Consumer 분리로 비동기 오버랩
2. **Pingpong Scheduling**: GEMM과 Softmax 인터리빙
3. **FP8 Block Quantization**: 블록별 스케일링으로 정확도 보존
4. **Incoherent Processing**: 랜덤 직교 행렬로 양자화 오류 감소

### 성능 개선 요약

| GPU | FA-2 효율 | FA-3 효율 | 개선 |
|-----|-----------|-----------|------|
| H100 (FP16) | 35% | 75% | 2.1× |
| H100 (FP8) | - | 61% | - |

### 실무 체크리스트

```python
# 1. H100 GPU인지 확인
import torch
assert torch.cuda.get_device_capability()[0] >= 9, "Hopper GPU 필요"

# 2. flash-attn 최신 버전 설치
# pip install flash-attn --no-build-isolation

# 3. FlashAttention-3 사용
from flash_attn import flash_attn_func
output = flash_attn_func(q, k, v, causal=True)

# 4. FP8 사용 (실험적)
# 현재는 forward만 FP8 권장
```

---

## 참고 자료

1. [FlashAttention-3 논문](https://arxiv.org/abs/2407.08608)
2. [공식 GitHub 저장소](https://github.com/Dao-AILab/flash-attention)
3. [Tri Dao 블로그 포스트](https://tridao.me/blog/2024/flash3/)
4. [NVIDIA Hopper 아키텍처 가이드](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
5. [PyTorch 공식 블로그](https://pytorch.org/blog/flashattention-3/)

---

*이전 리뷰: [FlashAttention-2](./002_FlashAttention-2.md)*
*다음 리뷰: [vLLM](./007_vLLM.md)*
