# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**논문 발표**: 2022년 (NeurIPS 2022)
**저자**: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
**소속**: Stanford University, University at Buffalo
**논문 링크**: [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
**공식 구현**: [GitHub](https://github.com/Dao-AILab/flash-attention)

---

## 한 줄 요약
> GPU 메모리 계층 구조를 고려한 IO-aware 알고리즘으로 Attention 연산의 메모리 접근을 최소화하여, 정확도 손실 없이 2-4배 빠른 속도와 5-20배 적은 메모리 사용을 달성

---

## 1. 문제 정의: 왜 이 연구가 필요한가?

### 1.1 Transformer의 근본적 한계

Transformer 모델의 핵심인 Self-Attention은 시퀀스 길이 N에 대해 **O(N²)** 의 시간 및 공간 복잡도를 가집니다.

```
시퀀스 길이 1024  → Attention 행렬 크기: 1,048,576
시퀀스 길이 4096  → Attention 행렬 크기: 16,777,216
시퀀스 길이 16384 → Attention 행렬 크기: 268,435,456
```

이로 인해:
- **메모리 부족**: 긴 시퀀스를 처리할 수 없음
- **속도 저하**: 대부분의 시간이 메모리 읽기/쓰기에 소비됨

### 1.2 기존 해결책의 한계

기존 연구들은 **근사(Approximation)** 방식을 사용했습니다:
- Sparse Attention (Longformer, BigBird)
- Low-rank Approximation (Linformer, Performer)

**문제점**: 정확도 손실이 발생하고, 실제 wall-clock 시간은 크게 개선되지 않음

### 1.3 FlashAttention의 핵심 통찰

> "문제는 연산량(FLOPs)이 아니라 **메모리 접근(Memory Access)** 이다!"

현대 GPU는 연산 능력에 비해 메모리 대역폭이 병목입니다:
- A100 GPU 연산 능력: 312 TFLOPS (FP16)
- A100 메모리 대역폭: 2 TB/s

즉, **Memory-bound** 상황에서는 FLOP 수를 줄이는 것보다 **메모리 접근을 줄이는 것**이 더 효과적입니다.

---

## 2. 배경 지식

### 2.1 GPU 메모리 계층 구조

```
┌─────────────────────────────────────┐
│            GPU 구조                  │
├─────────────────────────────────────┤
│                                     │
│  ┌─────────┐    ┌─────────────────┐ │
│  │  SRAM   │    │      HBM        │ │
│  │ (빠름)  │    │    (느림)       │ │
│  │  192KB  │    │   40-80GB      │ │
│  │ ~19TB/s │    │   ~2TB/s       │ │
│  └─────────┘    └─────────────────┘ │
│                                     │
└─────────────────────────────────────┘
```

| 메모리 타입 | 용량 | 대역폭 | 특징 |
|------------|------|--------|------|
| **SRAM** (On-chip) | ~192KB per SM | ~19 TB/s | 매우 빠름, 용량 작음 |
| **HBM** (Off-chip) | 40-80GB | ~2 TB/s | 느림, 용량 큼 |

**핵심**: SRAM은 HBM보다 약 **10배 빠르지만**, 용량이 매우 작음

### 2.2 표준 Attention의 동작 방식

표준 Self-Attention 수식:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

표준 구현의 메모리 접근 패턴:

```python
# 표준 Attention (의사 코드)
def standard_attention(Q, K, V):
    # Step 1: S = QK^T 계산 → HBM에서 읽고 HBM에 쓰기
    S = Q @ K.T / sqrt(d_k)      # O(N²) 메모리 사용

    # Step 2: P = softmax(S) → HBM에서 읽고 HBM에 쓰기
    P = softmax(S, dim=-1)       # O(N²) 메모리 사용

    # Step 3: O = PV → HBM에서 읽고 HBM에 쓰기
    O = P @ V                    # O(N²) 메모리 사용

    return O
```

**문제점**: N×N 크기의 중간 행렬(S, P)을 **HBM에 저장**해야 함

### 2.3 IO 복잡도 분석

표준 Attention의 HBM 접근량:
- Q, K, V 읽기: O(Nd)
- S 쓰기/읽기: O(N²)
- P 쓰기/읽기: O(N²)
- O 쓰기: O(Nd)

**총 HBM 접근**: O(Nd + N²)

시퀀스가 길어질수록 N² 항이 지배적 → **메모리 대역폭이 병목**

---

## 3. FlashAttention의 핵심 아이디어

### 3.1 타일링 (Tiling)

전체 Attention을 한 번에 계산하지 않고, **작은 블록(타일)** 단위로 나누어 계산합니다.

```
표준 방식:                    FlashAttention:
┌───────────────┐            ┌──┬──┬──┬──┐
│               │            │  │  │  │  │
│   N × N       │     →      ├──┼──┼──┼──┤
│  전체 계산    │            │  │  │  │  │
│               │            ├──┼──┼──┼──┤
└───────────────┘            │  │  │  │  │
                             └──┴──┴──┴──┘
                             블록 단위 계산
```

### 3.2 Kernel Fusion

여러 연산을 하나의 GPU 커널로 합칩니다:
1. QK^T 계산
2. Scaling
3. Masking (optional)
4. Softmax
5. Dropout (optional)
6. V와의 곱셈

**장점**: 중간 결과를 HBM에 쓰지 않고 SRAM에서 바로 처리

### 3.3 Recomputation (재계산)

Backward pass에서 중간 행렬(S, P)을 저장하지 않고 **다시 계산**합니다.

> "메모리를 저장하는 것보다 다시 계산하는 것이 더 빠르다!"

---

## 4. 알고리즘 상세 설명

### 4.1 Online Softmax 기법

Softmax를 블록 단위로 계산하기 위해 **Online Softmax** 알고리즘을 사용합니다.

#### 표준 Softmax의 문제

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}$$

전체 시퀀스를 봐야 분모를 계산할 수 있음 → 블록 단위 계산 불가능

#### Online Softmax 해결책

**핵심 아이디어**: 점진적으로 통계량(max, sum)을 업데이트

두 블록 x^(1)과 x^(2)의 softmax를 합칠 때:

$$m^{(new)} = \max(m^{(1)}, m^{(2)})$$

$$\ell^{(new)} = e^{m^{(1)} - m^{(new)}} \ell^{(1)} + e^{m^{(2)} - m^{(new)}} \ell^{(2)}$$

여기서:
- m: 현재까지의 최댓값 (수치 안정성을 위함)
- ℓ: 현재까지의 exp 합

### 4.2 Forward Pass 알고리즘

```
알고리즘: FlashAttention Forward Pass
─────────────────────────────────────
입력: Q, K, V ∈ ℝ^(N×d), 블록 크기 Br, Bc
출력: O ∈ ℝ^(N×d)

1. Q를 Tr = ⌈N/Br⌉ 개의 블록으로 분할
2. K, V를 Tc = ⌈N/Bc⌉ 개의 블록으로 분할
3. O = (0)_N×d, ℓ = (0)_N, m = (-∞)_N 초기화

4. for j = 1 to Tc do:                    # K, V 블록 순회
5.     HBM에서 K_j, V_j를 SRAM으로 로드
6.     for i = 1 to Tr do:                # Q 블록 순회
7.         HBM에서 Q_i, O_i, ℓ_i, m_i를 SRAM으로 로드
8.
9.         # SRAM에서 계산 수행
10.        S_ij = Q_i × K_j^T              # 블록 Attention 점수
11.        m̃_ij = rowmax(S_ij)             # 행별 최댓값
12.        P̃_ij = exp(S_ij - m̃_ij)        # 안정적인 exp
13.        ℓ̃_ij = rowsum(P̃_ij)            # 행별 합
14.
15.        # 통계량 업데이트
16.        m_i^new = max(m_i, m̃_ij)
17.        ℓ_i^new = e^(m_i - m_i^new) × ℓ_i + e^(m̃_ij - m_i^new) × ℓ̃_ij
18.
19.        # 출력 업데이트
20.        O_i = diag(ℓ_i^new)^(-1) × (
21.              diag(ℓ_i) × e^(m_i - m_i^new) × O_i +
22.              e^(m̃_ij - m_i^new) × P̃_ij × V_j
23.        )
24.
25.        ℓ_i = ℓ_i^new, m_i = m_i^new
26.        O_i를 HBM에 저장
27.    end for
28. end for

29. return O
```

### 4.3 시각적 이해

```
      K 블록들
    ┌──┬──┬──┬──┐
    │K₁│K₂│K₃│K₄│
    └──┴──┴──┴──┘
Q      j=1,2,3,4
블   ┌──┐
록   │Q₁│ → [S₁₁][S₁₂][S₁₃][S₁₄] → O₁ 점진적 업데이트
들   ├──┤
i=   │Q₂│ → [S₂₁][S₂₂][S₂₃][S₂₄] → O₂ 점진적 업데이트
1,2  ├──┤
3,4  │Q₃│ → [S₃₁][S₃₂][S₃₃][S₃₄] → O₃ 점진적 업데이트
     ├──┤
     │Q₄│ → [S₄₁][S₄₂][S₄₃][S₄₄] → O₄ 점진적 업데이트
     └──┘
```

각 블록 S_ij만 SRAM에 존재 → **전체 N×N 행렬이 HBM에 저장되지 않음**

---

## 5. IO 복잡도 분석

### 5.1 FlashAttention의 HBM 접근량

**정리**: FlashAttention의 HBM 접근 횟수는 O(N²d²M⁻¹)

여기서 M은 SRAM 크기 (보통 ~100KB)

### 5.2 비교

| 알고리즘 | HBM 접근 | 추가 메모리 |
|----------|----------|-------------|
| 표준 Attention | O(Nd + N²) | O(N²) |
| FlashAttention | O(N²d²M⁻¹) | O(N) |

d (head dimension) = 64-128, M = 100KB일 때:
- N이 충분히 크면 (예: N > 1024)
- **FlashAttention이 표준 방식보다 적은 HBM 접근**

### 5.3 왜 더 빠른가?

```
표준 Attention:
HBM ←→ SRAM ←→ Compute ←→ SRAM ←→ HBM (매 연산마다)
     느림                         느림

FlashAttention:
HBM → SRAM ←→ Compute ←→ SRAM → HBM (블록 단위로 한 번씩)
    한번                      한번
```

---

## 6. Backward Pass와 Recomputation

### 6.1 표준 역전파의 문제

역전파 시 Softmax 출력 P가 필요:
$$\frac{\partial \mathcal{L}}{\partial S} = P \odot \left(\frac{\partial \mathcal{L}}{\partial P} - \text{rowsum}\left(\frac{\partial \mathcal{L}}{\partial P} \odot P\right)\right)$$

P를 저장하려면 O(N²) 메모리 필요

### 6.2 FlashAttention의 해결책

P를 저장하지 않고 **다시 계산**:
- Forward에서 저장: O, ℓ, m (O(N) 메모리)
- Backward에서 S, P를 재계산

**Trade-off**: 메모리 O(N²) 절약 vs FLOP 약간 증가 (재계산 비용)

실제로는 메모리 접근 감소 효과가 재계산 비용보다 커서 **더 빠름**

---

## 7. 쉬운 예시로 이해하기

### 7.1 도서관 비유

**표준 Attention** = 모든 책을 책상에 펼쳐놓고 작업
- 책상 공간(SRAM) 부족
- 계속 서가(HBM)를 왔다 갔다
- 많은 시간 낭비

**FlashAttention** = 필요한 책만 가져와서 작업
- 한 번에 몇 권만 책상에
- 그 책들 작업 완료 후 반납
- 다음 책들 가져옴
- 효율적인 작업

### 7.2 요리 비유

**표준 Attention** = 모든 재료를 한꺼번에 도마 위에
- 도마(SRAM)가 작아서 재료가 넘침
- 계속 냉장고(HBM)에서 꺼내고 넣음

**FlashAttention** = 순서대로 재료를 처리
- 필요한 재료만 도마에
- 처리 완료 후 다음 재료
- 냉장고 왕복 최소화

### 7.3 숫자 예시

시퀀스 길이 N = 1024, head dimension d = 64

**표준 Attention 메모리**:
- S 행렬: 1024 × 1024 × 2 bytes = 2 MB
- P 행렬: 1024 × 1024 × 2 bytes = 2 MB
- 총: ~4 MB (HBM에 저장)

**FlashAttention 메모리**:
- 블록 크기 64 × 64 = 4096
- 블록당: 4096 × 2 bytes = 8 KB
- SRAM에서 처리 가능!

---

## 8. 실험 결과

### 8.1 속도 향상

| 모델 | 표준 PyTorch | FlashAttention | 속도 향상 |
|------|--------------|----------------|-----------|
| BERT-large | 기준 | - | 15% faster |
| GPT-2 | 기준 | - | 3× faster |

### 8.2 메모리 절약

시퀀스 길이별 메모리 사용량 (GPT-2 Medium):

| 시퀀스 길이 | 표준 | FlashAttention | 절약 |
|-------------|------|----------------|------|
| 1K | 기준 | - | 2× |
| 2K | OOM | 가능 | ∞ |
| 4K | OOM | 가능 | ∞ |

### 8.3 긴 시퀀스 처리

FlashAttention으로 가능해진 것들:
- GPT-2: 1K → **4K** 시퀀스
- BERT: 512 → **4K** 시퀀스

### 8.4 실제 태스크 성능

| 태스크 | 기존 SOTA | FlashAttention 모델 |
|--------|-----------|---------------------|
| Path-X (길이 16K) | 50% (random) | **61.4%** |
| Path-256 (길이 64K) | 50% (random) | **63.1%** |

긴 시퀀스를 처리할 수 있게 되어 성능 향상!

---

## 9. 구현 세부사항

### 9.1 블록 크기 선택

```python
# SRAM 크기에 따른 블록 크기 결정
# M = SRAM 크기, d = head dimension
block_size = min(
    ceil(M / (4 * d)),  # SRAM 제약
    d                    # 효율성
)
```

A100 GPU (M ≈ 192KB)에서:
- d = 64 → 블록 크기 ≈ 64-128
- d = 128 → 블록 크기 ≈ 64

### 9.2 CUDA 커널 구현

```cuda
// 간소화된 FlashAttention 커널 구조
__global__ void flash_attention_forward(
    float* Q, float* K, float* V, float* O,
    int N, int d, int Br, int Bc
) {
    // 공유 메모리(SRAM) 할당
    __shared__ float Qi[Br][d];
    __shared__ float Kj[Bc][d];
    __shared__ float Vj[Bc][d];
    __shared__ float Sij[Br][Bc];

    // 블록 단위 처리
    for (int j = 0; j < Tc; j++) {
        // K, V 블록 로드
        load_to_shared(K, Kj, j);
        load_to_shared(V, Vj, j);

        for (int i = 0; i < Tr; i++) {
            // Q 블록 로드
            load_to_shared(Q, Qi, i);

            // SRAM에서 Sij = Qi @ Kj^T 계산
            compute_attention_block(Qi, Kj, Sij);

            // Online softmax 및 출력 업데이트
            update_output(Sij, Vj, O, ...);
        }
    }
}
```

### 9.3 Causal Masking 지원

```python
# Causal (autoregressive) attention을 위한 마스킹
# 블록 단위로 효율적 처리

if causal:
    # j > i인 블록은 완전히 스킵
    # j == i인 블록만 마스킹 적용
    for j in range(Tc):
        for i in range(Tr):
            if j * Bc > (i + 1) * Br:
                continue  # 이 블록은 모두 마스킹됨
            elif j * Bc >= i * Br:
                # 부분 마스킹 필요
                apply_causal_mask(Sij, i, j)
```

---

## 10. 한계점 및 후속 연구

### 10.1 FlashAttention v1의 한계

1. **GPU 점유율 (Occupancy)**:
   - Backward pass에서 최적화 여지 있음
   - Sequence 축 병렬화 미지원

2. **Head dimension 제약**:
   - d ≤ 128만 지원
   - d = 256 등 큰 값 미지원

3. **호환성**:
   - 특정 attention 변형들 미지원

### 10.2 후속 연구들

- **FlashAttention-2** (2023): 병렬화 개선, 2배 추가 속도 향상
- **Flash-Decoding** (2023): 디코딩 단계 최적화
- **FlashAttention-3** (2024): Hopper 아키텍처 최적화

### 10.3 영향력

FlashAttention은 현대 LLM 서빙의 **필수 구성요소**가 되었습니다:
- Hugging Face Transformers 기본 통합
- PyTorch 2.0 native 지원
- 거의 모든 LLM 서빙 엔진에서 사용

---

## 11. 핵심 요약

### 기억해야 할 것들

1. **핵심 통찰**: 연산량(FLOPs)보다 메모리 접근(IO)이 병목
2. **해결책**: 타일링 + 커널 퓨전 + 재계산
3. **결과**: 정확도 손실 없이 2-4배 빠름, 5-20배 메모리 절약
4. **핵심 기술**: Online Softmax로 블록 단위 계산 가능

### 실무 적용

```python
# PyTorch 2.0+에서 사용
import torch
import torch.nn.functional as F

# 자동으로 FlashAttention 사용
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    output = F.scaled_dot_product_attention(query, key, value)

# 또는 flash-attn 라이브러리 직접 사용
from flash_attn import flash_attn_func
output = flash_attn_func(q, k, v, causal=True)
```

---

## 참고 자료

1. [원본 논문](https://arxiv.org/abs/2205.14135)
2. [공식 GitHub 저장소](https://github.com/Dao-AILab/flash-attention)
3. [Tri Dao의 블로그 포스트](https://tridao.me/publications/)
4. [ELI5 FlashAttention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)

---

*다음 리뷰: [FlashAttention-2](./002_FlashAttention-2.md) - 더욱 개선된 병렬화와 속도*
