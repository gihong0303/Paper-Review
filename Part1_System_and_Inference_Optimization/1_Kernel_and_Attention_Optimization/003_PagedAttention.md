# PagedAttention: Efficient Memory Management for Large Language Model Serving

**논문 발표**: 2023년 (SOSP 2023)
**저자**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
**소속**: UC Berkeley
**논문 링크**: [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
**공식 구현**: [vLLM GitHub](https://github.com/vllm-project/vllm)

---

## 한 줄 요약
> OS의 가상 메모리 페이징 기법을 KV Cache 관리에 적용하여, 메모리 낭비를 4% 미만으로 줄이고 처리량을 2-4배 향상시킨 LLM 서빙 시스템

---

## 1. 문제 정의: LLM 서빙의 메모리 병목

### 1.1 KV Cache란?

Autoregressive 생성에서 이전 토큰들의 Key, Value를 저장:

```python
# KV Cache 없이 (매우 비효율)
for i in range(seq_len):
    # 매번 처음부터 모든 K, V 재계산
    output = attention(Q[:i], K[:i], V[:i])

# KV Cache 사용 (효율적)
for i in range(seq_len):
    # 이전 K, V는 캐시에서 가져옴
    K_cache[i] = new_K
    V_cache[i] = new_V
    output = attention(Q[i], K_cache[:i+1], V_cache[:i+1])
```

### 1.2 KV Cache의 메모리 크기

단일 토큰의 KV Cache 크기:
$$\text{KV size per token} = 2 \times L \times H \times D \times \text{sizeof(dtype)}$$

- L: 레이어 수
- H: 어텐션 헤드 수
- D: 헤드 차원

**예시: LLaMA-13B (FP16)**
$$2 \times 40 \times 5120 \times 2 = 800 \text{ KB/token}$$

시퀀스 길이 2048이면: **1.6 GB per request**

### 1.3 기존 시스템의 문제점

#### 문제 1: 메모리 사전 할당

```
기존 방식: 최대 시퀀스 길이만큼 미리 할당

Request 1: [████████░░░░░░░░░░░░░░░░]  실제 사용: 8 / 예약: 24
Request 2: [██████░░░░░░░░░░░░░░░░░░]  실제 사용: 6 / 예약: 24
Request 3: [████░░░░░░░░░░░░░░░░░░░░]  실제 사용: 4 / 예약: 24

총 예약: 72 슬롯
실제 사용: 18 슬롯
낭비: 75%!
```

#### 문제 2: 내부 단편화 (Internal Fragmentation)

사용하지 않는 슬롯이 예약되어 있어 다른 요청이 사용 못함

#### 문제 3: 외부 단편화 (External Fragmentation)

요청이 끝나면 메모리가 조각나서 큰 요청 수용 불가

```
메모리 상태: [████][    ][██][    ][███][    ]
                 ^       ^        ^
             빈 공간들이 흩어져 있음

새 요청 (크기 5): 연속된 공간이 없어서 실패!
```

### 1.4 실제 메모리 낭비 측정

| 시스템 | 메모리 낭비율 |
|--------|---------------|
| HuggingFace | 60-80% |
| FasterTransformer | 20-40% |
| **vLLM (PagedAttention)** | **<4%** |

---

## 2. 핵심 아이디어: OS 페이징 기법 차용

### 2.1 OS 가상 메모리 개념

```
가상 주소 공간         물리 메모리
┌────────┐            ┌────────┐
│ Page 0 │ ─────────→ │ Frame 3│
├────────┤            ├────────┤
│ Page 1 │ ───┐       │ Frame 2│
├────────┤    │       ├────────┤
│ Page 2 │ ─┐ └─────→ │ Frame 7│
├────────┤  │         ├────────┤
│ Page 3 │  └───────→ │ Frame 1│
└────────┘            └────────┘

Page Table이 가상→물리 매핑 관리
```

**핵심**:
- 연속적인 가상 주소 → 비연속적인 물리 메모리
- 페이지 단위로 할당 → 단편화 최소화

### 2.2 PagedAttention으로의 적용

```
논리적 KV Cache (요청 시점)    물리적 KV Cache (GPU 메모리)
┌────────┐                    ┌────────┐
│ Block 0│ ─────────────────→ │ Block 7│
├────────┤                    ├────────┤
│ Block 1│ ─────────┐         │ Block 1│
├────────┤          │         ├────────┤
│ Block 2│ ───┐     └───────→ │ Block 3│
├────────┤    │               ├────────┤
│ Block 3│    └─────────────→ │ Block 9│
└────────┘                    └────────┘

Block Table이 논리→물리 매핑 관리
```

---

## 3. PagedAttention 아키텍처

### 3.1 핵심 구성 요소

```
┌─────────────────────────────────────────┐
│              vLLM Engine                │
├─────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────────┐   │
│  │  Scheduler  │  │   KV Cache      │   │
│  │             │  │   Manager       │   │
│  └─────────────┘  └─────────────────┘   │
│         │                  │            │
│         ▼                  ▼            │
│  ┌─────────────────────────────────┐    │
│  │      PagedAttention Kernel      │    │
│  └─────────────────────────────────┘    │
│                    │                    │
│                    ▼                    │
│  ┌─────────────────────────────────┐    │
│  │       Physical KV Blocks        │    │
│  │    (GPU Memory Pool)            │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

### 3.2 Block의 구조

각 Block은 고정된 수의 토큰의 KV를 저장:

```python
# Block 구조
class KVBlock:
    block_size: int = 16  # 토큰 수
    num_layers: int
    num_heads: int
    head_dim: int

    # 실제 데이터
    key: Tensor[num_layers, block_size, num_heads, head_dim]
    value: Tensor[num_layers, block_size, num_heads, head_dim]
```

### 3.3 Block Table

각 시퀀스의 논리 블록 → 물리 블록 매핑:

```python
# 예시: 시퀀스 길이 50, 블록 크기 16
# 필요한 블록 수: ceil(50/16) = 4

block_table = [7, 1, 3, 9]  # 논리 블록 0,1,2,3 → 물리 블록 7,1,3,9

# 토큰 위치 계산
def get_physical_position(token_idx):
    logical_block = token_idx // block_size
    offset = token_idx % block_size
    physical_block = block_table[logical_block]
    return (physical_block, offset)

# 예: 토큰 35의 위치
# logical_block = 35 // 16 = 2
# offset = 35 % 16 = 3
# physical_block = block_table[2] = 3
# → 물리 블록 3의 오프셋 3
```

---

## 4. PagedAttention 커널

### 4.1 표준 Attention vs PagedAttention

**표준 Attention**:
```python
# K, V가 연속 메모리에 저장
output = softmax(Q @ K.T / sqrt(d)) @ V
```

**PagedAttention**:
```python
# K, V가 블록 단위로 분산 저장
for block_idx in block_table:
    K_block = physical_blocks[block_idx].key
    V_block = physical_blocks[block_idx].value
    # 블록별 attention 계산 후 합산
```

### 4.2 PagedAttention 알고리즘

```
알고리즘: PagedAttention
─────────────────────────────
입력: Q (현재 토큰), Block Table, Physical Blocks
출력: Attention Output

1. 각 query head에 대해 병렬로:
2.     output = 0, normalizer = 0, max_score = -∞
3.
4.     for block_idx in block_table:
5.         K = physical_blocks[block_idx].key
6.         V = physical_blocks[block_idx].value
7.
8.         # 블록 내 attention score 계산
9.         scores = Q @ K.T / sqrt(d)
10.
11.        # Online softmax (FlashAttention과 유사)
12.        block_max = max(scores)
13.        new_max = max(max_score, block_max)
14.
15.        # 이전 결과 rescale
16.        scale = exp(max_score - new_max)
17.        output = output * scale
18.        normalizer = normalizer * scale
19.
20.        # 현재 블록 기여 추가
21.        exp_scores = exp(scores - new_max)
22.        output += exp_scores @ V
23.        normalizer += sum(exp_scores)
24.
25.        max_score = new_max
26.
27.    output = output / normalizer
28.    return output
```

### 4.3 Partition 기반 병렬화

긴 시퀀스의 경우 블록들을 partition으로 나누어 병렬 처리:

```
시퀀스 블록들: [B0][B1][B2][B3][B4][B5][B6][B7]
                  ↓
Partition 0:  [B0][B1][B2][B3] → Thread Block 0
Partition 1:  [B4][B5][B6][B7] → Thread Block 1
                  ↓
        각 partition 결과를 병합 (reduction)
```

---

## 5. 메모리 관리 기법들

### 5.1 동적 블록 할당

```python
class BlockAllocator:
    def __init__(self, num_blocks):
        self.free_blocks = list(range(num_blocks))

    def allocate(self):
        if self.free_blocks:
            return self.free_blocks.pop()
        return None  # OOM

    def free(self, block_idx):
        self.free_blocks.append(block_idx)
```

**장점**: 필요할 때만 할당 → 낭비 최소화

### 5.2 Copy-on-Write (CoW)

여러 시퀀스가 동일한 prefix를 공유:

```
시나리오: 동일한 시스템 프롬프트를 사용하는 여러 요청

Request 1: "You are helpful assistant." + "What is AI?"
Request 2: "You are helpful assistant." + "What is ML?"
Request 3: "You are helpful assistant." + "What is DL?"

CoW 적용:
                Shared Blocks (Read-only)
               ┌─────────────────────┐
Request 1 ────→│ "You are helpful   │──→ [Own blocks for "What is AI?"]
Request 2 ────→│  assistant."       │──→ [Own blocks for "What is ML?"]
Request 3 ────→│                    │──→ [Own blocks for "What is DL?"]
               └─────────────────────┘

메모리 절약: 3배의 중복 저장 방지
```

### 5.3 Preemption과 Swapping

GPU 메모리가 부족할 때:

```python
# Preemption 전략
def handle_oom():
    # 1. Swapping: KV Cache를 CPU로 이동
    victim = select_victim_sequence()
    victim.kv_cache.to('cpu')

    # 2. Recomputation: KV Cache 버리고 나중에 재계산
    # (더 극단적인 경우)
```

---

## 6. 스케줄링

### 6.1 Continuous Batching

전통적 배칭 vs Continuous Batching:

```
전통적 배칭:
Batch 1: [Req A: ████████████]
         [Req B: ████████████]
         [Req C: ████████████]
         모든 요청이 끝날 때까지 대기, 그 다음 Batch 2 시작

Continuous Batching:
Time 1: [Req A: █][Req B: █][Req C: █]
Time 2: [Req A: █][Req B: █][Req C: █]
Time 3: [Req A: done][Req B: █][Req D: █]  ← A 끝나자마자 D 추가
Time 4: [Req E: █][Req B: █][Req D: █]    ← 새 요청 즉시 투입
```

**장점**: GPU 유휴 시간 최소화

### 6.2 vLLM의 스케줄링 정책

```python
def schedule():
    running = []  # 현재 실행 중인 요청
    waiting = []  # 대기 중인 요청

    # 1. Running 요청 처리
    for req in running:
        if can_allocate_new_block(req):
            allocate_block(req)
        else:
            # Preemption 필요
            preempt(req)

    # 2. Waiting 요청 추가
    while waiting and has_free_blocks():
        req = waiting.pop(0)
        allocate_initial_blocks(req)
        running.append(req)

    return running
```

---

## 7. 쉬운 예시로 이해하기

### 7.1 주차장 비유

**기존 방식**: 예약 주차장
- 차량 A: 10칸 예약 (실제 사용 3칸)
- 차량 B: 10칸 예약 (실제 사용 5칸)
- 낭비: 12칸

**PagedAttention**: 발렛 파킹
- 들어오는 차량마다 빈 공간에 주차
- 필요한 만큼만 공간 사용
- 차량이 나가면 즉시 공간 회수

### 7.2 도서관 서가 비유

**기존 방식**:
- 각 사용자가 책장 하나 전체를 예약
- 책 3권만 놓아도 책장 전체 차지

**PagedAttention**:
- 필요한 책 수만큼 칸을 할당
- 칸들이 떨어져 있어도 OK
- 카탈로그(Block Table)로 위치 추적

### 7.3 숫자 예시

LLaMA-13B, 시퀀스 길이 2048, 요청 10개:

```
기존 방식 (최대 길이 할당):
메모리 = 10 × 2048 × 800KB = 16GB

PagedAttention (평균 실제 길이 500):
메모리 = 10 × 500 × 800KB = 4GB

절약: 75%!

추가로 수용 가능한 요청: 30개 더!
```

---

## 8. 실험 결과

### 8.1 처리량 비교

| 모델 | 시스템 | 처리량 (req/s) | 향상 |
|------|--------|----------------|------|
| OPT-13B | HuggingFace | 기준 | 1× |
| OPT-13B | FasterTransformer | 1.7× | 1.7× |
| OPT-13B | **vLLM** | **7.2×** | **7.2×** |
| OPT-66B | FasterTransformer | 기준 | 1× |
| OPT-66B | **vLLM** | **2.2×** | **2.2×** |

### 8.2 메모리 효율

```
메모리 낭비율:

HuggingFace:       ████████████████████  ~80%
FasterTransformer: ████████░░░░░░░░░░░░  ~40%
vLLM:              █░░░░░░░░░░░░░░░░░░░   <4%
                   0%   25%   50%   75%  100%
```

### 8.3 Parallel Sampling 성능

동일 프롬프트에서 여러 샘플 생성 (Copy-on-Write 활용):

| 샘플 수 | 기존 메모리 | vLLM 메모리 | 절약 |
|---------|-------------|-------------|------|
| 4 | 4× | 1.3× | 67% |
| 8 | 8× | 1.5× | 81% |
| 16 | 16× | 1.7× | 89% |

### 8.4 Beam Search 성능

| 모델 | Beam Width | FasterTransformer | vLLM | 향상 |
|------|------------|-------------------|------|------|
| OPT-13B | 4 | 기준 | 2.3× | 2.3× |
| OPT-13B | 6 | OOM | 가능 | ∞ |

---

## 9. 구현 세부사항

### 9.1 블록 크기 선택

```python
# 블록 크기 trade-off
# 작은 블록: 세밀한 관리, 오버헤드 증가
# 큰 블록: 오버헤드 감소, 내부 단편화 증가

# vLLM 기본값
BLOCK_SIZE = 16  # 토큰
```

### 9.2 CUDA 커널 구현

```cuda
// PagedAttention 커널 (간소화)
__global__ void paged_attention_kernel(
    float* out,
    float* query,
    float* key_cache,
    float* value_cache,
    int* block_tables,
    int num_blocks,
    int block_size
) {
    int head_idx = blockIdx.x;
    int seq_idx = blockIdx.y;

    // Block table에서 물리 블록 주소 가져오기
    int* block_table = block_tables + seq_idx * max_blocks;

    float qk_max = -INFINITY;
    float exp_sum = 0.0f;
    float output[HEAD_DIM] = {0};

    // 각 블록 순회
    for (int i = 0; i < num_blocks; i++) {
        int physical_block = block_table[i];

        // 해당 블록의 K, V 주소 계산
        float* k = key_cache + physical_block * block_size * HEAD_DIM;
        float* v = value_cache + physical_block * block_size * HEAD_DIM;

        // Attention 계산 (online softmax)
        for (int j = 0; j < block_size; j++) {
            float score = dot_product(query, k + j * HEAD_DIM);
            // ... softmax 업데이트 ...
        }
    }

    // 최종 출력
    normalize_and_store(output, exp_sum, out);
}
```

### 9.3 vLLM 사용 방법

```python
from vllm import LLM, SamplingParams

# 모델 로드
llm = LLM(model="meta-llama/Llama-2-13b-hf")

# 샘플링 파라미터
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256
)

# 추론
prompts = [
    "The future of AI is",
    "In a galaxy far away",
    "The recipe for success is"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

---

## 10. 한계점 및 후속 연구

### 10.1 PagedAttention의 한계

1. **Kernel 오버헤드**:
   - 비연속적 메모리 접근으로 인한 성능 저하
   - FlashAttention만큼 빠르지 않음

2. **블록 크기 고정**:
   - 다양한 시퀀스 길이에 최적화 어려움

3. **단일 GPU 한정**:
   - 초기 버전은 Tensor Parallelism 미지원

### 10.2 후속 발전

- **vLLM v2**: Tensor Parallelism 지원
- **FlashInfer**: PagedAttention + FlashAttention 통합
- **SGLang**: 추가 최적화 (RadixAttention 등)

### 10.3 생태계 영향

PagedAttention은 LLM 서빙의 **사실상 표준**이 되었습니다:
- TensorRT-LLM에 통합
- HuggingFace TGI에 통합
- 대부분의 서빙 엔진에서 채택

---

## 11. 관련 개념: Prefix Caching

### 11.1 아이디어

자주 사용되는 시스템 프롬프트의 KV Cache를 캐싱:

```
시스템 프롬프트: "You are a helpful AI assistant..."

첫 요청: 시스템 프롬프트 처리 (Prefill) → KV Cache 생성 및 저장
이후 요청: 저장된 KV Cache 재사용 → Prefill 스킵

속도 향상: 시스템 프롬프트가 길수록 큰 이득
```

### 11.2 구현 (vLLM)

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_prefix_caching=True  # Prefix Caching 활성화
)
```

---

## 12. 핵심 요약

### 기억해야 할 것들

1. **핵심 문제**: KV Cache의 메모리 단편화와 낭비
2. **해결책**: OS 페이징 기법 (가상 메모리) 차용
3. **주요 기법**: Block Table, Copy-on-Write, Continuous Batching
4. **결과**: 메모리 낭비 <4%, 처리량 2-24배 향상

### 핵심 공식

메모리 효율:
$$\text{Memory Efficiency} = \frac{\text{Used Memory}}{\text{Allocated Memory}}$$

기존: ~20-40%
vLLM: **>96%**

### 실무 적용

```python
# vLLM으로 고성능 서빙
from vllm import LLM, SamplingParams

llm = LLM(
    model="your-model",
    tensor_parallel_size=4,        # 4 GPU
    gpu_memory_utilization=0.9,    # GPU 메모리 90% 사용
    max_model_len=8192             # 최대 시퀀스 길이
)

# Continuous batching 자동 적용
outputs = llm.generate(prompts, sampling_params)
```

---

## 참고 자료

1. [PagedAttention 논문](https://arxiv.org/abs/2309.06180)
2. [vLLM GitHub](https://github.com/vllm-project/vllm)
3. [vLLM 블로그](https://blog.vllm.ai/)
4. [SOSP 2023 발표](https://www.youtube.com/watch?v=5ZlavKF_98U)

---

*이전 리뷰: [FlashAttention-2](./002_FlashAttention-2.md)*
*다음 리뷰: [Flash-Decoding](./004_Flash-Decoding.md)*
