# vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention

**논문 발표**: 2023년 (SOSP 2023)
**저자**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Chhavi Yadav, Zhanghao Wu, Chelsea Finn, Clark Barrett, Ion Stoica, Hao Zhang
**소속**: UC Berkeley, Stanford University
**논문 링크**: [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
**공식 구현**: [GitHub](https://github.com/vllm-project/vllm)

---

## 한 줄 요약
> 운영체제의 가상 메모리와 페이징 기법을 KV 캐시 관리에 적용한 PagedAttention으로, 메모리 낭비를 4% 미만으로 줄이고 기존 시스템 대비 2-4배 높은 처리량을 달성

---

## 1. 문제 정의

### 1.1 LLM 서빙의 메모리 문제

LLM 서빙에서 **KV 캐시**가 가장 큰 메모리 병목:

```
13B 모델의 메모리 사용량:
┌────────────────────────────────────────┐
│ 모델 가중치:        26 GB              │
│ KV 캐시 (하나의 요청): 1.7 GB          │
│ Activation:         ~1 GB              │
└────────────────────────────────────────┘

문제: 40GB GPU에서 ~7개 요청만 동시 처리 가능
     → 처리량 심각하게 제한!
```

### 1.2 KV 캐시의 특성

| 특성 | 설명 | 문제점 |
|------|------|--------|
| **동적 크기** | 토큰 생성마다 증가 | 사전 할당 필요 |
| **가변 수명** | 요청마다 다름 | 조기 해제 어려움 |
| **시퀀스 의존** | 연속 메모리 필요 | 단편화 발생 |

### 1.3 기존 시스템의 메모리 낭비

기존 시스템(FasterTransformer, Orca)의 메모리 관리:

```
기존 방식: 최대 길이로 사전 할당
┌──────────────────────────────────────┐
│ 요청 1: 실제 100 토큰 / 할당 2048    │
│         ████░░░░░░░░░░░░░░░░░░░░░░░░ │
│         낭비: 95%                     │
│                                      │
│ 요청 2: 실제 500 토큰 / 할당 2048    │
│         ████████████░░░░░░░░░░░░░░░░ │
│         낭비: 76%                     │
└──────────────────────────────────────┘

총 메모리 낭비: 60-80%!
```

### 1.4 낭비의 세 가지 유형

1. **예약 낭비 (Reserved)**: 미래 토큰을 위해 예약했지만 사용 안 됨
2. **내부 단편화 (Internal)**: 할당된 블록 내 빈 공간
3. **외부 단편화 (External)**: 블록 사이의 빈 공간

---

## 2. 배경 지식

### 2.1 KV 캐시란?

Transformer의 Attention에서 이전 토큰의 Key, Value를 저장:

```python
# Autoregressive Generation에서 KV 캐시
def generate_with_cache(model, prompt):
    kv_cache = []

    for layer in model.layers:
        # 이전 K, V 재사용
        k_cache, v_cache = kv_cache[layer]

        # 현재 토큰만 계산
        k_new = layer.key_proj(current_token)
        v_new = layer.value_proj(current_token)

        # 캐시에 추가
        k = concat(k_cache, k_new)  # 누적!
        v = concat(v_cache, v_new)

        # Attention
        attn = softmax(q @ k.T) @ v
```

### 2.2 KV 캐시 메모리 계산

단일 요청의 KV 캐시 크기:
$$\text{KV Cache Size} = 2 \times L \times H \times d \times s \times \text{dtype}$$

| 변수 | 설명 | LLaMA-13B |
|------|------|-----------|
| L | 레이어 수 | 40 |
| H | 헤드 수 | 40 |
| d | 헤드 차원 | 128 |
| s | 시퀀스 길이 | 2048 |
| dtype | FP16 | 2 bytes |

계산: $2 \times 40 \times 40 \times 128 \times 2048 \times 2 = 1.7\text{GB}$

### 2.3 운영체제의 가상 메모리

vLLM의 핵심 아이디어는 OS의 메모리 관리 기법:

```
가상 메모리 시스템:
┌─────────────┐     ┌─────────────┐
│ 가상 메모리  │     │ 물리 메모리  │
│ (연속)      │ ──→ │ (비연속)    │
│             │ 페이지│             │
│ 프로그램    │ 테이블│ RAM        │
└─────────────┘     └─────────────┘

장점:
- 프로그램은 연속 메모리로 인식
- 실제로는 여기저기 분산 저장
- 필요할 때만 물리 메모리 할당
```

---

## 3. 핵심 아이디어: PagedAttention

### 3.1 기본 개념

KV 캐시를 **고정 크기 블록**으로 나누어 비연속적으로 저장:

```
기존 방식:
┌─────────────────────────────┐
│ K: [tok1|tok2|tok3|...|tokN] │  연속 메모리 필요
│ V: [tok1|tok2|tok3|...|tokN] │
└─────────────────────────────┘

PagedAttention:
┌────────┐ ┌────────┐ ┌────────┐
│Block 0 │ │Block 1 │ │Block 2 │  비연속 OK!
│tok1-16 │ │tok17-32│ │tok33-48│
└────────┘ └────────┘ └────────┘
     ↓          ↓          ↓
   GPU 메모리 여기저기에 분산 저장
```

### 3.2 핵심 자료구조

```python
# Block Table: 논리 블록 → 물리 블록 매핑
class BlockTable:
    def __init__(self, max_blocks):
        # 각 시퀀스의 블록 매핑
        # logical_block_id → physical_block_id
        self.table = {}

    def allocate(self, seq_id, logical_id):
        physical_id = self.free_blocks.pop()
        self.table[(seq_id, logical_id)] = physical_id

    def lookup(self, seq_id, logical_id):
        return self.table[(seq_id, logical_id)]
```

### 3.3 PagedAttention 연산

```python
def paged_attention(query, key_cache, value_cache, block_table, context_len):
    """
    query: [num_heads, head_dim]
    key_cache: [num_blocks, block_size, num_heads, head_dim]
    value_cache: [num_blocks, block_size, num_heads, head_dim]
    block_table: [max_num_blocks] - 물리 블록 인덱스
    """
    output = zeros_like(query)

    num_blocks = (context_len + block_size - 1) // block_size

    for i in range(num_blocks):
        # 물리 블록 조회
        physical_block_id = block_table[i]

        # 해당 블록의 K, V 가져오기
        k_block = key_cache[physical_block_id]
        v_block = value_cache[physical_block_id]

        # Attention 계산
        scores = query @ k_block.T
        weights = softmax(scores)
        output += weights @ v_block

    return output
```

---

## 4. 알고리즘 상세 설명

### 4.1 메모리 할당 정책

**On-demand 할당**: 토큰이 생성될 때만 블록 할당

```
시간에 따른 블록 할당:
t=0:  요청 도착 → Block 0 할당
      ┌────┐
      │B0  │ (prompt 토큰들)
      └────┘

t=1:  16번째 토큰 생성 → Block 0 가득 참
      ┌────┐
      │B0██│
      └────┘

t=2:  17번째 토큰 생성 → Block 1 새로 할당
      ┌────┐ ┌────┐
      │B0██│ │B1░ │
      └────┘ └────┘

낭비는 마지막 블록의 빈 슬롯만!
```

### 4.2 Copy-on-Write (CoW)

여러 시퀀스가 같은 prefix를 공유할 때:

```
Parallel Sampling 예시:
같은 prompt에서 여러 응답 생성

초기 상태: prompt 공유
┌────────────┐
│ Seq A ─────┼──→ Block 0 (prompt)
│ Seq B ─────┤
│ Seq C ─────┘
└────────────┘

분기 발생: CoW로 복사
┌────────────┐
│ Seq A ─────┼──→ Block 0 (공유)  → Block 1a (A만)
│ Seq B ─────┼──→ Block 0 (공유)  → Block 1b (B만)
│ Seq C ─────┼──→ Block 0 (공유)  → Block 1c (C만)
└────────────┘

메모리 절약: prompt 부분은 한 번만 저장!
```

### 4.3 스케줄링

vLLM의 First-Come-First-Serve (FCFS) + Preemption:

```python
class Scheduler:
    def __init__(self, max_num_seqs, max_num_batched_tokens):
        self.waiting = []     # 대기 중인 요청
        self.running = []     # 실행 중인 요청
        self.swapped = []     # 스왑된 요청

    def schedule(self):
        # 1. 실행 중인 요청 계속 처리
        scheduled = []

        # 2. 메모리가 있으면 대기 요청 추가
        while self.waiting and self.can_allocate():
            seq = self.waiting.pop(0)
            self.allocate_blocks(seq)
            scheduled.append(seq)

        # 3. 메모리 부족 시 Preemption
        if not self.can_allocate() and self.waiting:
            victim = self.running[-1]  # 가장 최근 요청
            self.swap_out(victim)      # CPU로 스왑
            self.swapped.append(victim)

        return scheduled
```

### 4.4 Preemption 전략

두 가지 preemption 방식:

| 방식 | 설명 | 장단점 |
|------|------|--------|
| **Swapping** | KV 캐시를 CPU로 이동 | 느리지만 재사용 가능 |
| **Recomputation** | KV 캐시 삭제 후 재계산 | 빠르지만 연산 낭비 |

```python
def preempt(self, seq, method='swap'):
    if method == 'swap':
        # CPU 메모리로 KV 캐시 복사
        cpu_blocks = copy_to_cpu(seq.blocks)
        self.free_gpu_blocks(seq.blocks)
        seq.cpu_blocks = cpu_blocks

    elif method == 'recompute':
        # KV 캐시 삭제 (나중에 다시 계산)
        self.free_gpu_blocks(seq.blocks)
        seq.needs_recompute = True
```

---

## 5. 시스템 아키텍처

### 5.1 전체 구조

```
vLLM 아키텍처:
┌─────────────────────────────────────────┐
│              Frontend API               │
│    (OpenAI-compatible, HuggingFace)     │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│              Scheduler                  │
│  (요청 관리, 배치 구성, 메모리 할당)      │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│           Block Manager                 │
│   (물리 블록 할당, Block Table 관리)     │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│              Worker                     │
│    (PagedAttention 커널 실행)           │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│         GPU Memory (KV Cache)           │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐   │
│  │ B0 │ │ B1 │ │ B2 │ │... │ │ Bn │   │
│  └────┘ └────┘ └────┘ └────┘ └────┘   │
└─────────────────────────────────────────┘
```

### 5.2 Continuous Batching

동적으로 배치 구성 변경:

```
기존 Static Batching:
배치 1: [Req1, Req2, Req3] → 모두 끝날 때까지 대기
                            Req1 끝나도 빈 슬롯

Continuous Batching:
Step 1: [Req1, Req2, Req3]
Step 2: [Req1, Req2, Req3]  (Req1 완료)
Step 3: [Req4, Req2, Req3]  (Req4 즉시 투입!)
Step 4: [Req4, Req2, Req5]  (Req3 완료, Req5 투입)

→ GPU 항상 최대 활용!
```

---

## 6. 쉬운 예시로 이해하기

### 6.1 호텔 방 배정 비유

**기존 방식**: 전체 숙박 기간만큼 미리 예약

```
손님 A: 1박만 할 건데 7박 예약 (최대 기간)
┌───┬───┬───┬───┬───┬───┬───┐
│ A │░░░│░░░│░░░│░░░│░░░│░░░│  6박 낭비!
└───┴───┴───┴───┴───┴───┴───┘

손님 B, C, D: 방 없음 (실제론 빈 방 많음)
```

**PagedAttention**: 매일 방 배정

```
Day 1: A→101호
Day 2: A 체크아웃, B→101호, C→102호
Day 3: B→101호, C 체크아웃, D→102호
...

모든 방 효율적 사용!
```

### 6.2 도서관 책 보관 비유

**기존 방식**: 연속된 선반 필요

```
책 시리즈 A (3권): [A1][A2][A3][ ][ ][ ]  선반 1개 전체 예약
책 시리즈 B (2권): 선반 2 전체에 [B1][B2][ ][ ][ ][ ]

문제: 빈 칸 많지만 새 시리즈 들어갈 자리 없음
```

**PagedAttention**: 어디든 배치

```
선반 1: [A1][B1][C1][A2][D1][E1]
선반 2: [B2][A3][C2][...

색인표 (Block Table):
시리즈 A → 선반1-1번, 선반1-4번, 선반2-2번
시리즈 B → 선반1-2번, 선반2-1번

빈 공간 없이 효율적 사용!
```

### 6.3 숫자로 보는 개선

LLaMA-13B, 요청당 평균 512 토큰, 최대 2048:

```
기존 방식:
할당: 2048 × 1.7GB/2048 = 1.7GB
실제 사용: 512 × 1.7GB/2048 = 0.425GB
낭비: 75%

PagedAttention (블록 크기 16):
필요 블록: 512/16 = 32개
블록당 메모리: 1.7GB/128 = 13.3MB
총 사용: 32 × 13.3MB = 426MB
낭비: 마지막 블록의 빈 슬롯 = <4%
```

---

## 7. 구현

### 7.1 vLLM 설치 및 기본 사용

```python
# 설치
# pip install vllm

from vllm import LLM, SamplingParams

# 모델 로드
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,  # GPU 수
    gpu_memory_utilization=0.9,  # GPU 메모리 사용률
)

# 추론
prompts = [
    "The capital of France is",
    "The future of AI is",
    "Python is a programming language that",
]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100,
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated!r}\n")
```

### 7.2 OpenAI 호환 서버

```bash
# 서버 시작
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --port 8000
```

```python
# 클라이언트 사용
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",  # vLLM은 API 키 불필요
)

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ],
    max_tokens=100,
)

print(response.choices[0].message.content)
```

### 7.3 PagedAttention 커널 (간소화)

```python
import torch
import triton
import triton.language as tl

@triton.jit
def paged_attention_kernel(
    output_ptr, query_ptr,
    key_cache_ptr, value_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    num_seqs, num_heads, head_size, block_size,
    max_num_blocks_per_seq,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
):
    # 시퀀스와 헤드 인덱스
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # 현재 시퀀스 길이
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # Query 로드
    q_offset = seq_idx * num_heads * head_size + head_idx * head_size
    q = tl.load(query_ptr + q_offset + tl.arange(0, HEAD_SIZE))

    # 출력 초기화
    acc = tl.zeros([HEAD_SIZE], dtype=tl.float32)
    m_i = float('-inf')  # 최대값
    l_i = 0.0  # 합계

    # 각 블록 순회
    num_blocks = (seq_len + block_size - 1) // block_size

    for block_idx in range(num_blocks):
        # Block Table에서 물리 블록 ID 조회
        block_table_offset = seq_idx * max_num_blocks_per_seq + block_idx
        physical_block_id = tl.load(block_tables_ptr + block_table_offset)

        # K, V 블록 로드
        kv_offset = physical_block_id * block_size * num_heads * head_size
        k_block_ptr = key_cache_ptr + kv_offset
        v_block_ptr = value_cache_ptr + kv_offset

        # 블록 내 각 토큰에 대해 attention
        for token_idx in range(block_size):
            global_token_idx = block_idx * block_size + token_idx
            if global_token_idx >= seq_len:
                break

            # K, V 로드
            token_offset = token_idx * num_heads * head_size + head_idx * head_size
            k = tl.load(k_block_ptr + token_offset + tl.arange(0, HEAD_SIZE))
            v = tl.load(v_block_ptr + token_offset + tl.arange(0, HEAD_SIZE))

            # Attention score
            score = tl.sum(q * k) / tl.sqrt(float(head_size))

            # Online softmax
            m_i_new = tl.maximum(m_i, score)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(score - m_i_new)

            l_i = alpha * l_i + p
            acc = alpha * acc + p * v
            m_i = m_i_new

    # 최종 정규화
    acc = acc / l_i

    # 결과 저장
    out_offset = seq_idx * num_heads * head_size + head_idx * head_size
    tl.store(output_ptr + out_offset + tl.arange(0, HEAD_SIZE), acc)
```

### 7.4 고급 기능 사용

```python
from vllm import LLM, SamplingParams

# Prefix Caching (공유 prefix 활용)
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_prefix_caching=True,  # 공통 prefix 캐싱
)

# Beam Search
sampling_params = SamplingParams(
    use_beam_search=True,
    best_of=4,
    max_tokens=100,
)

# Speculative Decoding
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    speculative_model="meta-llama/Llama-2-7b-hf",
    num_speculative_tokens=5,
)
```

---

## 8. 실험 결과

### 8.1 메모리 낭비 비교

| 시스템 | 예약 낭비 | 내부 단편화 | 외부 단편화 | 총 낭비 |
|--------|-----------|-------------|-------------|---------|
| FasterTransformer | 36% | 27% | 1% | 64% |
| Orca | 31% | 22% | 7% | 60% |
| **vLLM** | **0%** | **<4%** | **0%** | **<4%** |

### 8.2 처리량 비교

LLaMA-13B, A100-40GB:

```
처리량 (requests/second):

FasterTransformer: ████░░░░░░░░░░░░  기준
Orca:              ████████░░░░░░░░  1.7×
vLLM:              ████████████████  2-4×

                   0    1×   2×   3×   4×
```

### 8.3 다양한 워크로드 성능

| 모델 | 워크로드 | FasterTransformer | Orca | vLLM | 개선 |
|------|----------|-------------------|------|------|------|
| LLaMA-7B | ShareGPT | 1.0× | 1.5× | 2.2× | +46% |
| LLaMA-13B | ShareGPT | 1.0× | 1.7× | 2.4× | +41% |
| LLaMA-7B | Alpaca | 1.0× | 1.3× | 3.5× | +169% |
| LLaMA-13B | Alpaca | 1.0× | 1.5× | 3.8× | +153% |

### 8.4 Parallel Sampling 성능

같은 prompt에서 여러 샘플 생성:

| 샘플 수 | Orca | vLLM | 메모리 절약 |
|---------|------|------|-------------|
| 2 | 기준 | 1.2× | 17% |
| 4 | 기준 | 1.7× | 39% |
| 6 | 기준 | 2.2× | 55% |

Copy-on-Write로 공유 prefix 메모리 절약!

### 8.5 Beam Search 성능

| Beam 크기 | 기존 방식 | vLLM | 처리량 향상 |
|-----------|-----------|------|-------------|
| 4 | 기준 | 1.4× | +40% |
| 8 | 기준 | 1.9× | +90% |
| 16 | OOM | 정상 동작 | - |

---

## 9. 한계점 및 후속 연구

### 9.1 현재 한계점

1. **블록 크기 고정**: 워크로드에 따른 최적 블록 크기 다름
   - 긴 시퀀스: 큰 블록이 효율적
   - 짧은 시퀀스: 작은 블록이 낭비 감소

2. **Prefill 단계 최적화 부족**: 첫 토큰 생성이 느림
   - 긴 prompt에서 병목

3. **Multi-GPU 확장성**: 분산 환경에서 블록 관리 복잡

4. **동적 메모리 요구사항**: 요청 수가 급변하면 성능 변동

### 9.2 후속 연구 및 발전

1. **SGLang (2024)**: Radix Tree 기반 prefix 공유
   - vLLM보다 더 효율적인 prefix 캐싱

2. **Splitwise**: Prefill과 Decode 분리
   - 서로 다른 GPU에서 실행

3. **ChunkAttention**: 청크 단위 스케줄링
   - Prefill 단계 최적화

4. **DistServe**: 분산 서빙 최적화
   - 여러 노드에서 효율적 블록 관리

### 9.3 vLLM의 영향

- **업계 표준**: LLM 서빙의 de facto standard
- **오픈소스 생태계**: 활발한 커뮤니티 기여
- **후속 연구 기반**: 많은 최적화 기법이 vLLM 위에 구축

---

## 10. 핵심 요약

### 기억해야 할 것들

1. **PagedAttention**: KV 캐시를 블록으로 나누어 비연속 저장
2. **Block Table**: 논리 블록 → 물리 블록 매핑
3. **Copy-on-Write**: 공유 prefix의 메모리 절약
4. **Continuous Batching**: 동적 배치 구성

### 핵심 수치

| 지표 | 기존 | vLLM | 개선 |
|------|------|------|------|
| 메모리 낭비 | 60-80% | <4% | 15-20× |
| 처리량 | 기준 | 2-4× | +100-300% |
| Beam Search 메모리 | 기준 | -55% | 2.2× |

### 실무 체크리스트

```python
# 1. vLLM 설치
# pip install vllm

# 2. 기본 사용
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# 3. 최적화 옵션
llm = LLM(
    model="...",
    gpu_memory_utilization=0.9,  # GPU 메모리 활용률
    enable_prefix_caching=True,   # Prefix 캐싱
    max_num_seqs=256,             # 최대 동시 요청
)

# 4. 서버 모드
# python -m vllm.entrypoints.openai.api_server --model ...
```

---

## 참고 자료

1. [vLLM 논문](https://arxiv.org/abs/2309.06180)
2. [공식 GitHub 저장소](https://github.com/vllm-project/vllm)
3. [vLLM 블로그](https://blog.vllm.ai/2023/06/20/vllm.html)
4. [vLLM 문서](https://docs.vllm.ai/)
5. [SOSP 2023 발표](https://www.youtube.com/watch?v=5ZlavKF_98U)

---

*이전 리뷰: [FlashAttention-3](./006_FlashAttention-3.md)*
*다음 리뷰: [Splitwise](./005_Splitwise.md)*
