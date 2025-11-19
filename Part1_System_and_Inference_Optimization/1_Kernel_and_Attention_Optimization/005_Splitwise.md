# Splitwise: Efficient Generative LLM Inference Using Phase Splitting

**논문 발표**: 2023년 (ISCA 2024)
**저자**: Pratyush Patel, Esha Choukse, Chaojie Zhang, Íñigo Goiri, Aashaka Shah, Saeed Maleki, Ricardo Bianchini
**소속**: Microsoft Azure
**논문 링크**: [arXiv:2311.18677](https://arxiv.org/abs/2311.18677)

---

## 한 줄 요약
> LLM 추론의 Prefill과 Decode 단계가 근본적으로 다른 특성을 가진다는 점에 착안하여, 두 단계를 서로 다른 하드웨어에서 실행함으로써 클러스터 전체의 효율을 1.4배 이상 향상

---

## 1. 문제 정의: Prefill과 Decode의 불일치

### 1.1 두 단계의 근본적 차이

```
┌─────────────────────────────────────────────────────┐
│           LLM 추론의 두 단계                         │
├────────────────────┬────────────────────────────────┤
│      Prefill       │          Decode                │
├────────────────────┼────────────────────────────────┤
│ • Compute-bound    │ • Memory-bound                 │
│ • 높은 연산 강도    │ • 낮은 연산 강도               │
│ • GPU 활용률 높음   │ • GPU 활용률 낮음              │
│ • 짧은 지연시간     │ • 긴 누적 지연시간             │
│ • 한 번 실행        │ • 반복 실행 (토큰당 한 번)     │
├────────────────────┼────────────────────────────────┤
│ 최적 하드웨어:      │ 최적 하드웨어:                 │
│ 고성능 GPU         │ 메모리 대역폭 높은 GPU          │
│ (A100, H100)       │ 또는 여러 개의 저가 GPU        │
└────────────────────┴────────────────────────────────┘
```

### 1.2 기존 시스템의 문제

동일한 GPU에서 두 단계를 번갈아 실행:

```
기존 방식 (동일 GPU에서):

Request 1: [Prefill][Decode][Decode][Decode]...
Request 2:         [Prefill][Decode][Decode]...

문제:
• Prefill 중 GPU는 compute-bound → 100% 활용
• Decode 중 GPU는 memory-bound → 30-50% 활용
• 전체적으로 GPU가 낭비됨
```

### 1.3 하드웨어 특성 불일치

| 지표 | Prefill에 필요 | Decode에 필요 |
|------|----------------|---------------|
| 연산 능력 | 높음 (TFLOPS) | 낮음 |
| 메모리 대역폭 | 보통 | 높음 (TB/s) |
| GPU 가격 | 고가 OK | 저가 선호 |

**핵심 통찰**: 한 GPU가 모든 것을 잘할 필요 없음!

---

## 2. Splitwise의 핵심 아이디어

### 2.1 Phase Splitting (단계 분리)

```
┌─────────────────────────────────────────────┐
│              Splitwise 아키텍처             │
├─────────────────────────────────────────────┤
│                                             │
│  Prefill Machine          Decode Machine    │
│  ┌───────────────┐       ┌───────────────┐ │
│  │ 고성능 GPU    │       │ 다수의 GPU    │ │
│  │ (A100/H100)   │       │ (A100 또는   │ │
│  │               │   →   │  저가 옵션)   │ │
│  │ Prefill만     │  KV   │ Decode만     │ │
│  │ 수행          │ Cache │ 수행          │ │
│  └───────────────┘       └───────────────┘ │
│                                             │
└─────────────────────────────────────────────┘
```

### 2.2 워크플로우

```
1. 요청 도착 → Prefill Machine

2. Prefill 실행 (Compute-bound 최적화)
   - 전체 프롬프트 처리
   - KV Cache 생성

3. KV Cache 전송 → Decode Machine
   (고속 네트워크: NVLink, InfiniBand)

4. Decode 실행 (Memory-bound 최적화)
   - 토큰 하나씩 생성
   - 배치 최대화

5. 완료 → 응답 반환
```

### 2.3 왜 효과적인가?

**Prefill Machine**:
- 항상 compute-bound 워크로드만 처리
- GPU 연산 유닛 100% 활용
- 빠르게 처리하고 다음 요청으로

**Decode Machine**:
- 큰 배치로 throughput 극대화
- 메모리 대역폭 병목 완화
- 여러 시퀀스 동시 처리

---

## 3. KV Cache 전송 문제

### 3.1 전송 비용

LLaMA-70B의 KV Cache 크기 (시퀀스 길이 1024, FP16):
$$\text{KV size} = 2 \times 80 \times 8192 \times 1024 \times 2 = 2.6 \text{ GB}$$

A100 NVLink 대역폭: 600 GB/s
전송 시간: 2.6 GB / 600 GB/s ≈ 4.3ms

### 3.2 전송 최적화

**1. 파이프라이닝**:
```
레이어 0 전송 ──────→
  레이어 1 전송 ──────→
    레이어 2 전송 ──────→
      ...
        Decode 시작 (레이어 0 도착 즉시)
```

**2. 압축**:
- Quantization: FP16 → INT8 (50% 압축)
- Sparse encoding: 중요하지 않은 값 생략

**3. 지역성 활용**:
- 같은 서버 내 전송 선호
- NUMA-aware 배치

### 3.3 전송 vs 재계산 Trade-off

| 방법 | 비용 | 사용 시점 |
|------|------|-----------|
| 전송 | 네트워크 대역폭 | 대역폭 여유 있을 때 |
| 재계산 | GPU 연산 | 네트워크 병목 시 |

Splitwise는 상황에 따라 적응적으로 선택

---

## 4. 스케줄링 최적화

### 4.1 Prefill Machine 스케줄링

목표: Prefill 처리량 최대화

```python
def schedule_prefill(requests):
    # 1. 프롬프트 길이 기준 정렬 (짧은 것 먼저)
    requests.sort(key=lambda r: r.prompt_length)

    # 2. 배치 구성 (연산 자원 고려)
    batch = []
    total_tokens = 0

    for req in requests:
        if total_tokens + req.prompt_length <= MAX_BATCH_TOKENS:
            batch.append(req)
            total_tokens += req.prompt_length
        else:
            yield batch
            batch = [req]
            total_tokens = req.prompt_length
```

### 4.2 Decode Machine 스케줄링

목표: 메모리 대역폭 활용 극대화 (큰 배치)

```python
def schedule_decode(sequences):
    # 1. 최대한 많은 시퀀스를 배치에 포함
    # (메모리 제한까지)

    batch = []
    memory_used = 0

    for seq in sequences:
        kv_memory = estimate_kv_memory(seq)
        if memory_used + kv_memory <= GPU_MEMORY:
            batch.append(seq)
            memory_used += kv_memory
        else:
            break

    return batch

    # 2. Continuous batching 적용
    # 완료된 시퀀스 즉시 제거, 새 시퀀스 추가
```

### 4.3 로드 밸런싱

```
시나리오: Prefill 3대, Decode 5대

Load Balancer
    │
    ├─→ Prefill 1 ─┐
    ├─→ Prefill 2 ─┼─→ Decode 1
    └─→ Prefill 3 ─┤   Decode 2
                   │   Decode 3
                   │   Decode 4
                   └─→ Decode 5

비율 결정:
• 평균 Prefill 시간 / 평균 Decode 시간
• 예: Prefill 10ms, Decode 100ms (10 토큰)
  → Prefill:Decode = 1:10 비율로 배치
```

---

## 5. 클러스터 구성 최적화

### 5.1 이기종 클러스터 구성

```
┌─────────────────────────────────────────┐
│        최적화된 클러스터 구성            │
├─────────────────────────────────────────┤
│                                         │
│  Prefill Tier                           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ H100    │ │ H100    │ │ H100    │   │
│  │ (고가)  │ │ (고가)  │ │ (고가)  │   │
│  └─────────┘ └─────────┘ └─────────┘   │
│       │           │           │        │
│       └───────────┼───────────┘        │
│                   ▼                     │
│              KV Cache                   │
│                   │                     │
│       ┌───────────┼───────────┐        │
│       ▼           ▼           ▼        │
│  Decode Tier                            │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      │
│  │A10G │ │A10G │ │A10G │ │A10G │ ...  │
│  │(저가)│ │(저가)│ │(저가)│ │(저가)│      │
│  └─────┘ └─────┘ └─────┘ └─────┘      │
│                                         │
└─────────────────────────────────────────┘
```

### 5.2 비용 최적화

| GPU 타입 | 역할 | 가격 ($/hr) | 활용률 |
|----------|------|-------------|--------|
| H100 | Prefill | $4.0 | ~100% |
| A10G | Decode | $1.0 | ~80% |

**동종 구성 (H100만 8대)**:
- 비용: $32/hr
- 활용률: ~60%

**이종 구성 (H100 2대 + A10G 12대)**:
- 비용: $8 + $12 = $20/hr
- 활용률: ~90%
- **비용 대비 성능 1.4배 향상**

### 5.3 Spot Instance 활용

Decode 머신은 Spot Instance로 비용 절감 가능:
- Preemption 시 KV Cache만 재생성하면 됨
- 전체 추론을 다시 시작할 필요 없음

---

## 6. 쉬운 예시로 이해하기

### 6.1 레스토랑 비유

**기존 방식**: 모든 요리사가 전체 과정 담당
- 재료 준비 (Prefill): 기술 필요, 짧은 시간
- 조리 (Decode): 시간 오래 걸림

문제: 실력 좋은 요리사가 단순 조리에 시간 낭비

**Splitwise**: 역할 분담
- 수셰프 (고급): 재료 준비만 담당
- 주니어 셰프 (다수): 조리 담당
- 효율 극대화!

### 6.2 공장 생산 라인 비유

**기존 방식**:
- 모든 기계가 전체 공정 수행
- 복잡한 초기 가공 + 단순 반복 작업

**Splitwise**:
- 고가의 정밀 기계: 초기 가공 (Prefill)
- 저가의 단순 기계 다수: 반복 작업 (Decode)
- 생산성 향상, 비용 절감

### 6.3 숫자 예시

LLaMA-70B 서빙, 1000 req/s 목표:

**동종 구성 (A100 32대)**:
- Prefill 처리량: 2000 req/s
- Decode 처리량: 1000 req/s (병목)
- GPU 비용: 32 × $3 = $96/hr

**Splitwise (A100 8대 Prefill + A100 16대 Decode)**:
- Prefill: 2000 req/s
- Decode: 1500 req/s
- 사용: 24대로 동일 성능
- GPU 비용: 24 × $3 = $72/hr
- **25% 비용 절감**

---

## 7. 실험 결과

### 7.1 처리량 비교

| 모델 | 시스템 | 처리량 (req/s) | 향상 |
|------|--------|----------------|------|
| OPT-66B | vLLM | 기준 | 1× |
| OPT-66B | Splitwise | 1.4× | **1.4×** |
| LLaMA-70B | vLLM | 기준 | 1× |
| LLaMA-70B | Splitwise | 1.5× | **1.5×** |

### 7.2 비용 효율

동일 처리량 달성을 위한 GPU 수:

| 시스템 | GPU 수 | 비용 비율 |
|--------|--------|-----------|
| 동종 (모두 A100) | 32 | 100% |
| Splitwise (A100 + A100) | 24 | 75% |
| Splitwise (H100 + A10G) | 20 | **65%** |

### 7.3 지연시간

| 시나리오 | P50 지연시간 | P99 지연시간 |
|----------|--------------|--------------|
| 기존 | 120ms | 350ms |
| Splitwise | 100ms | 280ms |

Prefill-Decode 간섭 제거로 지연시간도 개선

---

## 8. 구현 고려사항

### 8.1 네트워크 요구사항

```python
def check_network_feasibility(kv_cache_size, sla_latency):
    """
    네트워크가 KV Cache 전송을 SLA 내에 처리할 수 있는지 확인
    """
    required_bandwidth = kv_cache_size / sla_latency

    # 예: 2GB / 10ms = 200 GB/s
    # NVLink (600 GB/s) → OK
    # InfiniBand 400G (50 GB/s) → 어려움

    return required_bandwidth < available_bandwidth
```

### 8.2 메모리 관리

```python
class SplitWiseKVManager:
    def __init__(self):
        self.prefill_kv_buffer = {}  # 임시 저장
        self.decode_kv_cache = {}    # 장기 저장

    def on_prefill_complete(self, request_id, kv_cache):
        # Prefill 완료 후
        self.prefill_kv_buffer[request_id] = kv_cache
        self.initiate_transfer(request_id)

    def initiate_transfer(self, request_id):
        # 비동기 전송
        async_transfer(
            self.prefill_kv_buffer[request_id],
            destination=decode_machine
        )

    def on_transfer_complete(self, request_id, kv_cache):
        # Decode 머신에서 수신
        self.decode_kv_cache[request_id] = kv_cache
        # Decode 시작 가능
```

### 8.3 장애 복구

```python
def handle_failure(failed_machine, requests):
    if failed_machine.type == "prefill":
        # Prefill 다시 실행 (빠름)
        reschedule_prefill(requests)

    elif failed_machine.type == "decode":
        # KV Cache가 있으면 다른 Decode 머신으로 전송
        if kv_cache_available(requests):
            migrate_to_other_decode(requests)
        else:
            # Prefill부터 다시
            reschedule_prefill(requests)
```

---

## 9. 다른 기술과의 관계

### 9.1 vs Orca/Continuous Batching

- **Orca**: 단일 머신에서 Prefill/Decode interleaving
- **Splitwise**: 완전히 분리된 머신에서 실행

상호 보완적: Splitwise의 각 머신 내에서 Continuous Batching 적용

### 9.2 vs Speculative Decoding

- **Speculative Decoding**: Decode 단계 자체를 가속
- **Splitwise**: Decode 단계의 클러스터 효율 향상

조합 가능: Decode 머신에서 Speculative Decoding 사용

### 9.3 vs Disaggregated Serving

유사한 아이디어들:
- **DistServe** (2024): 유사 접근
- **Mooncake** (2024): 더 세분화된 분리

---

## 10. 한계점 및 후속 연구

### 10.1 한계점

1. **네트워크 의존성**:
   - 고속 네트워크 필수
   - 네트워크 지연이 전체 성능 좌우

2. **구현 복잡성**:
   - 두 시스템 간 조율 필요
   - 장애 복구 복잡

3. **특정 워크로드**:
   - 긴 프롬프트에서 더 효과적
   - 짧은 프롬프트에서는 전송 오버헤드

### 10.2 후속 연구 방향

1. **더 세밀한 분리**:
   - Attention만 분리
   - 레이어별 분리

2. **적응적 분리**:
   - 워크로드에 따라 동적으로 전환
   - Prefill-Decode 경계 유동적

3. **엣지-클라우드 분리**:
   - 엣지에서 Prefill, 클라우드에서 Decode
   - 또는 그 반대

---

## 11. 핵심 요약

### 기억해야 할 것들

1. **핵심 관찰**: Prefill은 Compute-bound, Decode는 Memory-bound
2. **해결책**: 두 단계를 다른 하드웨어에서 실행
3. **이점**: 클러스터 효율 1.4배+, 비용 25-35% 절감
4. **과제**: 고속 네트워크 필요, KV Cache 전송 비용

### 언제 사용해야 하는가?

| 상황 | Splitwise 효과 |
|------|---------------|
| 대규모 클러스터 | 매우 효과적 |
| 긴 프롬프트 많음 | 효과적 |
| 고속 네트워크 있음 | 효과적 |
| 소규모/저비용 환경 | 비효율적 |

### 핵심 공식

클러스터 효율:
$$\text{Efficiency} = \frac{\text{Useful Work}}{\text{Total Capacity}}$$

기존: ~60% (Prefill/Decode 혼합)
Splitwise: ~90% (전문화)

---

## 참고 자료

1. [Splitwise 논문](https://arxiv.org/abs/2311.18677)
2. [Microsoft Research Blog](https://www.microsoft.com/en-us/research/)
3. [ISCA 2024 발표](https://www.iscaconf.org/)
4. [DistServe (유사 연구)](https://arxiv.org/abs/2401.09670)

---

*이전 리뷰: [Flash-Decoding](./004_Flash-Decoding.md)*
*다음 리뷰: [LLM.int8()](../2_Quantization/001_LLM_int8.md) - Quantization 섹션 시작*
