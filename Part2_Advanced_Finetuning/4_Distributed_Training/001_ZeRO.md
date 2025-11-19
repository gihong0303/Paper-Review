# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

**논문 발표**: 2019년 (SC 2020)
**저자**: Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He
**소속**: Microsoft
**논문 링크**: [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
**공식 구현**: [DeepSpeed](https://github.com/microsoft/DeepSpeed)

---

## 한 줄 요약
> Data Parallelism에서 메모리 중복을 제거하는 3단계 최적화로, 모델 병렬화 없이도 1조 파라미터 모델 학습을 가능하게 한 분산 학습의 핵심 기술

---

## 1. 문제: Data Parallelism의 메모리 낭비

### 1.1 메모리 구성

```
모델 학습 메모리 = 모델 상태 + 잔여 상태

모델 상태 (16bit 기준):
- Parameters: 2Ψ bytes
- Gradients: 2Ψ bytes
- Optimizer states (Adam): 12Ψ bytes
  - fp32 params: 4Ψ
  - momentum: 4Ψ
  - variance: 4Ψ

총: 16Ψ bytes (Ψ = 파라미터 수)
```

### 1.2 Data Parallelism의 문제

N개 GPU에서 **모든 GPU가 동일한 16Ψ를 저장**

1.5B 모델, 32 GPU:
- 이론적 필요: 24GB
- 실제 사용: 32 × 24GB = **768GB** (낭비!)

---

## 2. ZeRO의 3단계

### 2.1 ZeRO-1: Optimizer State Partitioning

```
기존: 모든 GPU가 optimizer state 저장
ZeRO-1: N개 GPU에 분산 저장

메모리: 16Ψ → 4Ψ + 12Ψ/N
```

### 2.2 ZeRO-2: + Gradient Partitioning

```
ZeRO-2: Optimizer state + Gradient 분산

메모리: 4Ψ → 2Ψ + (2Ψ + 12Ψ)/N
```

### 2.3 ZeRO-3: + Parameter Partitioning

```
ZeRO-3: 모든 것을 분산

메모리: 2Ψ → 16Ψ/N

완전한 분산!
```

---

## 3. 시각화

```
ZeRO 단계별 GPU 메모리:

           GPU 0    GPU 1    GPU 2    GPU 3

기존 DP:   [P G O]  [P G O]  [P G O]  [P G O]
           전부 복제 (낭비!)

ZeRO-1:    [P G O1] [P G O2] [P G O3] [P G O4]
           Optimizer만 분산

ZeRO-2:    [P G1O1] [P G2O2] [P G3O3] [P G4O4]
           +Gradient 분산

ZeRO-3:    [P1G1O1] [P2G2O2] [P3G3O3] [P4G4O4]
           모두 분산
```

---

## 4. 통신 비용

### 4.1 ZeRO-1 & 2

통신량 = 기존 Data Parallelism과 동일!

### 4.2 ZeRO-3

추가 통신:
- Forward: parameter all-gather
- Backward: parameter all-gather

약 1.5× 통신량 증가

---

## 5. 실험 결과

### 5.1 메모리 효율

| 방법 | 최대 모델 크기 (8 GPU) |
|------|------------------------|
| Data Parallel | 1.4B |
| ZeRO-1 | 6B |
| ZeRO-2 | 8B |
| **ZeRO-3** | **13B** |

### 5.2 1000억 파라미터

ZeRO-3로 100B 모델 학습 달성 (400 GPU)

---

## 6. ZeRO-Offload & ZeRO-Infinity

### 6.1 ZeRO-Offload

GPU → CPU 메모리로 offload:
```
GPU: 활성화, 연산
CPU: Optimizer state, Gradient
```

### 6.2 ZeRO-Infinity

GPU → CPU → NVMe SSD로 확장:
- 무한한 모델 크기 가능!

---

## 7. 사용법 (DeepSpeed)

```python
# deepspeed_config.json
{
    "zero_optimization": {
        "stage": 3,  # ZeRO-3
        "offload_optimizer": {
            "device": "cpu"
        },
        "offload_param": {
            "device": "cpu"
        }
    }
}

# 학습
deepspeed train.py --deepspeed_config deepspeed_config.json
```

---

## 8. 핵심 요약

### 기억해야 할 것들

1. **핵심**: 메모리 중복 제거
2. **3단계**: Optimizer → Gradient → Parameter 분산
3. **결과**: N배 메모리 효율 (N = GPU 수)
4. **의의**: 1조 파라미터 학습 가능

### ZeRO 선택 가이드

| 단계 | 메모리 절약 | 통신 비용 | 사용 시점 |
|------|-------------|-----------|-----------|
| ZeRO-1 | 4× | 동일 | 기본 |
| ZeRO-2 | 8× | 동일 | 더 큰 모델 |
| ZeRO-3 | N× | 1.5× | 매우 큰 모델 |

---

## 참고 자료

1. [ZeRO 논문](https://arxiv.org/abs/1910.02054)
2. [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
3. [DeepSpeed 문서](https://www.deepspeed.ai/)

---

*다음 리뷰: [Megatron-LM](./002_Megatron-LM.md)*
