# Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

**논문 발표**: 2019년 (SC 2020)
**저자**: Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro
**소속**: NVIDIA
**논문 링크**: [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
**공식 구현**: [GitHub](https://github.com/NVIDIA/Megatron-LM)

---

## 한 줄 요약
> Transformer 레이어를 여러 GPU에 분산하는 효율적인 Tensor Parallelism 기법으로, 단일 GPU 한계를 넘어 수십억 파라미터 모델 학습을 가능하게 함

---

## 1. Model Parallelism이란?

### 1.1 Data vs Model Parallelism

```
Data Parallelism:
- 데이터를 분산
- 모델을 복제
- 배치 크기 증가

Model Parallelism:
- 모델을 분산
- 데이터는 동일
- 모델 크기 증가
```

### 1.2 Tensor vs Pipeline

```
Tensor Parallelism:
- 레이어 내부를 분할
- 동시 연산

Pipeline Parallelism:
- 레이어별로 분할
- 순차 연산
```

Megatron은 **Tensor Parallelism**

---

## 2. Tensor Parallelism

### 2.1 Linear Layer 분할

$$Y = XA$$

A를 열 방향으로 분할:
$$Y = X[A_1, A_2] = [XA_1, XA_2] = [Y_1, Y_2]$$

각 GPU가 $A_i$ 담당

### 2.2 MLP 분할

```
MLP: Y = GeLU(XA) × B

GPU 1: GeLU(XA₁) × B₁
GPU 2: GeLU(XA₂) × B₂

최종: AllReduce(Y₁ + Y₂)
```

### 2.3 Attention 분할

```
각 Head를 다른 GPU에 할당

GPU 1: Head 1-4
GPU 2: Head 5-8
...

Q, K, V projection도 분할
```

---

## 3. 통신 최적화

### 3.1 통신 횟수

각 Transformer 레이어당:
- Forward: 2번 AllReduce
- Backward: 2번 AllReduce

### 3.2 통신과 연산 중첩

비동기 통신으로 숨김

---

## 4. 실험 결과

### 4.1 스케일링 효율

| GPU 수 | 효율 |
|--------|------|
| 1 | 100% |
| 2 | 98% |
| 4 | 94% |
| 8 | 76% |

### 4.2 학습 성능

8.3B GPT-2 모델을 512 GPU에서 학습:
- 15.1 PetaFLOPs
- 76% 하드웨어 효율

---

## 5. 3D Parallelism

### 5.1 구성

```
3D Parallelism = Tensor + Pipeline + Data

Tensor: GPU 내 모델 분할
Pipeline: GPU 간 레이어 분할
Data: 노드 간 배치 분할
```

### 5.2 GPT-3 스케일

175B 모델:
- Tensor: 8-way (NVLink)
- Pipeline: 64-way
- Data: 8-way
- 총: 4096 GPU

---

## 6. 사용법

```python
# Megatron-LM 설정
args = {
    'tensor_model_parallel_size': 8,
    'pipeline_model_parallel_size': 4,
    'num_layers': 96,
    'hidden_size': 12288,
    'num_attention_heads': 96,
}

# DeepSpeed와 통합
from megatron import get_model
model = get_model(args)
```

---

## 7. ZeRO와의 관계

### 7.1 상호 보완

```
ZeRO: Data Parallelism 최적화
Megatron: Model Parallelism

조합: ZeRO + Megatron = 최적의 효율
```

### 7.2 권장 구성

- 작은 모델 (<10B): ZeRO-3
- 중간 모델 (10-100B): Tensor + ZeRO
- 큰 모델 (>100B): 3D Parallelism

---

## 8. 핵심 요약

### 기억해야 할 것들

1. **핵심**: Tensor를 GPU에 분산
2. **방법**: Linear/Attention 열/행 분할
3. **효율**: 8-way에서 76%
4. **의의**: 모델 크기 한계 돌파

### Megatron vs ZeRO

| 측면 | Megatron | ZeRO |
|------|----------|------|
| 타입 | Model Parallel | Data Parallel |
| 분할 | Tensor | Memory state |
| 장점 | 속도 | 구현 간단 |
| 통신 | AllReduce | AllGather |

---

## 참고 자료

1. [Megatron-LM 논문](https://arxiv.org/abs/1909.08053)
2. [Megatron-LM v2](https://arxiv.org/abs/2104.04473)
3. [GitHub](https://github.com/NVIDIA/Megatron-LM)

---

*이전 리뷰: [ZeRO](./001_ZeRO.md)*
*Part 2 완료! 다음: [Part 3 - Domain Adaptation](../../Part3_Domain_Adaptation/)*
