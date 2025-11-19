# BitNet b1.58: The Era of 1-bit LLMs

**논문 발표**: 2024년 2월
**저자**: Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei
**소속**: Microsoft Research
**논문 링크**: [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)

---

## 한 줄 요약
> 가중치를 {-1, 0, 1} 세 값(1.58bit)으로 양자화하여, FP16 LLaMA와 동등한 성능을 달성하면서 메모리 3.55배, 에너지 71배, 지연시간 2.71배 개선

---

## 1. 핵심 혁신: 1.58-bit Quantization

### 1.1 왜 1.58bit인가?

$$\log_2(3) = 1.58 \text{ bits}$$

세 가지 값 {-1, 0, 1}을 표현하는 데 필요한 정보량

### 1.2 Ternary Quantization

$$\tilde{W} = \text{RoundClip}\left(\frac{W}{\gamma + \epsilon}, -1, 1\right)$$

$$\gamma = \frac{1}{nm}\sum_{ij}|W_{ij}|$$

```python
def absmean_quantize(W):
    gamma = W.abs().mean()
    W_scaled = W / gamma
    W_ternary = W_scaled.round().clamp(-1, 1)  # {-1, 0, 1}
    return W_ternary, gamma
```

### 1.3 0의 도입이 중요한 이유

```
BitNet (1bit):   {-1, +1}    → 정보 손실
BitNet b1.58:   {-1, 0, +1} → 특징 필터링 가능

0의 역할:
- 중요하지 않은 특징 무시
- 네트워크의 sparsity 증가
- 명시적인 "off" 신호
```

---

## 2. 아키텍처

### 2.1 전체 구조

```
LLaMA 아키텍처 + BitLinear 교체

RMSNorm → BitLinear(Q,K,V) → Attention → BitLinear(O)
       → BitLinear(gate,up) → SiLU → BitLinear(down)
```

### 2.2 BitLinear b1.58

```python
def BitLinear_b158(x, W, gamma):
    # 활성화: absmax 양자화 (8bit)
    x_scale = x.abs().max()
    x_quant = (x * 127 / x_scale).round().clamp(-128, 127)

    # 가중치: 이미 {-1, 0, 1}로 저장됨
    # → 곱셈이 덧셈/뺄셈으로 대체!

    # 행렬 곱
    y = x_quant @ W  # 정수 연산

    # 역양자화
    return y * (gamma * x_scale / 127)
```

### 2.3 곱셈 없는 행렬 곱

{-1, 0, 1} × INT8:
- ×(-1) = 부호 반전
- ×0 = 스킵
- ×1 = 그대로

**곱셈 연산 완전 제거!** → 덧셈만으로 행렬 곱

---

## 3. 학습 방법

### 3.1 처음부터 학습

PTQ가 아닌 **QAT (Quantization-Aware Training)**:
- 처음부터 1.58bit로 학습
- STE (Straight-Through Estimator) 사용

### 3.2 학습 레시피

```python
# BitNet b1.58 학습 설정
config = {
    'weight_quant': 'absmean_ternary',  # {-1, 0, 1}
    'activation_quant': 'absmax_int8',   # INT8
    'optimizer': 'AdamW',
    'learning_rate': 1.5e-4,
    'warmup': 375 steps,
    'total_tokens': 100B
}
```

---

## 4. 실험 결과

### 4.1 FP16 LLaMA와 비교

| 모델 | PPL | ARC-c | ARC-e | HS | WG |
|------|-----|-------|-------|----|----|
| LLaMA 3B | 8.24 | 38.1 | 69.7 | 71.0 | 66.1 |
| **BitNet b1.58 3B** | **8.34** | **37.8** | **69.6** | **70.9** | **65.2** |

**성능이 거의 동일!**

### 4.2 효율성 비교 (3B 모델)

| 메트릭 | LLaMA FP16 | BitNet b1.58 | 개선 |
|--------|------------|--------------|------|
| 메모리 | 7.89 GB | 2.22 GB | **3.55×** |
| 지연시간 | 24.8 ms | 9.1 ms | **2.71×** |
| 에너지 (matmul) | 27.3 J | 0.38 J | **71×** |

### 4.3 스케일링 법칙

모델 크기가 커질수록 BitNet b1.58이 유리:

```
700M:  BitNet ≈ LLaMA
1.3B:  BitNet ≈ LLaMA
3B:    BitNet ≈ LLaMA
7B+:   BitNet > LLaMA (예상)
```

---

## 5. 하드웨어 최적화 가능성

### 5.1 전용 하드웨어

```
현재 GPU: FP16/INT8 연산 최적화
미래:     {-1, 0, 1} 전용 하드웨어

잠재적 이점:
- 곱셈기 제거 → 면적/전력 절감
- 메모리 대역폭 3.55배 감소
- 에너지 효율 71배 향상
```

### 5.2 NPU/엣지 디바이스

모바일/IoT에서 LLM 구동 가능성:
- 메모리 요구량 대폭 감소
- 배터리 수명 연장
- 실시간 추론

---

## 6. 쉬운 예시로 이해하기

### 6.1 투표 비유

```
FP16:       복잡한 가중 투표 (각자 다른 점수)
INT8:       100단계 점수
INT4:       16단계 점수
BitNet 1.58: 찬성/반대/기권 {+1, -1, 0}
```

세 가지만으로도 충분히 의사결정 가능!

### 6.2 스위치 비유

```
기존: 다이얼 스위치 (무한 단계 조절)
b1.58: 3-way 스위치 (On/Off/Neutral)

놀랍게도 3-way로 충분한 성능!
```

---

## 7. 한계점 및 미래 방향

### 7.1 현재 한계

1. **학습 필요**: 기존 모델 변환 불가
2. **하드웨어 미성숙**: 전용 가속기 필요
3. **생태계**: 아직 초기 단계

### 7.2 미래 방향

- **전용 하드웨어**: 1-bit 연산 가속기
- **더 큰 모델**: 70B+ 스케일 검증
- **혼합 정밀도**: 일부 레이어만 1.58bit

---

## 8. 핵심 요약

### 기억해야 할 것들

1. **핵심 혁신**: {-1, 0, 1} 삼진법 양자화
2. **0의 역할**: 특징 필터링, sparsity
3. **성능**: FP16과 동등
4. **효율성**: 메모리 3.55×, 에너지 71×

### 비교 표

| 버전 | 값 | 성능 | 효율성 |
|------|-----|------|--------|
| BitNet 1.0 | {-1, 1} | 저하 | 좋음 |
| **BitNet b1.58** | {-1, 0, 1} | **FP16 동등** | **최고** |

### 의의

> "1-bit LLM의 시대가 열렸다"

PTQ를 넘어, 처음부터 극단적 양자화로 학습하는 새로운 패러다임

---

## 참고 자료

1. [BitNet b1.58 논문](https://arxiv.org/abs/2402.17764)
2. [Microsoft Research 블로그](https://www.microsoft.com/en-us/research/)
3. [Karpathy의 분석](https://twitter.com/karpathy)

---

*이전 리뷰: [BitNet](./006_BitNet.md)*
*다음 리뷰: [Q-GaLore](./008_Q-GaLore.md)*
