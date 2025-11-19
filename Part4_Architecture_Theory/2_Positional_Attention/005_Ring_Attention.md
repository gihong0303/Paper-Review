# Ring Attention with Blockwise Transformers

**논문 발표**: 2023년
**저자**: Hao Liu, Matei Zaharia, Pieter Abbeel
**소속**: UC Berkeley
**논문 링크**: [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)

---

## 한 줄 요약
> 시퀀스를 여러 디바이스에 분산하고 KV를 링 형태로 전달하여, 단일 GPU 메모리를 초과하는 무한대에 가까운 컨텍스트 길이 지원

---

## 1. 문제: 메모리 한계

### 1.1 Attention 메모리

```
시퀀스 길이 N, 디바이스 메모리 M:

Full attention: O(N²) 메모리
→ 100K 시퀀스면 80GB+ 필요

단일 GPU로는 긴 시퀀스 불가능
```

### 1.2 Ring Attention 해결

```
N개 디바이스에 시퀀스 분산:
각 디바이스: N/devices 토큰만 저장
KV를 링 형태로 순환
```

---

## 2. 핵심 아이디어

### 2.1 Ring 구조

```
Device 0 → Device 1 → Device 2 → Device 3
    ↑                                    ↓
    ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←

KV blocks가 링을 따라 순환
```

### 2.2 Blockwise Computation

```python
# 각 디바이스에서
for step in range(num_devices):
    # 현재 가진 KV block으로 attention
    local_output = attention(Q_local, K_current, V_current)

    # KV를 다음 디바이스로 전송
    K_current, V_current = ring_send_recv(K_current, V_current)

    # 결과 누적
    output += local_output
```

---

## 3. 구현

```python
def ring_attention(Q, K, V, num_devices):
    """
    각 디바이스가 시퀀스의 1/num_devices 담당
    """
    device_id = get_device_id()
    seq_per_device = seq_len // num_devices

    # 내 Q block
    Q_local = Q[device_id * seq_per_device : (device_id + 1) * seq_per_device]

    # 초기 KV는 내 것
    K_current = K[device_id * seq_per_device : (device_id + 1) * seq_per_device]
    V_current = V[device_id * seq_per_device : (device_id + 1) * seq_per_device]

    output = 0
    normalizer = 0

    for step in range(num_devices):
        # Local attention
        scores = Q_local @ K_current.T
        weights = softmax(scores)
        local_out = weights @ V_current

        # Online softmax normalization
        output, normalizer = online_softmax_update(
            output, normalizer, local_out, weights
        )

        # Ring communication
        K_current = ring_send_recv(K_current, (device_id + 1) % num_devices)
        V_current = ring_send_recv(V_current, (device_id + 1) % num_devices)

    return output / normalizer
```

---

## 4. Online Softmax

### 4.1 왜 필요한가?

```
일반 softmax: 모든 score 필요
Ring: block별로 계산

→ Online softmax로 점진적 정규화
```

### 4.2 수식

$$m_{new} = \max(m_{old}, m_{block})$$
$$l_{new} = e^{m_{old} - m_{new}} l_{old} + e^{m_{block} - m_{new}} l_{block}$$
$$o_{new} = \frac{e^{m_{old} - m_{new}} o_{old} + e^{m_{block} - m_{new}} o_{block}}{l_{new}}$$

---

## 5. 효과

### 5.1 Context Length Scaling

```
디바이스 수에 비례:
- 8 GPUs: 8× context
- 64 GPUs: 64× context

1M+ 토큰 가능!
```

### 5.2 메모리 효율

```
각 디바이스:
- Q: N/D tokens
- KV: N/D tokens (순환)

총 메모리: O(N/D)
```

---

## 6. 통신 비용

### 6.1 통신량

```
각 step: 2 × (N/D) × d bytes

D steps → 총 2 × N × d bytes
= Full attention과 동일
```

### 6.2 통신-연산 겹침

```
연산하면서 동시에 통신:
→ 오버헤드 최소화
```

---

## 7. 핵심 요약

### 기억해야 할 것들

1. **핵심**: KV를 링 형태로 순환
2. **효과**: 디바이스 수만큼 context 확장
3. **방법**: Blockwise attention + Online softmax
4. **결과**: 1M+ 토큰 context 가능

### 실무 팁

```
긴 문서 처리:
- 책 전체
- 긴 코드 레포지토리
- 비디오 transcript

Ring attention으로 해결!
```

---

## 참고 자료

1. [Ring Attention 논문](https://arxiv.org/abs/2310.01889)

---

*이전 리뷰: [Sliding Window](./004_Sliding_Window.md)*
*다음 섹션: [Mixture of Experts](../3_MoE/)*
