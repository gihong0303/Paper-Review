# Jamba: A Hybrid Transformer-Mamba Language Model

**논문 발표**: 2024년
**저자**: Opher Lieber, Barak Lenz, Hofit Bata, et al.
**소속**: AI21 Labs
**논문 링크**: [arXiv:2403.19887](https://arxiv.org/abs/2403.19887)
**공식 모델**: [HuggingFace](https://huggingface.co/ai21labs/Jamba-v0.1)

---

## 한 줄 요약
> Transformer, Mamba, MoE를 결합한 하이브리드 아키텍처로, 256K 컨텍스트를 80GB GPU 하나에서 처리

---

## 1. 하이브리드 구조

### 1.1 왜 하이브리드인가?

```
Transformer 장점:
- In-context learning 강함
- 전역 attention

Mamba 장점:
- O(N) 복잡도
- 긴 컨텍스트 효율적

둘 다 사용!
```

### 1.2 구조

```
Layer 1: Mamba
Layer 2: Mamba
Layer 3: Mamba
Layer 4: Attention + MoE ← 1/8만 Attention
Layer 5: Mamba
...

비율: Mamba 7 : Attention 1
```

---

## 2. 메모리 효율

### 2.1 KV 캐시 비교

| 모델 | 256K 컨텍스트 KV 캐시 |
|------|----------------------|
| Mixtral 8x7B | 128 GB |
| **Jamba** | **4 GB** |

32배 감소!

### 2.2 이유

```
대부분 Mamba 레이어:
- State만 저장 (고정 크기)
- 시퀀스 길이에 무관

일부 Attention:
- KV 캐시 필요
- 하지만 1/8만
```

---

## 3. 구현

```python
class JambaBlock(nn.Module):
    def __init__(self, config, layer_idx):
        # 매 8번째 레이어만 Attention
        if layer_idx % 8 == 0:
            self.mixer = AttentionMoE(config)
        else:
            self.mixer = MambaBlock(config)

    def forward(self, x):
        return self.mixer(x)
```

---

## 4. 실험 결과

### 4.1 성능

| 모델 | 활성 파라미터 | MMLU | HellaSwag |
|------|---------------|------|-----------|
| Mixtral 8x7B | 12.9B | 70.6 | 82.6 |
| LLaMA 2 70B | 70B | 68.9 | 85.3 |
| **Jamba** | **12B** | **67.4** | **87.1** |

### 4.2 Long Context

```
256K context:
- Needle-in-haystack: 100%
- 80GB GPU 1개에서 동작

Mixtral은 메모리 부족!
```

---

## 5. 핵심 요약

### 기억해야 할 것들

1. **구조**: Mamba + Attention + MoE
2. **비율**: Mamba 7 : Attention 1
3. **효율**: 256K context, 단일 GPU
4. **성능**: Mixtral급

### 설정

| 항목 | 값 |
|------|-----|
| 총 파라미터 | 52B |
| 활성 파라미터 | 12B |
| Context | 256K |
| Mamba:Attn | 7:1 |

---

## 참고 자료

1. [Jamba 논문](https://arxiv.org/abs/2403.19887)

---

*이전 리뷰: [Mamba](./002_Mamba.md)*
*다음 리뷰: [Mamba-2](./004_Mamba-2.md)*
