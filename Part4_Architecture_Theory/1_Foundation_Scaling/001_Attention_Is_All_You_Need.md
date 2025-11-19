# Attention Is All You Need

**논문 발표**: 2017년 (NeurIPS 2017)
**저자**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
**소속**: Google Brain, Google Research
**논문 링크**: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

## 한 줄 요약
> RNN과 CNN 없이 오직 Attention 메커니즘만으로 시퀀스 모델링을 수행하는 Transformer 아키텍처를 제안하여, 현대 AI의 기반을 마련

---

## 1. 혁명의 시작

### 1.1 기존 문제: RNN의 한계

```
RNN/LSTM의 문제:
1. 순차 처리 → 병렬화 불가
2. 장거리 의존성 학습 어려움
3. 학습 속도 느림

시퀀스 길이 N에 대해:
- 연산: O(N) steps (병렬화 불가)
- 메모리: gradient vanishing/exploding
```

### 1.2 Transformer의 해결책

```
Attention만 사용:
1. 완전 병렬화 가능
2. 모든 위치 직접 연결
3. 학습 속도 대폭 향상
```

---

## 2. 핵심 구성 요소

### 2.1 Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: Query [batch, heads, seq_len, d_k]
    K: Key [batch, heads, seq_len, d_k]
    V: Value [batch, heads, seq_len, d_v]
    """
    d_k = Q.shape[-1]

    # Attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Optional masking
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Softmax
    attention_weights = F.softmax(scores, dim=-1)

    # Apply to values
    output = torch.matmul(attention_weights, V)

    return output, attention_weights
```

### 2.2 왜 √d_k로 나누는가?

```
Q·K의 분산:
- Q, K가 평균 0, 분산 1이면
- Q·K의 분산 = d_k

d_k가 크면:
- dot product 값이 매우 커짐
- softmax가 극단값으로 수렴
- gradient가 거의 0

√d_k로 나누면:
- 분산이 1로 정규화
- softmax가 부드럽게 동작
```

### 2.3 Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)

        # Concat and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.W_o(attn_output)
```

### 2.4 왜 Multi-Head인가?

```
단일 Attention의 한계:
- 하나의 "관점"으로만 봄

Multi-Head의 장점:
- 여러 관점에서 동시에 attention
- Head 1: 문법적 관계
- Head 2: 의미적 관계
- Head 3: 위치적 관계
- ...
```

---

## 3. Transformer 아키텍처

### 3.1 전체 구조

```
Encoder-Decoder:

Encoder (N=6):
┌─────────────────┐
│ Multi-Head Attn │ ← Self-Attention
├─────────────────┤
│ Add & Norm      │
├─────────────────┤
│ Feed Forward    │
├─────────────────┤
│ Add & Norm      │
└─────────────────┘

Decoder (N=6):
┌─────────────────┐
│ Masked MH Attn  │ ← Causal Self-Attention
├─────────────────┤
│ Add & Norm      │
├─────────────────┤
│ Cross Attention │ ← Encoder-Decoder Attention
├─────────────────┤
│ Add & Norm      │
├─────────────────┤
│ Feed Forward    │
├─────────────────┤
│ Add & Norm      │
└─────────────────┘
```

### 3.2 Position-wise Feed-Forward

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))
```

### 3.3 Residual Connection & Layer Norm

$$\text{LayerNorm}(x + \text{Sublayer}(x))$$

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x
```

---

## 4. Positional Encoding

### 4.1 왜 필요한가?

```
Attention은 순서를 모름:
- "나는 밥을 먹는다" vs "밥을 나는 먹는다"
- Attention만으로는 구분 불가

위치 정보 추가 필요!
```

### 4.2 Sinusoidal Encoding

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

```python
def positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         -(math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe
```

### 4.3 왜 Sinusoidal인가?

```
장점:
1. 상대 위치 학습 가능
   - PE(pos+k)는 PE(pos)의 선형 변환
2. 임의의 길이로 확장 가능
3. 학습 불필요
```

---

## 5. 세 가지 Attention

### 5.1 Self-Attention (Encoder)

```
Q = K = V = 입력

각 토큰이 다른 모든 토큰 참조
"The cat sat on the mat"
→ 모든 단어가 서로 연결
```

### 5.2 Masked Self-Attention (Decoder)

```
미래 토큰 마스킹:
생성 시 아직 없는 토큰 참조 불가

"The cat [MASK] [MASK] [MASK]"
→ "cat"은 "The"만 볼 수 있음
```

### 5.3 Cross-Attention (Encoder-Decoder)

```
Q = Decoder
K = V = Encoder

Decoder가 Encoder 출력 참조
번역: 타겟이 소스를 참조
```

---

## 6. 실험 결과

### 6.1 기계 번역

| 모델 | En→De BLEU | En→Fr BLEU |
|------|------------|------------|
| 기존 SOTA | 26.0 | 41.0 |
| **Transformer (base)** | **27.3** | **38.1** |
| **Transformer (big)** | **28.4** | **41.8** |

### 6.2 학습 속도

```
Transformer vs RNN:
- 학습 시간: 3.5일 vs 수주
- 병렬화: 완전 vs 제한적
- 8 GPU로 달성
```

### 6.3 모델 설정

| 항목 | Base | Big |
|------|------|-----|
| d_model | 512 | 1024 |
| d_ff | 2048 | 4096 |
| heads | 8 | 16 |
| layers | 6 | 6 |
| params | 65M | 213M |

---

## 7. 이후 발전

### 7.1 Decoder-Only (GPT)

```
Encoder 제거, Decoder만 사용
→ 언어 모델링에 적합
→ GPT-1, 2, 3, 4
```

### 7.2 Encoder-Only (BERT)

```
Decoder 제거, Encoder만 사용
→ 이해 태스크에 적합
→ BERT, RoBERTa
```

### 7.3 현대 LLM

```
LLaMA, GPT-4, Claude 등:
- Decoder-only 기반
- RoPE (Sinusoidal 대체)
- GQA (MHA 효율화)
- SwiGLU (ReLU 대체)
```

---

## 8. 쉬운 예시

### 8.1 회의 비유

```
RNN = 전화 릴레이
- A → B → C → D 순서대로 전달
- 마지막에 정보 손실
- 직접 소통 불가

Transformer = 화상 회의
- 모든 참석자가 서로 직접 대화
- 정보 손실 없음
- 병렬 소통 가능
```

### 8.2 Multi-Head 비유

```
하나의 문장을 여러 관점으로:
"나는 어제 서울에서 친구를 만났다"

Head 1 (문법): 나는 → 만났다 (주어-동사)
Head 2 (시간): 어제 → 만났다 (시간-동작)
Head 3 (장소): 서울에서 → 만났다 (장소-동작)
Head 4 (대상): 친구를 → 만났다 (목적어-동사)
```

---

## 9. 핵심 요약

### 기억해야 할 것들

1. **핵심**: Attention만으로 시퀀스 모델링
2. **장점**: 병렬화, 장거리 의존성
3. **구성**: Self-Attn + FFN + Residual
4. **의의**: 현대 AI의 기반 아키텍처

### 주요 수식

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 복잡도 비교

| 모델 | 시퀀스당 복잡도 | 병렬화 |
|------|-----------------|--------|
| RNN | O(n) | O(n) |
| CNN | O(n/k) | O(1) |
| **Transformer** | **O(1)** | **O(1)** |

### 영향

- BERT, GPT, T5, LLaMA, Claude, Gemini...
- 모든 현대 LLM의 기반
- NLP → Vision → Audio → Multimodal

---

## 참고 자료

1. [논문](https://arxiv.org/abs/1706.03762)
2. [Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
3. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

*다음 리뷰: [GPT-3](./002_GPT-3.md)*
