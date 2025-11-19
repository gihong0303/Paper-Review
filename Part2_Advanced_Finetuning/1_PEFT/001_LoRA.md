# LoRA: Low-Rank Adaptation of Large Language Models

**논문 발표**: 2021년 (ICLR 2022)
**저자**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
**소속**: Microsoft
**논문 링크**: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
**공식 구현**: [GitHub](https://github.com/microsoft/LoRA)

---

## 한 줄 요약
> 사전 학습된 가중치를 고정하고 저차원(low-rank) 행렬 쌍만 학습하여, 파라미터 수를 10,000배 줄이면서 Full Fine-tuning과 동등한 성능 달성

---

## 1. 문제 정의

### 1.1 Full Fine-tuning의 문제

```
GPT-3 175B 파인튜닝:
- 학습 파라미터: 175B
- Optimizer state: 350B (Adam)
- 총 메모리: ~1TB+
- 태스크별 모델 저장: 350GB × N개

→ 비실용적!
```

### 1.2 기존 PEFT 방법들의 한계

**Adapter**: 추론 지연시간 증가
**Prefix-tuning**: 최적화 어려움, 시퀀스 길이 감소

---

## 2. 핵심 아이디어: Low-Rank Decomposition

### 2.1 가설

> "Fine-tuning 시 가중치 변화는 **저차원(low-rank)**이다"

수학적으로:
$$W' = W + \Delta W$$

$\Delta W$가 low-rank라면:
$$\Delta W = BA$$

여기서 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$

### 2.2 파라미터 절약

```
원래: d × k 파라미터
LoRA: d × r + r × k = r(d + k) 파라미터

예: d = k = 4096, r = 8
원래: 16,777,216
LoRA: 65,536
절약: 99.6%!
```

### 2.3 시각화

```
원본 가중치:              LoRA:
┌─────────────┐          ┌───┐   ┌─────────────┐
│             │          │   │   │             │
│    W        │    →     │ B │ × │      A      │
│  (d × k)    │          │   │   │   (r × k)   │
│   고정      │          │(d×r)│   └─────────────┘
│             │          │학습│
└─────────────┘          └───┘
```

---

## 3. 수학적 상세

### 3.1 Forward Pass

$$h = W_0 x + \Delta W x = W_0 x + BA x$$

```python
def lora_forward(x, W0, A, B, scaling):
    # 원본 경로 (고정)
    h = x @ W0

    # LoRA 경로 (학습)
    h += (x @ A @ B) * scaling

    return h
```

### 3.2 Scaling Factor

$$h = W_0 x + \frac{\alpha}{r} BA x$$

- $\alpha$: LoRA의 학습률 조절
- $r$: rank
- 일반적으로 $\alpha = r$ 또는 $\alpha = 2r$

### 3.3 초기화

- **A**: Gaussian 초기화
- **B**: Zero 초기화 → 학습 시작 시 $\Delta W = 0$

---

## 4. 적용 대상

### 4.1 Transformer에서 어디에 적용?

```
Self-Attention:
- Wq (Query)     ← LoRA 적용
- Wk (Key)       ← LoRA 적용 (선택)
- Wv (Value)     ← LoRA 적용
- Wo (Output)    ← LoRA 적용

FFN:
- W_up          ← LoRA 적용 (선택)
- W_down        ← LoRA 적용 (선택)
```

### 4.2 권장 설정

| 설정 | Rank | Alpha | 적용 레이어 |
|------|------|-------|------------|
| 효율적 | 8 | 16 | Q, V |
| 균형 | 16 | 32 | Q, K, V, O |
| 고성능 | 64 | 128 | 모든 Linear |

---

## 5. 쉬운 예시

### 5.1 조각 맞추기 비유

**Full Fine-tuning**: 전체 그림을 다시 그림
**LoRA**: 작은 스티커를 붙여서 수정

원본 그림(W)은 그대로 두고, 작은 스티커(BA)만 추가

### 5.2 옷 수선 비유

**Full Fine-tuning**: 새 옷을 맞춤 제작
**LoRA**: 기존 옷에 패치만 추가

패치가 작아도(low-rank) 원하는 스타일 표현 가능

---

## 6. 구현

### 6.1 PyTorch 구현

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # 원본 가중치 (고정)
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.W.weight.requires_grad = False

        # LoRA 파라미터
        self.A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_features))

        self.scaling = alpha / rank

    def forward(self, x):
        # W는 고정, A와 B만 학습
        return self.W(x) + (x @ self.A @ self.B) * self.scaling
```

### 6.2 HuggingFace PEFT 사용

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 모델 로드
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# LoRA 설정
lora_config = LoraConfig(
    r=16,                      # rank
    lora_alpha=32,             # scaling
    target_modules=["q_proj", "v_proj"],  # 적용 대상
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRA 적용
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 출력: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%
```

---

## 7. 실험 결과

### 7.1 GPT-3 175B 결과

| 방법 | 파라미터 | WikiSQL | MNLI | SAMSum |
|------|----------|---------|------|--------|
| Fine-tune | 175B | 73.8 | 89.5 | 52.0 |
| Adapter | 7.1M | 73.2 | 89.3 | 51.5 |
| **LoRA** | **4.7M** | **73.4** | **91.7** | **53.8** |

### 7.2 학습 효율

| 메트릭 | Full FT | LoRA | 개선 |
|--------|---------|------|------|
| 학습 파라미터 | 100% | 0.01% | 10000× |
| GPU 메모리 | 100% | 30% | 3× |
| 저장 공간 | 350GB | 35MB | 10000× |

### 7.3 추론 속도

```
Full Fine-tuning: 기준
Adapter:          0.95× (5% 느림)
LoRA:             1.0× (동일!)

LoRA는 W' = W + BA로 merge 가능 → 추가 지연 없음
```

---

## 8. LoRA의 장점

### 8.1 메모리 효율

```
LLaMA-7B 학습 메모리:
Full FT:  ~120GB (불가능)
LoRA r=8: ~16GB  (RTX 4090 가능!)
```

### 8.2 태스크 전환

```python
# 여러 LoRA 어댑터
model.load_adapter("adapter_task_a")  # 태스크 A
model.load_adapter("adapter_task_b")  # 태스크 B

# 빠른 전환 (35MB만 교체)
```

### 8.3 병합 가능

```python
# 추론 시 merge
W_merged = W + (alpha/r) * B @ A

# 추가 연산 없음!
```

---

## 9. 한계점

### 9.1 Rank 선택

- 너무 낮음: 성능 저하
- 너무 높음: 효율성 감소
- 최적 rank는 태스크마다 다름

### 9.2 모든 태스크에 적합하지 않음

- 도메인이 크게 다른 경우 성능 저하
- 매우 복잡한 태스크는 Full FT가 나을 수 있음

### 9.3 학습 불안정성

- 일부 경우 학습이 불안정
- DoRA, rsLoRA 등이 이를 개선

---

## 10. 핵심 요약

### 기억해야 할 것들

1. **핵심 가설**: Fine-tuning 변화는 low-rank
2. **방법**: $\Delta W = BA$ 로 분해
3. **장점**: 10000× 파라미터 절약, 동등 성능
4. **적용**: Q, V projection이 기본

### 핵심 수식

$$h = W_0 x + \frac{\alpha}{r} BA x$$

### 실무 체크리스트

- [ ] Rank 선택: 8-64 (태스크에 따라)
- [ ] Alpha: 보통 rank의 2배
- [ ] 적용 레이어: 최소 Q, V
- [ ] 초기화: A는 Gaussian, B는 Zero

---

## 참고 자료

1. [LoRA 논문](https://arxiv.org/abs/2106.09685)
2. [Microsoft LoRA GitHub](https://github.com/microsoft/LoRA)
3. [HuggingFace PEFT](https://github.com/huggingface/peft)
4. [LoRA 심층 분석](https://lightning.ai/pages/community/lora-insights/)

---

*다음 리뷰: [QLoRA](./002_QLoRA.md)*
