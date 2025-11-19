# QLoRA: Efficient Finetuning of Quantized LLMs

**논문 발표**: 2023년 (NeurIPS 2023)
**저자**: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
**소속**: University of Washington
**논문 링크**: [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
**공식 구현**: [GitHub](https://github.com/artidoro/qlora)

---

## 한 줄 요약
> 4-bit 양자화된 모델 위에 LoRA를 적용하고, 새로운 NF4 데이터 타입과 Double Quantization을 도입하여, 단일 48GB GPU에서 65B 모델을 16-bit 성능으로 파인튜닝

---

## 1. 핵심 혁신

### 1.1 세 가지 기술

1. **NF4 (4-bit NormalFloat)**: 정규분포에 최적화된 양자화
2. **Double Quantization**: 양자화 상수도 양자화
3. **Paged Optimizers**: GPU 메모리 스파이크 방지

### 1.2 메모리 절약

```
LLaMA-65B 파인튜닝:
16-bit LoRA: ~780GB (불가능)
QLoRA:       ~48GB  (단일 A100!)
```

---

## 2. NF4 (4-bit NormalFloat)

### 2.1 아이디어

신경망 가중치는 **정규분포**를 따름 → 정규분포에 최적화된 양자화

```python
# NF4 양자화 레벨 (정규분포 분위수)
nf4_levels = [
    -1.0, -0.6962, -0.5251, -0.3949,
    -0.2844, -0.1848, -0.0911, 0.0,
    0.0796, 0.1609, 0.2461, 0.3379,
    0.4407, 0.5626, 0.7230, 1.0
]
```

### 2.2 vs INT4

```
INT4: 균일 간격       NF4: 정규분포 분위수
|--|--|--|--|--|       |--|-|--|--|---|
0  4  8  12 16        균일하지 않지만
                      정규분포에 최적
```

---

## 3. Double Quantization

양자화 상수(scale)도 양자화:

```
Single Quantization:
가중치: 4bit, Scale: FP32 (32bit)
→ 추가 메모리: 32/64 = 0.5bit/param

Double Quantization:
Scale을 다시 8bit로 양자화
→ 추가 메모리: 8/64 + 32/(64×256) ≈ 0.127bit/param

절약: ~0.37bit/param
```

---

## 4. Paged Optimizers

### 4.1 문제

긴 시퀀스에서 gradient checkpointing → 메모리 스파이크

### 4.2 해결

NVIDIA Unified Memory로 CPU ↔ GPU 자동 페이징

---

## 5. 실험 결과

### 5.1 Guanaco 모델

QLoRA로 학습한 Guanaco가 ChatGPT의 99.3% 성능 달성:

| 모델 | Elo Rating |
|------|------------|
| GPT-4 | 1348 |
| ChatGPT | 1176 |
| **Guanaco-65B** | **1168** |
| Guanaco-33B | 1116 |

### 5.2 메모리 효율

| 모델 | Full FT | LoRA 16bit | **QLoRA** |
|------|---------|------------|-----------|
| 7B | 160GB | 56GB | **12GB** |
| 13B | 320GB | 112GB | **20GB** |
| 65B | 1500GB | 560GB | **48GB** |

---

## 6. 사용법

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4
    bnb_4bit_use_double_quant=True,      # Double Quant
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA 적용
lora_config = LoraConfig(r=64, lora_alpha=16, ...)
model = get_peft_model(model, lora_config)
```

---

## 7. 핵심 요약

### 기억해야 할 것들

1. **NF4**: 정규분포 최적화 4bit 양자화
2. **Double Quant**: Scale도 양자화
3. **결과**: 65B를 단일 48GB GPU에서 학습
4. **성능**: 16bit와 동등

### 실무 팁

- 4bit 모델 위에 16bit LoRA 학습
- Compute dtype은 bfloat16 권장
- Gradient checkpointing 필수

---

## 참고 자료

1. [QLoRA 논문](https://arxiv.org/abs/2305.14314)
2. [GitHub](https://github.com/artidoro/qlora)

---

*이전 리뷰: [LoRA](./001_LoRA.md)*
*다음 리뷰: [DoRA](./003_DoRA.md)*
