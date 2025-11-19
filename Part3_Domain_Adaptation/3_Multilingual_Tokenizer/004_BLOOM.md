# BLOOM: A 176B-Parameter Open-Access Multilingual Language Model

**논문 발표**: 2022년
**저자**: BigScience Workshop (1000+ 연구자)
**소속**: BigScience (Hugging Face 주도)
**논문 링크**: [arXiv:2211.05100](https://arxiv.org/abs/2211.05100)
**공식 모델**: [HuggingFace](https://huggingface.co/bigscience/bloom)

---

## 한 줄 요약
> 전 세계 1000명 이상의 연구자가 협력하여 만든 176B 파라미터 오픈소스 다국어 모델로, 46개 자연어와 13개 프로그래밍 언어 지원

---

## 1. BigScience 프로젝트

### 1.1 배경

```
2021년 상황:
- GPT-3: 비공개, API만 제공
- 대형 모델 연구 접근성 제한
- 다국어 지원 부족

BigScience 목표:
- 완전 오픈소스 대형 모델
- 투명한 연구 과정
- 다양한 언어 지원
```

### 1.2 협력 규모

```
참여자:
- 1000+ 연구자
- 60+ 국가
- 250+ 기관

기간: 1년+
컴퓨팅: Jean Zay 슈퍼컴퓨터
```

---

## 2. 모델 구성

### 2.1 모델 크기

| 모델 | 파라미터 | 레이어 | Hidden | Heads |
|------|----------|--------|--------|-------|
| BLOOM-560M | 560M | 24 | 1024 | 16 |
| BLOOM-1.1B | 1.1B | 24 | 1536 | 16 |
| BLOOM-1.7B | 1.7B | 24 | 2048 | 16 |
| BLOOM-3B | 3B | 30 | 2560 | 32 |
| BLOOM-7.1B | 7.1B | 30 | 4096 | 32 |
| **BLOOM-176B** | **176B** | **70** | **14336** | **112** |

### 2.2 아키텍처 특징

```python
# ALiBi (Attention with Linear Biases)
# Position embedding 대신 사용
# 더 긴 컨텍스트 일반화

# Embedding LayerNorm
# 학습 안정성 향상
```

---

## 3. 학습 데이터: ROOTS

### 3.1 데이터 구성

| 분류 | 크기 | 비율 |
|------|------|------|
| 자연어 (46개) | 350B | 70% |
| 프로그래밍 언어 (13개) | 150B | 30% |
| **총계** | **500B** | 100% |

### 3.2 언어 분포 (상위)

| 언어 | 비율 |
|------|------|
| 영어 | 30.0% |
| 중국어 | 16.2% |
| 프랑스어 | 12.9% |
| 스페인어 | 10.8% |
| ... | ... |
| **한국어** | **0.3%** |

### 3.3 데이터 거버넌스

```
BigScience Data Governance:
1. 데이터 출처 투명성
2. 개인정보 필터링
3. 저작권 고려
4. 편향 분석
5. 문서화 (Data Card)
```

---

## 4. 학습 과정

### 4.1 학습 설정

```python
training_config = {
    "total_tokens": 366B,
    "batch_size": 2048,
    "sequence_length": 2048,
    "learning_rate": 1e-4,
    "optimizer": "Adam",
    "precision": "bfloat16",
    "hardware": "384 A100 80GB GPUs"
}
```

### 4.2 학습 기간

```
총 학습 시간: 3.5개월
- 117일 학습
- 일부 재시작 포함
```

### 4.3 분산 학습

```
Megatron-DeepSpeed 사용:
- Tensor Parallelism: 4-way
- Pipeline Parallelism: 12-way
- Data Parallelism: 8-way

총 384 GPUs
```

---

## 5. 토크나이저

### 5.1 특징

```python
# BPE 기반 SentencePiece
vocab_size = 250680

# 언어별 균형 있는 vocabulary
# 비라틴 문자 지원 강화
```

### 5.2 토큰 효율 비교

| 언어 | BLOOM | GPT-2 |
|------|-------|-------|
| 영어 | 1.2 | 1.1 |
| 한국어 | 1.8 | 3.5 |
| 아랍어 | 1.5 | 2.8 |

BLOOM이 비영어 효율 더 좋음

---

## 6. 실험 결과

### 6.1 SuperGLUE (영어)

| 모델 | 평균 |
|------|------|
| GPT-3 175B | 71.8 |
| BLOOM-176B | 65.4 |

영어에서는 GPT-3가 우위

### 6.2 다국어 태스크

XNLI (Zero-shot):

| 언어 | BLOOM-176B | mGPT |
|------|------------|------|
| 영어 | 68.5 | 52.1 |
| 프랑스어 | 65.2 | 48.3 |
| 중국어 | 60.1 | 45.8 |
| 아랍어 | 58.7 | 42.1 |

### 6.3 코드 생성

HumanEval:

| 모델 | Pass@1 |
|------|--------|
| GPT-3 175B | 20.1% |
| BLOOM-176B | 15.5% |
| Codex-12B | 28.8% |

---

## 7. 사용법

### 7.1 기본 사용

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "bigscience/bloom-7b1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 생성
prompt = "La capitale de la France est"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

### 7.2 다국어 사용

```python
# 여러 언어로 동일 질문
prompts = {
    "en": "The capital of France is",
    "fr": "La capitale de la France est",
    "ko": "프랑스의 수도는",
    "zh": "法国的首都是"
}

for lang, prompt in prompts.items():
    output = generate(prompt)
    print(f"{lang}: {output}")
```

### 7.3 BLOOMZ (Instruction-tuned)

```python
# Instruction-tuned 버전
model_name = "bigscience/bloomz-7b1"

prompt = "Translate to French: Hello, how are you?"
# 출력: "Bonjour, comment allez-vous?"
```

---

## 8. 한계와 개선

### 8.1 한계

```
1. 영어 성능: GPT-3 대비 낮음
2. 코드 생성: 전문 모델 대비 약함
3. 한국어: 데이터 비율 낮음 (0.3%)
4. 추론 속도: 176B는 매우 느림
```

### 8.2 후속 모델

```
BLOOMZ: Instruction-tuned BLOOM
- 다국어 instruction following
- Zero-shot 성능 크게 향상

mT0: T0 스타일 학습
```

---

## 9. 쉬운 예시

### 9.1 UN 통역사 비유

```
GPT-3 = 영어 전문 통역사
- 영어 최고
- 다른 언어는 보조적

BLOOM = UN 다국어 통역사
- 46개 언어 가능
- 각각은 전문가만큼은 아님
- 하지만 모든 회의 커버 가능
```

### 9.2 오픈소스 비유

```
GPT-3 = 맛집의 비밀 레시피
- 맛있지만 어떻게 만드는지 모름

BLOOM = 공개된 요리 교실
- 모든 재료와 과정 공개
- 누구나 배우고 개선 가능
```

---

## 10. 핵심 요약

### 기억해야 할 것들

1. **규모**: 176B 파라미터, 46개 언어
2. **오픈소스**: 완전 공개 (모델, 데이터, 코드)
3. **협력**: 1000+ 연구자 참여
4. **의의**: 투명한 대형 모델 연구의 시작

### 주요 수치

| 항목 | 값 |
|------|-----|
| 최대 파라미터 | 176B |
| 자연어 | 46개 |
| 프로그래밍 언어 | 13개 |
| 학습 토큰 | 366B |
| Vocab | 250K |

### BLOOM vs GPT-3

| 특성 | BLOOM | GPT-3 |
|------|-------|-------|
| 공개 | 완전 오픈 | API만 |
| 언어 | 46개 | 영어 중심 |
| 영어 성능 | 낮음 | 높음 |
| 다국어 | 좋음 | 제한적 |

### 실무 팁

- 다국어 태스크에 적합
- 영어 전용이면 다른 모델 권장
- BLOOMZ가 instruction following 더 좋음
- 메모리 많이 필요 (176B는 300GB+)

---

## 참고 자료

1. [BLOOM 논문](https://arxiv.org/abs/2211.05100)
2. [HuggingFace](https://huggingface.co/bigscience/bloom)
3. [BigScience](https://bigscience.huggingface.co/)
4. [ROOTS Dataset](https://huggingface.co/datasets/bigscience/roots)

---

*이전 리뷰: [XGLM](./003_XGLM.md)*
*다음 리뷰: [No Language Left Behind](./005_NLLB.md)*
