# Swallow: Continual Pre-training for Building Japanese Language Models

**논문 발표**: 2024년
**저자**: Toyo University NLP Group, National Institute of Informatics
**소속**: Toyo University, NII
**논문 링크**: [arXiv:2404.17790](https://arxiv.org/abs/2404.17790)
**공식 모델**: [HuggingFace](https://huggingface.co/tokyotech-llm/Swallow-70b-hf)

---

## 한 줄 요약
> LLaMA-2를 일본어 데이터로 CPT하고 vocabulary를 확장하여, 일본어 성능을 대폭 향상시키면서 영어 능력도 유지한 일본어 특화 LLM

---

## 1. 배경: 일본어 LLM의 필요성

### 1.1 LLaMA-2의 일본어 한계

```
LLaMA-2 학습 데이터:
- 영어: 89.7%
- 일본어: 0.1% 미만

결과:
- 일본어 이해 제한적
- 토큰 효율 낮음
- 일본어 생성 품질 낮음
```

### 1.2 해결 전략

```
Swallow의 접근법:
1. Vocabulary 확장 (일본어 토큰 추가)
2. 일본어 데이터로 CPT
3. 영어 능력 유지

→ 한국어 LLM 개발에도 적용 가능!
```

---

## 2. Vocabulary 확장

### 2.1 왜 필요한가?

```
LLaMA-2 토크나이저로 일본어:
"自然言語処理" → ["自", "然", "言", "語", "処", "理"]
6개 토큰!

확장 후:
"自然言語処理" → ["自然", "言語", "処理"]
3개 토큰으로 감소!
```

### 2.2 확장 방법

```python
# 1. 일본어 코퍼스로 BPE 학습
japanese_tokenizer = train_bpe(
    japanese_corpus,
    vocab_size=10000
)

# 2. 기존 vocabulary와 병합
original_vocab = llama_tokenizer.get_vocab()
new_tokens = [t for t in japanese_tokenizer.get_vocab()
              if t not in original_vocab]

# 3. 임베딩 초기화
for token in new_tokens:
    # 서브워드 평균으로 초기화
    subword_ids = original_tokenizer(token)
    new_embedding = mean([embeddings[id] for id in subword_ids])
```

### 2.3 Swallow의 vocabulary

| 항목 | LLaMA-2 | Swallow |
|------|---------|---------|
| Vocab 크기 | 32,000 | 43,176 |
| 추가 토큰 | - | 11,176 |
| 일본어 효율 | 1x | 1.8x |

---

## 3. Continual Pre-training

### 3.1 학습 데이터

| 언어 | 토큰 수 | 비율 |
|------|---------|------|
| 일본어 | 100B | 80% |
| 영어 | 25B | 20% |

### 3.2 학습 설정

```python
training_config = {
    "base_model": "LLaMA-2-70B",
    "learning_rate": 1e-4,
    "warmup_steps": 2000,
    "total_steps": 50000,
    "batch_size": 256,
    "sequence_length": 4096
}
```

### 3.3 영어 유지를 위한 혼합

```python
# 일본어:영어 = 8:2 혼합
def sample_batch():
    if random.random() < 0.8:
        return japanese_data.sample()
    else:
        return english_data.sample()
```

---

## 4. 임베딩 초기화 방법

### 4.1 서브워드 평균

```python
def initialize_embedding(new_token, tokenizer, embeddings):
    """새 토큰의 임베딩을 서브워드 평균으로 초기화"""
    # 기존 토크나이저로 분해
    subword_ids = tokenizer.encode(new_token)

    # 서브워드 임베딩 평균
    subword_embeds = [embeddings[id] for id in subword_ids]
    new_embed = torch.mean(torch.stack(subword_embeds), dim=0)

    return new_embed
```

### 4.2 대안: Random 초기화

```python
# 랜덤 초기화 (성능 낮음)
new_embed = torch.randn(hidden_size) * 0.02
```

### 4.3 비교

| 초기화 방법 | 초기 Loss | 최종 성능 |
|-------------|-----------|-----------|
| Random | 높음 | 낮음 |
| 서브워드 평균 | 낮음 | 높음 |

---

## 5. 실험 결과

### 5.1 일본어 벤치마크

| 모델 | JCommonsenseQA | JNLI | JSQuAD |
|------|----------------|------|--------|
| LLaMA-2-70B | 45.2 | 48.3 | 52.1 |
| Japanese-LLaMA | 62.8 | 58.7 | 68.4 |
| **Swallow-70B** | **78.5** | **72.1** | **85.3** |

### 5.2 영어 벤치마크

| 모델 | MMLU | HellaSwag | ARC |
|------|------|-----------|-----|
| LLaMA-2-70B | 68.9 | 85.3 | 67.3 |
| **Swallow-70B** | **65.2** | **82.1** | **64.8** |

영어 성능 약간 감소 (3-5%)하지만 대부분 유지!

### 5.3 토큰 효율

| 텍스트 | LLaMA-2 | Swallow | 감소율 |
|--------|---------|---------|--------|
| 일본어 뉴스 | 1.0x | 1.8x | 44% |
| 일본어 위키 | 1.0x | 1.7x | 41% |

---

## 6. 한국어 적용 가이드

### 6.1 Vocabulary 확장

```python
# 한국어 토큰 추가
korean_tokens = [
    "안녕하세요",
    "감사합니다",
    "대한민국",
    "서울",
    ...
]

# 또는 한국어 코퍼스로 BPE 학습
korean_bpe = train_bpe(korean_corpus, vocab_size=10000)
```

### 6.2 학습 데이터

```python
# 한국어:영어 비율
# 영어 능력 유지를 위해 영어도 포함
data_mix = {
    "korean": 0.8,  # 80%
    "english": 0.2  # 20%
}

korean_sources = [
    "나무위키",
    "한국어 위키피디아",
    "뉴스 기사",
    "정부 문서",
    ...
]
```

### 6.3 체크리스트

- [ ] 한국어 토큰 10K+ 추가
- [ ] 한국어 코퍼스 100B+ 토큰
- [ ] 영어 데이터 20% 혼합
- [ ] 임베딩 서브워드 평균 초기화
- [ ] 낮은 학습률 (1e-4 ~ 1e-5)

---

## 7. 모델 크기별 결과

### 7.1 Swallow 패밀리

| 모델 | 파라미터 | JCom | MMLU |
|------|----------|------|------|
| Swallow-7B | 7B | 65.2 | 48.3 |
| Swallow-13B | 13B | 71.8 | 55.1 |
| Swallow-70B | 70B | 78.5 | 65.2 |

### 7.2 성능 대비 효율

```
Swallow-13B:
- 일본어: LLaMA-2-70B 능가
- 파라미터: 5배 적음
- 추론 속도: 5배 빠름

도메인 적응의 힘!
```

---

## 8. 사용법

### 8.1 기본 사용

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "tokyotech-llm/Swallow-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 일본어 생성
prompt = "日本の首都は"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

### 8.2 Instruct 모델

```python
# Instruction-tuned 버전
model_name = "tokyotech-llm/Swallow-7b-instruct-hf"

prompt = """以下の質問に日本語で答えてください。

質問: 東京タワーの高さは何メートルですか？

回答:"""
```

---

## 9. 쉬운 예시

### 9.1 언어 학습 비유

```
LLaMA-2 = 영어 원어민
- 영어 완벽
- 일본어는 "니호고 조즈데스"수준

Swallow = 일본어 집중 학습한 LLaMA
- 영어 거의 유지
- 일본어도 자연스럽게

추가 학습으로 새 언어 습득!
```

### 9.2 도구 비유

```
LLaMA 토크나이저 = 영어 자판
- 영어: 효율적
- 일본어: 한 글자씩 입력

Swallow 토크나이저 = 일본어 IME 추가
- 일본어도 빠르게 입력
- "わたし" → 3타 → 1토큰
```

---

## 10. 핵심 요약

### 기억해야 할 것들

1. **핵심**: Vocab 확장 + CPT
2. **결과**: 일본어 78%, 영어 95% 유지
3. **효율**: 토큰 44% 감소
4. **적용**: 한국어 LLM에 동일 방법 가능

### 주요 수치

| 항목 | 값 |
|------|-----|
| 추가 토큰 | 11,176 |
| 일본어 데이터 | 100B |
| 일본어 향상 | +33% |
| 영어 유지 | 95% |

### 한국어 적용 공식

$$\text{Korean LLM} = \text{Base LLM} + \text{Korean Vocab} + \text{Korean CPT}$$

### 실무 팁

- 토큰 10K+ 추가 권장
- 임베딩은 서브워드 평균으로 초기화
- 영어 20% 혼합으로 영어 유지
- 학습률 낮게 (1e-4 이하)
- 100B+ 토큰으로 CPT

---

## 참고 자료

1. [Swallow 논문](https://arxiv.org/abs/2404.17790)
2. [HuggingFace](https://huggingface.co/tokyotech-llm)
3. [GitHub](https://github.com/tokyotech-llm/swallow)

---

*이전 리뷰: [No Language Left Behind](./005_NLLB.md)*
*Part 3 완료!*
