# No Language Left Behind: Scaling Human-Centered Machine Translation

**논문 발표**: 2022년
**저자**: NLLB Team (Meta AI)
**소속**: Meta AI
**논문 링크**: [arXiv:2207.04672](https://arxiv.org/abs/2207.04672)
**공식 모델**: [HuggingFace](https://huggingface.co/facebook/nllb-200-distilled-600M)

---

## 한 줄 요약
> 200개 언어를 지원하는 단일 번역 모델로, 저자원 언어 번역 품질을 40% 이상 향상시킨 대규모 다국어 기계 번역 시스템

---

## 1. 문제: 언어 불평등

### 1.1 현실

```
전 세계 ~7000개 언어
기존 번역 시스템:
- 고자원 언어 100개만 지원
- 저자원 언어는 품질 매우 낮음
- 수십억 명이 혜택 못 받음
```

### 1.2 NLLB의 목표

```
"어떤 언어도 뒤처지지 않게"

1. 200개 언어 고품질 번역
2. 특히 저자원 언어 집중
3. 인간 중심 평가
```

---

## 2. 데이터 수집

### 2.1 전략

```
저자원 언어 데이터 확보:
1. 웹 크롤링 (LASER 필터링)
2. 병렬 코퍼스 마이닝
3. Back-translation
4. 커뮤니티 협력
```

### 2.2 LASER 3

```python
# 200개 언어 임베딩
# 문장 유사도로 병렬 문장 찾기

def mine_parallel_sentences(source_lang, target_lang):
    source_embeddings = laser.encode(source_sentences)
    target_embeddings = laser.encode(target_sentences)

    # 코사인 유사도로 매칭
    similarities = cosine_similarity(source_embeddings, target_embeddings)

    # 높은 유사도 쌍 선택
    pairs = []
    for i, j in find_best_matches(similarities):
        if similarities[i, j] > 1.04:  # 임계값
            pairs.append((source_sentences[i], target_sentences[j]))

    return pairs
```

### 2.3 데이터 규모

| 자원 수준 | 언어 수 | 평균 문장 쌍 |
|-----------|---------|--------------|
| 고자원 | 30 | 100M+ |
| 중자원 | 70 | 1M-100M |
| 저자원 | 100 | 10K-1M |

---

## 3. 모델 아키텍처

### 3.1 Sparsely Gated MoE

```
Mixture of Experts (MoE):
- 총 파라미터: 54.5B
- 활성 파라미터: 3.3B (추론 시)
- 효율적인 스케일링

각 토큰이 전문가 2개만 활성화
```

### 3.2 모델 크기

| 모델 | 파라미터 | 활성 파라미터 |
|------|----------|---------------|
| Dense-600M | 600M | 600M |
| Dense-1.3B | 1.3B | 1.3B |
| Dense-3.3B | 3.3B | 3.3B |
| **MoE-54.5B** | **54.5B** | **3.3B** |

### 3.3 언어 토큰

```python
# 소스/타겟 언어 지정
source_text = "Hello, how are you?"
target_lang = "kor_Hang"  # 한국어

# 입력 형식
input_text = f"<{target_lang}> {source_text}"
# 출력: "안녕하세요, 어떻게 지내세요?"
```

---

## 4. 학습 전략

### 4.1 Temperature Sampling

$$p_l \propto n_l^{1/T}$$

- $T = 5$: 저자원 언어 비율 증가

### 4.2 다단계 학습

```
Stage 1: 고자원 언어로 기본 학습
Stage 2: 저자원 언어 집중 학습
Stage 3: Back-translation으로 보강
```

### 4.3 Curriculum Learning

```
쉬운 것 → 어려운 것:
1. 단문 (5-10 토큰)
2. 중문 (10-30 토큰)
3. 장문 (30+ 토큰)
```

---

## 5. 평가

### 5.1 FLORES-200 벤치마크

```
200개 언어 평가 데이터셋:
- 문장 2009개 × 200개 언어
- 전문가 번역
- 다양한 도메인
```

### 5.2 결과 (BLEU)

| 방향 | NLLB-54B | M2M-100 | 개선 |
|------|----------|---------|------|
| eng→xxx (평균) | 24.4 | 18.5 | +32% |
| xxx→eng (평균) | 28.2 | 22.1 | +28% |
| 저자원 | 18.7 | 12.3 | **+52%** |

### 5.3 인간 평가

```
품질 점수 (1-5):
- NLLB: 3.8
- 기존 최고: 3.2

저자원 언어에서 특히 큰 차이
```

---

## 6. 사용법

### 6.1 기본 번역

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")

    translated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )

    return tokenizer.decode(translated[0], skip_special_tokens=True)

# 영어 → 한국어
result = translate("Hello, how are you?", "eng_Latn", "kor_Hang")
# "안녕하세요, 어떻게 지내세요?"
```

### 6.2 언어 코드

```python
# 형식: [언어]_[스크립트]
language_codes = {
    "한국어": "kor_Hang",
    "영어": "eng_Latn",
    "일본어": "jpn_Jpan",
    "중국어 간체": "zho_Hans",
    "아랍어": "arb_Arab",
    "스와힐리어": "swh_Latn"
}
```

### 6.3 다양한 언어 쌍

```python
# 비영어 ↔ 비영어도 가능
# 한국어 → 일본어
translate("안녕하세요", "kor_Hang", "jpn_Jpan")
# "こんにちは"

# 프랑스어 → 아랍어
translate("Bonjour", "fra_Latn", "arb_Arab")
```

---

## 7. 한국어 성능

### 7.1 FLORES-200 결과

| 방향 | BLEU |
|------|------|
| eng→kor | 15.2 |
| kor→eng | 18.7 |

### 7.2 개선 방향

```
한국어는 중자원 언어:
- 성능 준수하지만 개선 여지 있음
- 한국어 특화 모델이 더 좋을 수 있음
- 도메인 적응으로 향상 가능
```

---

## 8. 한계와 윤리

### 8.1 기술적 한계

```
1. 고자원 언어: 전용 모델이 더 좋음
2. 긴 문서: 문장 단위 번역
3. 도메인 특화: 일반 도메인 학습
4. 문화적 뉘앙스: 아직 어려움
```

### 8.2 윤리적 고려

```
1. 유해 콘텐츠 번역 위험
2. 번역 품질 불균형
3. 언어 다양성 보존
4. 커뮤니티 참여 필요
```

---

## 9. 쉬운 예시

### 9.1 UN 통역 시스템 비유

```
기존 번역:
- 영어 통역사 10명
- 영어↔주요언어만 가능

NLLB:
- 만능 통역 시스템
- 200개 언어 모두 직통
- 한국어↔스와힐리어도 가능
```

### 9.2 언어 교량 비유

```
기존: 영어가 중심 허브
A → 영어 → B

NLLB: 직접 연결
A ↔ B (영어 거치지 않음)

더 정확하고 빠른 번역
```

---

## 10. 핵심 요약

### 기억해야 할 것들

1. **규모**: 200개 언어 지원
2. **집중**: 저자원 언어 품질 향상
3. **방법**: MoE + Data Mining
4. **결과**: 저자원 52% 향상

### 주요 수치

| 항목 | 값 |
|------|-----|
| 지원 언어 | 200개 |
| 최대 파라미터 | 54.5B (MoE) |
| 저자원 향상 | +52% |
| 데이터셋 | FLORES-200 |

### 모델 선택 가이드

| 용도 | 추천 모델 |
|------|-----------|
| 빠른 추론 | nllb-200-distilled-600M |
| 균형 | nllb-200-1.3B |
| 최고 품질 | nllb-200-3.3B |
| 연구용 | nllb-moe-54b |

### 실무 팁

- 언어 코드 형식: `[언어]_[스크립트]`
- 저자원 언어도 직접 번역 가능
- 도메인 특화 시 fine-tuning 권장
- Distilled 모델이 실용적

---

## 참고 자료

1. [NLLB 논문](https://arxiv.org/abs/2207.04672)
2. [HuggingFace](https://huggingface.co/facebook/nllb-200-distilled-600M)
3. [FLORES-200](https://github.com/facebookresearch/flores)
4. [Meta AI Blog](https://ai.facebook.com/blog/nllb-200-high-quality-machine-translation/)

---

*이전 리뷰: [BLOOM](./004_BLOOM.md)*
*다음 리뷰: [Swallow](./006_Swallow.md)*
