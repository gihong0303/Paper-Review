# Cosmopedia: Synthetic Data for LLM Pre-training

**논문 발표**: 2024년
**저자**: Loubna Ben Allal, Anton Lozhkov, Guilherme Penedo, Thomas Wolf, Leandro von Werra
**소속**: Hugging Face
**논문 링크**: [arXiv:2403.xxxxx](https://huggingface.co/blog/cosmopedia) (Technical Report)
**데이터셋**: [HuggingFace](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)

---

## 한 줄 요약
> Mixtral-8x7B를 사용해 25B 토큰의 합성 교육 콘텐츠를 생성하여, Phi-1.5 스타일의 "교과서" 데이터를 오픈소스로 공개

---

## 1. 배경: 합성 데이터의 필요성

### 1.1 Phi 시리즈의 영감

```
Phi-1/1.5의 성공:
- "교과서 품질" 합성 데이터
- 적은 양으로 높은 성능
- 하지만 데이터 비공개

Cosmopedia의 목표:
→ 오픈소스 "교과서" 데이터 만들기!
```

### 1.2 합성 데이터의 장점

| 실제 데이터 | 합성 데이터 |
|-------------|-------------|
| 품질 불균일 | 균일한 품질 |
| 저작권 문제 | 저작권 자유 |
| 양 제한적 | 무한 생성 가능 |
| 필터링 필요 | 처음부터 깨끗 |

---

## 2. 데이터 생성 파이프라인

### 2.1 전체 구조

```
시드 데이터 (Wikipedia, 웹 등)
           ↓
    주제/컨텍스트 추출
           ↓
    프롬프트 템플릿 적용
           ↓
    Mixtral-8x7B 생성
           ↓
    품질 필터링
           ↓
   Cosmopedia (25B tokens)
```

### 2.2 콘텐츠 유형

| 유형 | 비율 | 설명 |
|------|------|------|
| Textbooks | 35% | 교과서 스타일 설명 |
| Blog Posts | 25% | 블로그 형식 |
| Stories | 20% | 교육적 이야기 |
| WikiHow | 10% | 단계별 가이드 |
| Forum Q&A | 10% | 질문-답변 |

---

## 3. 프롬프트 엔지니어링

### 3.1 교과서 스타일 프롬프트

```python
textbook_prompt = """
Write a chapter of a textbook about the following topic:
Topic: {topic}

Requirements:
- Write at an undergraduate level
- Include clear explanations
- Provide 2-3 illustrative examples
- Build from simple to complex concepts
- Include a summary at the end

Chapter:
"""
```

### 3.2 이야기 스타일 프롬프트

```python
story_prompt = """
Write an educational story for {age_group} about:
Topic: {topic}

The story should:
- Be engaging and age-appropriate
- Naturally incorporate the educational content
- Have clear characters and plot
- End with a lesson or insight

Story:
"""

# 다양한 타겟 청중
age_groups = ["children", "teenagers", "adults"]
```

### 3.3 Q&A 스타일 프롬프트

```python
qa_prompt = """
Imagine you're answering a question on an educational forum.

Question: {question}

Provide a detailed, helpful answer that:
- Directly addresses the question
- Explains underlying concepts
- Gives practical examples
- Suggests further learning resources

Answer:
"""
```

---

## 4. 시드 데이터 소스

### 4.1 주제 추출

```python
seed_sources = {
    "Wikipedia": "다양한 주제의 개요",
    "OpenWebMath": "수학 문제와 개념",
    "AutoMathText": "수학 교육 콘텐츠",
    "KhanAcademy": "교육 과정 주제",
    "Stanford Encyclopedia": "철학/과학 개념"
}

def extract_topics(source):
    """시드 소스에서 주제 추출"""
    topics = []
    for doc in source:
        # 제목, 키워드 추출
        topic = extract_main_topic(doc)
        context = extract_context(doc)
        topics.append({
            "topic": topic,
            "context": context,
            "source": source.name
        })
    return topics
```

### 4.2 컨텍스트 보강

```python
def enrich_context(topic):
    """주제에 대한 컨텍스트 추가"""
    return {
        "topic": topic,
        "related_concepts": find_related(topic),
        "difficulty": estimate_difficulty(topic),
        "prerequisites": find_prerequisites(topic)
    }
```

---

## 5. 품질 관리

### 5.1 생성 시 품질 제어

```python
generation_config = {
    "temperature": 0.8,  # 다양성
    "top_p": 0.95,
    "max_tokens": 2000,
    "repetition_penalty": 1.1,  # 반복 방지
}
```

### 5.2 후처리 필터링

```python
def quality_filter(generated_text):
    # 1. 길이 체크
    if len(generated_text) < 500:
        return False

    # 2. 반복 체크
    if has_excessive_repetition(generated_text):
        return False

    # 3. 완결성 체크
    if not is_complete(generated_text):
        return False

    # 4. 코드 실행 체크 (코드 포함 시)
    if contains_code(generated_text):
        if not code_executes(generated_text):
            return False

    return True
```

---

## 6. 실험 결과

### 6.1 Cosmopedia로 학습한 모델

```
실험 설정:
- 모델: 1.8B 파라미터
- 데이터: Cosmopedia 25B 토큰
- 에폭: 1
```

### 6.2 성능 비교

| 데이터셋 | MMLU | ARC | HellaSwag |
|----------|------|-----|-----------|
| FineWeb (실제) | 28.4 | 38.2 | 58.3 |
| Cosmopedia (합성) | 26.8 | 36.1 | 55.7 |
| **혼합 (50:50)** | **30.2** | **40.5** | **61.2** |

### 6.3 핵심 발견

```
1. 합성 데이터만으로도 경쟁력 있음
2. 실제+합성 혼합이 최고
3. 특히 추론 태스크에서 합성 데이터 효과적
```

---

## 7. 다양한 도메인 확장

### 7.1 Cosmopedia-v2 계획

```
추가 도메인:
- 코드 교육 콘텐츠
- 과학 실험 설명
- 역사적 사건 분석
- 수학 문제 풀이 과정
```

### 7.2 다국어 확장

```python
# 한국어 합성 데이터 예시
korean_prompt = """
다음 주제에 대한 교과서 스타일의 설명을 작성하세요:
주제: {topic}

요구사항:
- 대학교 수준으로 작성
- 명확한 설명 포함
- 2-3개의 예시 제공
- 단순한 것에서 복잡한 것으로 진행
- 마지막에 요약 포함

내용:
"""
```

---

## 8. 사용법

### 8.1 데이터셋 로드

```python
from datasets import load_dataset

# 전체 데이터셋
dataset = load_dataset(
    "HuggingFaceTB/cosmopedia",
    split="train"
)

# 샘플 확인
print(dataset[0]["text"][:1000])
print(f"Type: {dataset[0]['format']}")
```

### 8.2 특정 유형만 필터링

```python
# 교과서만
textbooks = dataset.filter(
    lambda x: x["format"] == "textbook"
)

# 수학 관련만
math_content = dataset.filter(
    lambda x: "math" in x["topic"].lower()
)
```

### 8.3 자체 합성 데이터 생성

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_textbook_content(topic):
    prompt = f"Write a textbook chapter about {topic}..."
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=2000,
        temperature=0.8
    )
    return tokenizer.decode(outputs[0])
```

---

## 9. 쉬운 예시

### 9.1 요리책 비유

```
실제 데이터 = 인터넷 레시피
- 품질 제각각
- 설명 부족한 것도 많음
- 같은 요리 중복 많음

합성 데이터 = 전문 요리책
- 전문가가 체계적으로 작성
- 모든 과정 상세 설명
- 난이도별 정리
```

### 9.2 교육 비유

```
웹 데이터로 학습 = 인터넷 서핑으로 공부
- 정보가 산발적
- 품질 검증 안됨
- 비효율적

Cosmopedia로 학습 = 교과서로 공부
- 체계적 구성
- 전문가 검토됨
- 효율적
```

---

## 10. 핵심 요약

### 기억해야 할 것들

1. **목표**: 오픈소스 "교과서" 데이터
2. **규모**: 25B 토큰
3. **방법**: Mixtral로 합성 생성
4. **결과**: 실제+합성 혼합이 최고

### 주요 수치

| 항목 | 값 |
|------|-----|
| 총 토큰 | 25B |
| 생성 모델 | Mixtral-8x7B |
| 콘텐츠 유형 | 5가지 |
| 시드 소스 | 5가지 |

### 데이터 혼합 공식

$$\text{Best Data} = \alpha \cdot \text{Real} + (1-\alpha) \cdot \text{Synthetic}$$

여기서 $\alpha \approx 0.5$가 최적

### 실무 팁

- 다양한 프롬프트 템플릿 사용
- 시드 데이터로 주제 다양성 확보
- 품질 필터링 필수
- 실제 데이터와 혼합 사용

---

## 참고 자료

1. [Cosmopedia Dataset](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)
2. [Hugging Face Blog](https://huggingface.co/blog/cosmopedia)
3. [SmolLM Blog](https://huggingface.co/blog/smollm)

---

*이전 리뷰: [FineWeb](./003_FineWeb.md)*
*다음 리뷰: [Self-Instruct](./005_Self-Instruct.md)*
