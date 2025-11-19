# Code Llama: Open Foundation Models for Code

**논문 발표**: 2023년
**저자**: Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, et al.
**소속**: Meta AI
**논문 링크**: [arXiv:2308.12950](https://arxiv.org/abs/2308.12950)
**공식 모델**: [HuggingFace](https://huggingface.co/codellama)

---

## 한 줄 요약
> LLaMA 2를 코드 데이터로 추가 학습하고, infilling과 긴 컨텍스트를 지원하는 코드 특화 모델 패밀리로, HumanEval에서 GPT-4에 근접한 성능 달성

---

## 1. Code Llama 개요

### 1.1 모델 패밀리

```
Code Llama Family:
├── Code Llama (기본)
│   └── 7B, 13B, 34B, 70B
├── Code Llama - Python
│   └── Python 특화
└── Code Llama - Instruct
    └── Instruction-tuned
```

### 1.2 학습 파이프라인

```
LLaMA 2
    ↓ 500B 코드 토큰으로 DAPT
Code Llama
    ↓ 100B Python 토큰
Code Llama - Python
    ↓ Instruction Fine-tuning
Code Llama - Instruct
```

---

## 2. 학습 데이터

### 2.1 코드 코퍼스

| 언어 | 비율 |
|------|------|
| Python | 15% |
| C++ | 12% |
| Java | 10% |
| JavaScript | 9% |
| 기타 | 54% |

### 2.2 데이터 양

- **기본 DAPT**: 500B 토큰
- **Python 특화**: +100B Python 토큰
- **총 학습량**: LLaMA 2 (2T) + Code (500B+)

---

## 3. 핵심 기술 1: Infilling

### 3.1 개념

중간 부분을 채우는 능력:

```python
# 입력 (prefix와 suffix만 제공)
def factorial(n):
    <FILL>
    return result

# 출력 (middle 생성)
    if n <= 1:
        return 1
    result = n * factorial(n - 1)
```

### 3.2 학습 방법

```python
# 원본 코드
original = "def add(a, b):\n    return a + b"

# Infilling 형식으로 변환
prefix = "def add(a, b):\n"
middle = "    return a + b"
suffix = ""

# 학습 데이터
training_input = f"<PRE>{prefix}<SUF>{suffix}<MID>"
training_target = middle
```

### 3.3 Infilling 비율

```
학습 데이터 중:
- 90%: 일반 생성 (left-to-right)
- 10%: Infilling 형식

→ 두 능력 모두 유지
```

---

## 4. 핵심 기술 2: Long Context

### 4.1 컨텍스트 확장

```
LLaMA 2: 4,096 토큰
Code Llama: 16,384 토큰 (4배)

추가 학습으로:
→ 100,000 토큰까지 가능
```

### 4.2 RoPE 주파수 조정

$$\theta_i = 10000^{-2i/d} \rightarrow \theta_i' = \theta_i \cdot \beta$$

```python
# RoPE base frequency 수정
rope_scaling = {
    "type": "linear",
    "factor": 4.0  # 4x 확장
}

# 또는
rope_theta = 1000000  # 기본 10000에서 증가
```

### 4.3 Long Context Fine-tuning

```
16K 컨텍스트 학습:
- 20B 토큰 추가 학습
- 긴 코드 파일, 레포지토리 수준 데이터
```

---

## 5. Instruction Tuning

### 5.1 데이터 구성

```python
instruction_data = [
    {
        "instruction": "Write a function to sort a list",
        "input": "[3, 1, 4, 1, 5]",
        "output": "def sort_list(lst):\n    return sorted(lst)"
    },
    # ...
]
```

### 5.2 특별 학습

- Self-instruct 방식으로 데이터 생성
- Unit test 생성 및 실행
- 코드 설명 생성

---

## 6. 실험 결과

### 6.1 HumanEval (Python)

| 모델 | Pass@1 | Pass@100 |
|------|--------|----------|
| LLaMA 2 34B | 22.6% | 77.2% |
| Code Llama 34B | 48.8% | 85.4% |
| GPT-3.5 Turbo | 48.1% | - |
| **GPT-4** | **67.0%** | - |
| Code Llama 70B | 53.0% | 87.0% |

### 6.2 다국어 코드 (MultiPL-E)

| 언어 | Code Llama 34B |
|------|----------------|
| Python | 48.8% |
| JavaScript | 48.1% |
| C++ | 45.6% |
| Java | 41.8% |
| Rust | 37.8% |

### 6.3 Infilling 성능

| 모델 | Single Line | Multi Line |
|------|-------------|------------|
| InCoder 6.7B | 28.4% | 14.2% |
| Code Llama 7B | 39.5% | 25.1% |
| Code Llama 13B | 44.3% | 32.8% |

---

## 7. 모델 선택 가이드

### 7.1 용도별 추천

| 용도 | 추천 모델 |
|------|-----------|
| 일반 코드 생성 | Code Llama |
| Python 집중 | Code Llama - Python |
| 대화형 코딩 | Code Llama - Instruct |
| 최고 성능 | 70B 모델 |
| 빠른 추론 | 7B 모델 |

### 7.2 크기별 특성

```
7B: 빠름, 단순 작업
13B: 균형 잡힌 선택
34B: 복잡한 로직
70B: 최고 품질
```

---

## 8. 사용법

### 8.1 기본 코드 생성

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")

prompt = "def fibonacci(n):"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

### 8.2 Infilling

```python
# Infilling 형식
prompt = "<PRE>def add(a, b):\n<SUF>\n    return result<MID>"

# 생성
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)

# 결과: "    result = a + b"
```

### 8.3 Instruction 모델

```python
prompt = """[INST] Write a Python function that calculates the factorial of a number recursively. [/INST]"""

# Code Llama Instruct 사용
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
```

---

## 9. 쉬운 예시

### 9.1 언어 학습 비유

```
영어 기초 (LLaMA 2)
→ 프로그래밍 언어 집중 학습 (Code Llama)
→ Python 전문 과정 (Code Llama - Python)
→ 코딩 튜터 훈련 (Code Llama - Instruct)

점점 전문화!
```

### 9.2 Infilling 비유

```
일반 글쓰기: 처음부터 끝까지 순서대로

Infilling:
"오늘 아침 ___을/를 먹고 학교에 갔다"
→ "오늘 아침 [빵]을/를 먹고 학교에 갔다"

코드에서:
def func():
    <빈칸>
    return x
→ x = compute_value()
```

---

## 10. 핵심 요약

### 기억해야 할 것들

1. **기반**: LLaMA 2 + 500B 코드 토큰
2. **특징**: Infilling + Long Context (16K+)
3. **변형**: 기본, Python, Instruct
4. **성능**: GPT-4의 70-80% (오픈소스 최고)

### 주요 수치

| 항목 | 값 |
|------|-----|
| 코드 학습량 | 500B 토큰 |
| 컨텍스트 | 16,384 토큰 |
| HumanEval (70B) | 53% |
| Infilling 비율 | 10% |

### 실무 팁

- **IDE 통합**: Infilling으로 자동완성
- **코드 리뷰**: Instruct로 설명 생성
- **긴 파일**: Long context 활용
- **Python**: Python 특화 모델 사용

---

## 참고 자료

1. [Code Llama 논문](https://arxiv.org/abs/2308.12950)
2. [GitHub](https://github.com/facebookresearch/codellama)
3. [HuggingFace Models](https://huggingface.co/codellama)
4. [Meta AI Blog](https://ai.meta.com/blog/code-llama-large-language-model-coding/)

---

*이전 리뷰: [ChipNeMo](./003_ChipNeMo.md)*
*다음 섹션: [Data Engineering & Synthetic Data](../2_Data_Engineering/)*
