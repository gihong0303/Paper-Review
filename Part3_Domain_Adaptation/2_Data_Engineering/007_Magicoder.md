# Magicoder: Empowering Code Generation with OSS-Instruct

**논문 발표**: 2024년 (ICML 2024)
**저자**: Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, Lingming Zhang
**소속**: University of Illinois Urbana-Champaign
**논문 링크**: [arXiv:2312.02120](https://arxiv.org/abs/2312.02120)
**공식 구현**: [GitHub](https://github.com/ise-uiuc/magicoder)

---

## 한 줄 요약
> 오픈소스 코드 스니펫을 시드로 활용하여 LLM이 다양하고 현실적인 코드 instruction 데이터를 생성하게 하는 OSS-Instruct로, 7B 모델이 ChatGPT를 능가하는 코드 생성 성능 달성

---

## 1. 문제 정의

### 1.1 기존 코드 instruction 데이터의 한계

합성 데이터 생성의 두 가지 주요 방법:

| 방법 | 설명 | 한계 |
|------|------|------|
| **Self-Instruct** | LLM이 스스로 데이터 생성 | 편향된 분포, 다양성 부족 |
| **Evol-Instruct** | 기존 데이터를 진화 | 원본 데이터에 의존 |

### 1.2 Self-Instruct/Evol-Instruct의 문제

```
Self-Instruct의 편향:
┌─────────────────────────────────────┐
│ "코딩 문제를 만들어줘"                │
│                                     │
│ LLM의 응답:                          │
│ - "피보나치 수열 구현하기"            │
│ - "팩토리얼 계산하기"                │
│ - "버블 정렬 구현하기"               │
│                                     │
│ 문제: 항상 비슷한 "교과서적" 예제만!   │
│      실제 코딩과 다름                │
└─────────────────────────────────────┘

Evol-Instruct의 한계:
원본: "리스트 정렬하기"
    ↓ 진화
변형: "리스트를 내림차순으로 정렬하기"
    ↓ 진화
변형: "중복 제거 후 내림차순 정렬하기"

→ 여전히 원본의 연장선, 새로운 주제 없음
```

### 1.3 실제 코드와의 격차

```
교과서적 코드:               실제 프로젝트 코드:
┌─────────────────┐          ┌─────────────────┐
│ def fibonacci(n):│          │ @retry(max=3)   │
│   if n <= 1:     │          │ def fetch_api(): │
│     return n     │          │   with Session() │
│   return fib(n-1)│          │     as sess:    │
│     + fib(n-2)   │          │     resp = sess │
│                  │          │       .get(url, │
└─────────────────┘          │       timeout=30│
단순한 알고리즘              │       headers=h)│
                             └─────────────────┘
                             API, 라이브러리, 예외처리

합성 데이터가 후자를 다루지 못함!
```

---

## 2. 핵심 아이디어: OSS-Instruct

### 2.1 개념

**Open-Source Software Instruct**: 오픈소스 코드를 시드로 활용

```
OSS-Instruct 프로세스:
┌─────────────────────────────────────────┐
│ 1. 오픈소스 코드에서 스니펫 추출         │
│    예: GitHub의 실제 코드               │
│                                         │
│ 2. 스니펫을 시드로 instruction 생성     │
│    LLM에게: "이 코드에서 영감을 받아     │
│             코딩 문제를 만들어줘"       │
│                                         │
│ 3. 생성된 instruction으로 모델 학습     │
└─────────────────────────────────────────┘

장점:
- 실제 코드의 다양성 반영
- 라이브러리, API, 패턴 포함
- LLM의 편향 완화
```

### 2.2 Self-Instruct와의 비교

```
Self-Instruct:
┌─────────┐     ┌─────────┐
│ LLM     │ → → │ Problem │
│ (biased)│     │ (biased)│
└─────────┘     └─────────┘

OSS-Instruct:
┌─────────┐     ┌─────────┐     ┌─────────┐
│ OSS Code│ → → │ LLM     │ → → │ Problem │
│ (diverse)│    │         │     │ (diverse)│
└─────────┘     └─────────┘     └─────────┘
               시드가 다양성 제공
```

### 2.3 왜 효과적인가?

1. **다양성**: 오픈소스의 방대한 코드 패턴
2. **현실성**: 실제 사용되는 코드 기반
3. **제어 가능**: 시드 선택으로 분포 조절
4. **확장성**: 무한한 오픈소스 코드 활용

---

## 3. OSS-Instruct 상세

### 3.1 프롬프트 설계

```python
OSS_INSTRUCT_PROMPT = """
Below is a code snippet from an open-source project:

```
{code_snippet}
```

Inspired by this code snippet, create a self-contained coding problem
that is:
1. Clear and unambiguous
2. Solvable with a single function or short program
3. Diverse in terms of concepts (avoid basic algorithms only)

Provide:
- Problem description
- Example input/output
- Solution code

Format your response as:
[Problem Description]
...
[Example]
Input: ...
Output: ...
[Solution]
```python
...
```
"""
```

### 3.2 시드 코드 선택

```python
def select_code_snippets(repo_path: str, max_snippets: int = 1000):
    """오픈소스 레포에서 코드 스니펫 선택"""
    snippets = []

    for file_path in find_python_files(repo_path):
        code = read_file(file_path)

        # 품질 필터링
        if not is_quality_code(code):
            continue

        # 함수/클래스 추출
        for node in extract_functions_and_classes(code):
            snippet = get_source_code(node)

            # 길이 필터링 (너무 짧거나 긴 것 제외)
            if 50 < len(snippet) < 500:
                snippets.append({
                    'code': snippet,
                    'file': file_path,
                    'type': type(node).__name__,
                })

    # 다양성을 위한 샘플링
    return diverse_sample(snippets, max_snippets)


def is_quality_code(code: str) -> bool:
    """코드 품질 체크"""
    # 주석 비율
    comment_ratio = count_comments(code) / len(code)
    if comment_ratio > 0.5:  # 주석이 너무 많으면 제외
        return False

    # 복잡도 체크
    if cyclomatic_complexity(code) > 20:  # 너무 복잡하면 제외
        return False

    # docstring 존재
    if not has_docstring(code):
        return False

    return True
```

### 3.3 데이터 생성 파이프라인

```python
from openai import OpenAI

def generate_oss_instruct_data(
    code_snippets: list[str],
    num_samples: int = 75000
) -> list[dict]:
    """OSS-Instruct 데이터 생성"""
    client = OpenAI()
    dataset = []

    for snippet in code_snippets:
        prompt = OSS_INSTRUCT_PROMPT.format(code_snippet=snippet)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048,
        )

        result = response.choices[0].message.content

        # 파싱
        try:
            problem, example, solution = parse_response(result)
            dataset.append({
                'instruction': problem,
                'input': example,
                'output': solution,
                'seed_code': snippet,
            })
        except:
            continue  # 파싱 실패 시 스킵

        if len(dataset) >= num_samples:
            break

    return dataset
```

---

## 4. Magicoder 모델

### 4.1 모델 구성

| 모델 | 베이스 | 학습 데이터 | 특징 |
|------|--------|-------------|------|
| **Magicoder-CL** | CodeLlama-7B | OSS-Instruct 75K | 기본 |
| **Magicoder-DS** | DeepSeek-Coder-6.7B | OSS-Instruct 75K | 강력한 베이스 |
| **MagicoderS-CL** | CodeLlama-7B | + Evol-Instruct 110K | 추가 학습 |
| **MagicoderS-DS** | DeepSeek-Coder-6.7B | + Evol-Instruct 110K | 최고 성능 |

### 4.2 학습 설정

```python
# 학습 하이퍼파라미터
training_config = {
    "model": "codellama/CodeLlama-7b-Python-hf",
    "dataset": "ise-uiuc/Magicoder-OSS-Instruct-75K",

    # 학습 설정
    "num_epochs": 2,
    "learning_rate": 2e-5,
    "batch_size": 128,  # effective batch size
    "max_length": 2048,
    "warmup_ratio": 0.05,

    # 최적화
    "optimizer": "adamw",
    "lr_scheduler": "cosine",
    "weight_decay": 0.0,

    # 효율화
    "bf16": True,
    "gradient_checkpointing": True,
}
```

### 4.3 Evol-Instruct와의 결합

OSS-Instruct와 Evol-Instruct는 **직교적**:

```
OSS-Instruct: 시드 → 새로운 문제 생성
Evol-Instruct: 기존 문제 → 복잡한 변형

결합:
1단계: OSS-Instruct 75K로 Magicoder 학습
2단계: Evol-Instruct 110K로 MagicoderS 추가 학습

두 방법이 서로 보완!
```

---

## 5. 구현

### 5.1 전체 파이프라인

```python
import json
from pathlib import Path
from tqdm import tqdm

class MagicoderDataGenerator:
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def collect_seed_snippets(
        self,
        repos_dir: str,
        max_per_repo: int = 100
    ) -> list[dict]:
        """GitHub 레포에서 시드 코드 수집"""
        all_snippets = []

        for repo_path in Path(repos_dir).iterdir():
            if not repo_path.is_dir():
                continue

            snippets = self.extract_snippets(repo_path, max_per_repo)
            all_snippets.extend(snippets)

        print(f"Collected {len(all_snippets)} seed snippets")
        return all_snippets

    def extract_snippets(self, repo_path: Path, max_count: int) -> list[dict]:
        """단일 레포에서 스니펫 추출"""
        import ast

        snippets = []

        for py_file in repo_path.rglob("*.py"):
            try:
                code = py_file.read_text(encoding='utf-8')
                tree = ast.parse(code)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        snippet = ast.get_source_segment(code, node)
                        if snippet and 50 < len(snippet) < 500:
                            snippets.append({
                                'code': snippet,
                                'repo': repo_path.name,
                                'file': str(py_file.relative_to(repo_path)),
                            })

                        if len(snippets) >= max_count:
                            return snippets
            except:
                continue

        return snippets

    def generate_instruction(self, snippet: dict) -> dict | None:
        """시드 코드에서 instruction 생성"""
        prompt = f"""Below is a code snippet from an open-source project:

```python
{snippet['code']}
```

Inspired by this code, create a self-contained coding problem.

Requirements:
1. The problem should be clear and unambiguous
2. It should be solvable with a single function
3. Include diverse programming concepts

Provide your response in this exact format:

[Problem Description]
<describe the problem here>

[Example]
Input: <example input>
Output: <example output>

[Solution]
```python
<solution code>
```"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "system", "content": "You are an expert programmer creating coding challenges."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2048,
            )

            content = response.choices[0].message.content
            return self.parse_response(content, snippet)

        except Exception as e:
            print(f"Error generating instruction: {e}")
            return None

    def parse_response(self, content: str, snippet: dict) -> dict:
        """응답 파싱"""
        # Problem Description 추출
        desc_match = content.split("[Problem Description]")[1].split("[Example]")[0].strip()

        # Example 추출
        example_match = content.split("[Example]")[1].split("[Solution]")[0].strip()

        # Solution 추출
        solution_match = content.split("```python")[1].split("```")[0].strip()

        return {
            "instruction": desc_match,
            "input": example_match,
            "output": solution_match,
            "seed_code": snippet['code'],
            "source_repo": snippet['repo'],
        }

    def generate_dataset(
        self,
        snippets: list[dict],
        output_path: str,
        num_samples: int = 75000
    ):
        """전체 데이터셋 생성"""
        dataset = []

        for snippet in tqdm(snippets, desc="Generating instructions"):
            if len(dataset) >= num_samples:
                break

            result = self.generate_instruction(snippet)
            if result:
                dataset.append(result)

        # 저장
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        print(f"Generated {len(dataset)} instruction samples")
        return dataset


# 사용 예시
generator = MagicoderDataGenerator(api_key="your-api-key")

# 1. 시드 코드 수집
snippets = generator.collect_seed_snippets(
    repos_dir="./github_repos",
    max_per_repo=100
)

# 2. instruction 데이터 생성
dataset = generator.generate_dataset(
    snippets,
    output_path="./magicoder_data.json",
    num_samples=75000
)
```

### 5.2 학습 스크립트

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

def train_magicoder():
    # 모델 로드
    model_name = "codellama/CodeLlama-7b-Python-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 데이터 로드
    dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K")

    def format_prompt(sample):
        """Alpaca 형식으로 포맷팅"""
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""
        return {"text": prompt}

    formatted_dataset = dataset.map(format_prompt)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            truncation=True,
            max_length=2048,
            padding="max_length",
        )

    tokenized_dataset = formatted_dataset.map(
        tokenize,
        remove_columns=formatted_dataset["train"].column_names,
    )

    # 학습 설정
    training_args = TrainingArguments(
        output_dir="./magicoder-cl-7b",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=32,
        learning_rate=2e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        gradient_checkpointing=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
    )

    # 학습
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    train_magicoder()
```

### 5.3 추론 사용

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_code(prompt: str, model_name: str = "ise-uiuc/Magicoder-S-CL-7B"):
    """Magicoder로 코드 생성"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 프롬프트 포맷팅
    formatted_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.0,  # 결정론적 생성
        do_sample=False,
    )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return response


# 사용
prompt = """Write a Python function that finds the longest common subsequence of two strings."""

code = generate_code(prompt)
print(code)
```

---

## 6. 쉬운 예시로 이해하기

### 6.1 요리 레시피 비유

**Self-Instruct** (자체 생성):
```
"레시피를 만들어줘"

LLM 응답:
- 계란 프라이
- 라면 끓이기
- 토스트 만들기

→ 항상 기본적인 것만!
```

**OSS-Instruct** (실제 요리에서 영감):
```
실제 레스토랑 메뉴에서 시드:
"트러플 리조또와 버섯 크림 소스"

LLM: "이 요리에서 영감받아 레시피를 만들어줘"

응답:
"버섯과 크림을 사용한 파스타 만들기"
- 실제 재료 사용법 포함
- 적절한 난이도
- 다양한 기법

→ 더 현실적이고 다양한 레시피!
```

### 6.2 수학 문제 비유

**기존 방법**:
```
"수학 문제 만들어줘"
→ "2 + 2 = ?"
→ "10의 제곱은?"
→ "삼각형의 넓이는?"

항상 교과서적 문제
```

**OSS-Instruct**:
```
실제 수학 논문의 공식에서 시드:
"푸아송 분포의 확률 질량 함수"

LLM이 생성:
"어떤 콜센터에 시간당 평균 5통의 전화가 온다.
 10분 동안 정확히 2통의 전화가 올 확률은?"

→ 실제 응용과 연결된 문제
```

### 6.3 실제 예시

**시드 코드** (GitHub에서):
```python
@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print(f"{name}: {end-start:.2f}s")
```

**생성된 instruction**:
```
[Problem]
Create a context manager that tracks memory usage
before and after a code block, and prints the difference.

[Solution]
import tracemalloc

@contextmanager
def memory_tracker(name):
    tracemalloc.start()
    yield
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{name}: Current={current/1024:.2f}KB, Peak={peak/1024:.2f}KB")
```

시드의 패턴(context manager, 측정)을 새로운 문제로 변환!

---

## 7. 실험 결과

### 7.1 HumanEval+ 성능

| 모델 | 크기 | pass@1 |
|------|------|--------|
| ChatGPT (GPT-3.5) | - | 65.9% |
| GPT-4 | - | 86.6% |
| CodeLlama-Python | 7B | 38.4% |
| WizardCoder | 7B | 50.6% |
| **Magicoder-CL** | **7B** | **60.4%** |
| **MagicoderS-CL** | **7B** | **66.5%** |

**MagicoderS-CL-7B가 ChatGPT를 능가!** (66.5% vs 65.9%)

### 7.2 다양한 벤치마크

| 벤치마크 | WizardCoder-7B | Magicoder-CL-7B | MagicoderS-CL-7B |
|----------|----------------|-----------------|------------------|
| HumanEval | 54.9% | 64.0% | **68.3%** |
| HumanEval+ | 50.6% | 60.4% | **66.5%** |
| MBPP | 57.2% | 61.0% | **64.2%** |
| MBPP+ | 47.5% | 52.4% | **55.6%** |
| MultiPL-E (Python) | 54.9% | 64.0% | **68.3%** |

### 7.3 DeepSeek-Coder 기반 성능

| 모델 | HumanEval+ | MBPP+ |
|------|------------|-------|
| DeepSeek-Coder-6.7B | 66.5% | 60.2% |
| Magicoder-DS | 68.3% | 62.1% |
| **MagicoderS-DS** | **71.3%** | **65.4%** |

### 7.4 데이터 품질 분석

```
생성된 instruction의 토픽 분포:

Self-Instruct:
알고리즘 ████████████████████ 45%
문자열  ██████████           25%
수학    ██████████           25%
기타    ██                   5%

OSS-Instruct:
알고리즘 ████████             20%
API/라이브러리 ████████████  30%
데이터처리 ██████████        25%
시스템/네트워크 ██████        15%
기타 ████                    10%

→ OSS-Instruct가 훨씬 다양한 토픽!
```

### 7.5 Ablation Study

| 설정 | HumanEval+ |
|------|------------|
| OSS-Instruct (75K) | 60.4% |
| Evol-Instruct (110K) | 58.5% |
| **OSS + Evol (185K)** | **66.5%** |
| OSS only 150K | 62.1% |

두 방법의 결합이 최고 성능!

---

## 8. 한계점 및 후속 연구

### 8.1 현재 한계점

1. **교사 모델 의존**: GPT-3.5에 의존
   - 더 강력한 교사 모델 사용 시 개선 가능

2. **Python 중심**: 다른 언어 지원 제한
   - 다국어 OSS-Instruct 연구 필요

3. **품질 필터링**: 생성된 데이터의 품질 검증 부족
   - 자동 품질 평가 필요

4. **코드 실행 검증**: 생성된 솔루션의 정확성 미검증
   - 실행 기반 필터링 추가 가능

### 8.2 후속 연구

1. **자체 개선 (Self-Improvement)**:
   - 생성된 데이터로 더 강한 모델 학습
   - 그 모델로 더 좋은 데이터 생성

2. **다국어 확장**:
   - JavaScript, Java, C++ 등
   - 언어별 특화 시드 선택

3. **도메인 특화**:
   - 웹 개발, 데이터 과학, 시스템 프로그래밍
   - 도메인별 레포 선택

4. **실행 기반 필터링**:
   - 생성된 코드 실행하여 검증
   - 올바른 솔루션만 유지

### 8.3 관련 연구

| 방법 | 특징 | Magicoder와의 차이 |
|------|------|-------------------|
| WizardCoder | Evol-Instruct | 시드가 기존 데이터 |
| CodeAlpaca | Self-Instruct | LLM 편향 |
| StarCoder | 대규모 사전학습 | Instruction 튜닝 없음 |

---

## 9. 핵심 요약

### 기억해야 할 것들

1. **OSS-Instruct**: 오픈소스 코드를 시드로 instruction 생성
2. **다양성**: Self-Instruct보다 훨씬 다양한 토픽/패턴
3. **직교성**: Evol-Instruct와 결합하여 추가 개선
4. **성능**: 7B 모델이 ChatGPT 능가

### 핵심 데이터

| 데이터셋 | 크기 | 용도 |
|----------|------|------|
| OSS-Instruct | 75K | Magicoder 학습 |
| Evol-Instruct | 110K | MagicoderS 추가 학습 |
| 총합 | 185K | 최고 성능 |

### 실무 체크리스트

```python
# 1. 데이터 수집
# GitHub에서 고품질 레포 선택
# - 별 수가 많은 레포
# - 잘 문서화된 코드
# - 다양한 도메인

# 2. 시드 추출
# - 50-500 문자의 함수/클래스
# - 주석과 docstring이 있는 코드

# 3. Instruction 생성
# - GPT-3.5 또는 GPT-4 사용
# - 명확한 프롬프트 설계

# 4. 모델 학습
from datasets import load_dataset
dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K")
# + Evol-Instruct 110K for MagicoderS

# 5. 평가
# - HumanEval, MBPP 등 코드 벤치마크
```

---

## 참고 자료

1. [Magicoder 논문](https://arxiv.org/abs/2312.02120)
2. [공식 GitHub 저장소](https://github.com/ise-uiuc/magicoder)
3. [HuggingFace 모델](https://huggingface.co/ise-uiuc)
4. [OSS-Instruct 데이터셋](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K)
5. [WizardCoder 논문](https://arxiv.org/abs/2306.08568)

---

*이전 리뷰: [DCLM](./006_DCLM.md)*
*다음 리뷰: [Phi-1](./001_Phi-1_Textbooks.md)*
