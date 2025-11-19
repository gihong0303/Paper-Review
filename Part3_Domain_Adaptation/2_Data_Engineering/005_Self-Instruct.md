# Self-Instruct: Aligning Language Models with Self-Generated Instructions

**논문 발표**: 2022년 (ACL 2023)
**저자**: Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi
**소속**: University of Washington, Allen Institute for AI
**논문 링크**: [arXiv:2212.10560](https://arxiv.org/abs/2212.10560)
**공식 구현**: [GitHub](https://github.com/yizhongw/self-instruct)

---

## 한 줄 요약
> 소수의 시드 태스크에서 시작하여 LLM이 스스로 instruction 데이터를 생성하는 부트스트래핑 방법으로, GPT-3로 52K instruction을 생성해 사람이 작성한 것의 82% 성능 달성

---

## 1. 문제: Instruction 데이터의 비용

### 1.1 기존 방식의 한계

```
사람이 작성하는 instruction 데이터:
- 비용: 수천~수만 달러
- 시간: 수개월
- 확장성: 제한적

예: InstructGPT
- 13K demonstration
- 33K comparison
- 수십 명의 라벨러
```

### 1.2 Self-Instruct의 아이디어

```
LLM을 이용해 instruction 데이터를 자동 생성!

적은 시드 (175개) → 대량 데이터 (52K)
부트스트래핑 방식
```

---

## 2. Self-Instruct 파이프라인

### 2.1 전체 구조

```
시드 태스크 (175개)
       ↓
 새 instruction 생성
       ↓
 instance 생성 (input-output)
       ↓
    필터링
       ↓
  Task Pool에 추가
       ↓
     반복...
       ↓
  52K instruction
```

### 2.2 4단계 프로세스

1. **Instruction 생성**
2. **Input 생성**
3. **Output 생성**
4. **필터링**

---

## 3. 상세 알고리즘

### 3.1 Step 1: Instruction 생성

```python
# 기존 태스크에서 8개 샘플링
sampled_tasks = random.sample(task_pool, 8)

prompt = f"""
Come up with a series of tasks:

{format_tasks(sampled_tasks)}

Task 9:
"""

new_instruction = gpt3.generate(prompt)
```

### 3.2 Step 2: Input/Output 생성

```python
# Input-first 방식 (대부분)
input_first_prompt = f"""
Given the instruction: "{instruction}"

Generate an input for this instruction:
"""
input_text = gpt3.generate(input_first_prompt)

output_prompt = f"""
Instruction: {instruction}
Input: {input_text}
Output:
"""
output_text = gpt3.generate(output_prompt)
```

```python
# Output-first 방식 (일부 태스크)
output_first_prompt = f"""
Given the instruction: "{instruction}"

First generate a possible output, then generate an input that would lead to it:
"""
```

### 3.3 Step 3: 필터링

```python
def filter_instruction(new_inst, task_pool):
    # 1. 길이 필터
    if len(new_inst.split()) < 3 or len(new_inst.split()) > 150:
        return False

    # 2. 시작 단어 필터
    invalid_starts = ["image", "picture", "graph", "video", "audio"]
    if any(new_inst.lower().startswith(w) for w in invalid_starts):
        return False

    # 3. 유사도 필터 (ROUGE-L)
    for existing in task_pool:
        if rouge_l(new_inst, existing) > 0.7:
            return False

    return True
```

---

## 4. 시드 태스크

### 4.1 구성

```
총 175개 시드 태스크:
- 저자가 직접 작성: 89개
- 자유형 태스크: 86개

다양한 유형 포함:
- 분류
- 생성
- 추출
- 변환
- QA
```

### 4.2 시드 예시

```
Task 1: 주어진 문장의 감정을 분류하세요.
Task 2: 이 문단을 요약하세요.
Task 3: 두 문장이 같은 의미인지 판단하세요.
Task 4: 주어진 텍스트를 한국어로 번역하세요.
Task 5: 이 코드에서 버그를 찾으세요.
...
```

---

## 5. 실험 결과

### 5.1 평가 방법

```
SuperNaturalInstructions (SupNatInst):
- 1600+ NLP 태스크
- 인간 작성 instruction

비교:
- GPT-3 (vanilla)
- GPT-3 + Self-Instruct
- InstructGPT
```

### 5.2 결과

| 모델 | SupNatInst (252 tasks) |
|------|------------------------|
| GPT-3 (vanilla) | 28.1 |
| GPT-3 + Self-Instruct | 39.9 |
| InstructGPT | 43.1 |

**Self-Instruct로 +11.8% 향상!**

### 5.3 사람 평가

| 비교 | Self-Instruct 승률 |
|------|-------------------|
| vs GPT-3 | 71% |
| vs InstructGPT | 46% |

**InstructGPT의 82% 품질, 비용은 훨씬 적음!**

---

## 6. 생성된 데이터 분석

### 6.1 태스크 유형 분포

```
생성: 47%
분류: 31%
추출: 12%
변환: 6%
기타: 4%
```

### 6.2 품질 분석

사람 평가자 기준:
- 유효한 instruction: 88%
- 유효한 input: 91%
- 유효한 output: 85%

---

## 7. 코드 구현

### 7.1 전체 파이프라인

```python
class SelfInstruct:
    def __init__(self, seed_tasks, model="gpt-3.5-turbo"):
        self.task_pool = seed_tasks.copy()
        self.model = model

    def generate_instructions(self, num_instructions=100):
        new_tasks = []

        for _ in range(num_instructions):
            # 기존 태스크에서 샘플링
            samples = random.sample(self.task_pool, 8)

            # 새 instruction 생성
            new_inst = self._generate_instruction(samples)

            # 필터링
            if self._filter(new_inst):
                # Input/Output 생성
                input_text = self._generate_input(new_inst)
                output_text = self._generate_output(new_inst, input_text)

                task = {
                    "instruction": new_inst,
                    "input": input_text,
                    "output": output_text
                }

                new_tasks.append(task)
                self.task_pool.append(task)

        return new_tasks
```

### 7.2 프롬프트 템플릿

```python
INSTRUCTION_PROMPT = """
Come up with a diverse list of tasks. These tasks should be:
- Diverse in topic and domain
- Diverse in difficulty
- Diverse in type (generation, classification, etc.)

Examples:
{examples}

New Task:
"""

INPUT_PROMPT = """
Given the task below, generate a concrete input:

Task: {instruction}

Input:
"""

OUTPUT_PROMPT = """
Complete the following task:

Task: {instruction}
Input: {input}

Output:
"""
```

---

## 8. Alpaca: Self-Instruct의 응용

### 8.1 Stanford Alpaca

```
Self-Instruct 방법으로:
- GPT-3.5로 52K instruction 생성
- LLaMA-7B fine-tuning
- 비용: $500 미만

결과: GPT-3.5와 유사한 성능!
```

### 8.2 Alpaca 데이터 형식

```json
{
    "instruction": "Give three tips for staying healthy.",
    "input": "",
    "output": "1. Eat a balanced diet...\n2. Exercise regularly...\n3. Get enough sleep..."
}
```

---

## 9. 한계와 개선 방향

### 9.1 한계점

1. **품질 상한**: 생성 모델의 능력에 의존
2. **다양성**: 점차 비슷한 태스크 생성 경향
3. **사실성**: 환각(hallucination) 문제
4. **편향**: 시드 태스크의 편향 증폭

### 9.2 개선 방향

```
Evol-Instruct (WizardLM):
- 점진적으로 복잡한 instruction 생성

Self-Alignment:
- 모델이 자신의 output 평가

Rejection Sampling:
- 여러 개 생성 후 최고 선택
```

---

## 10. 쉬운 예시

### 10.1 언어 학습 비유

```
전통 방식:
선생님이 100개 문제를 직접 만듦
→ 비용 높고 시간 오래 걸림

Self-Instruct:
10개 예시 문제 제공
→ AI가 90개 유사 문제 자동 생성
→ 빠르고 저렴!
```

### 10.2 요리 레시피 비유

```
시드 레시피 (10개):
- 김치찌개
- 된장찌개
- 순두부찌개
...

자동 생성 (100개):
- 참치김치찌개
- 돼지고기된장찌개
- 해물순두부찌개
- 버섯된장찌개
...

기존 패턴에서 변형 생성!
```

---

## 11. 핵심 요약

### 기억해야 할 것들

1. **핵심**: LLM으로 instruction 자동 생성
2. **방법**: 부트스트래핑 (175 → 52K)
3. **효율**: InstructGPT의 82% 성능, 훨씬 적은 비용
4. **영향**: Alpaca, Vicuna 등의 기반

### 주요 수치

| 항목 | 값 |
|------|-----|
| 시드 태스크 | 175개 |
| 생성 태스크 | 52K |
| 성능 (vs InstructGPT) | 82% |
| 비용 절감 | 90%+ |

### 알고리즘 요약

$$\text{Task Pool}_n = \text{Task Pool}_{n-1} \cup \text{Generate}(\text{Sample}(\text{Task Pool}_{n-1}))$$

### 실무 팁

- 다양한 시드 태스크 준비
- ROUGE-L로 중복 필터링
- 주기적으로 품질 검증
- Input-first vs Output-first 균형

---

## 참고 자료

1. [Self-Instruct 논문](https://arxiv.org/abs/2212.10560)
2. [GitHub](https://github.com/yizhongw/self-instruct)
3. [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)

---

*이전 리뷰: [Cosmopedia](./004_Cosmopedia.md)*
*다음 섹션: [Multilingual & Tokenizer](../3_Multilingual_Tokenizer/)*
