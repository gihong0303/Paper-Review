# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

**논문 발표**: 2025년 1월
**저자**: DeepSeek-AI Team
**소속**: DeepSeek AI
**논문 링크**: [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
**공식 구현**: [GitHub](https://github.com/deepseek-ai/DeepSeek-R1)
**모델**: [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1)

---

## 한 줄 요약
> 순수 강화학습만으로 LLM의 추론 능력을 유도하여, SFT 없이도 자기 검증, 반성, 긴 사고 사슬을 자발적으로 학습하고, OpenAI o1과 동등한 수학/코딩 성능을 달성한 최초의 오픈 추론 모델

---

## 1. 문제 정의

### 1.1 현재 추론 모델의 한계

OpenAI o1의 성공 이후 추론 모델에 대한 관심이 높아졌지만:

```
기존 접근법의 문제:
┌─────────────────────────────────────────┐
│ 1. 폐쇄적 개발                          │
│    - o1의 학습 방법 비공개              │
│    - 재현 불가능                        │
│                                         │
│ 2. SFT 의존                             │
│    - 고품질 CoT 데이터 필요             │
│    - 인간/모델이 만든 데이터에 한정     │
│                                         │
│ 3. 추론 패턴 고정                       │
│    - 학습 데이터의 패턴만 모방          │
│    - 새로운 추론 전략 발견 못함         │
└─────────────────────────────────────────┘
```

### 1.2 핵심 질문

> SFT 없이 순수 RL만으로 LLM이 추론 능력을 스스로 발견할 수 있을까?

DeepSeek의 답: **Yes! 그리고 더 강력한 추론 능력이 창발한다!**

### 1.3 목표

1. **오픈소스 추론 모델**: o1급 성능의 공개 모델
2. **순수 RL 검증**: SFT 없이 추론 능력 학습 가능성 입증
3. **창발적 행동**: 자기 검증, 반성 등 자발적 학습

---

## 2. 핵심 발견: RL만으로 추론 창발

### 2.1 DeepSeek-R1-Zero

SFT 전혀 없이, 순수 RL로만 학습한 모델:

```
DeepSeek-R1-Zero 학습:
┌─────────────────────────────────────┐
│ 시작: DeepSeek-V3-Base              │
│       (일반 언어 모델)               │
│                                     │
│ 학습: GRPO (순수 RL)                 │
│       - 수학 문제 정답 여부만 보상   │
│       - CoT 데이터 없음              │
│                                     │
│ 결과: 놀라운 추론 능력 창발!         │
│       - 자기 검증                   │
│       - 반성 (reflection)           │
│       - 긴 사고 사슬                │
└─────────────────────────────────────┘
```

### 2.2 창발된 능력들

학습 과정에서 **자발적으로** 나타난 행동:

| 능력 | 설명 | 예시 |
|------|------|------|
| **자기 검증** | 답을 스스로 확인 | "Let me verify this answer..." |
| **반성** | 실수를 인식하고 수정 | "Wait, I made an error. Let me reconsider..." |
| **탐색** | 여러 접근법 시도 | "Alternatively, we could try..." |
| **요약** | 사고 과정 정리 | "To summarize my reasoning..." |

### 2.3 Aha Moment

학습 중 발생한 놀라운 순간:

```
학습 초기 (Step 100):
"15 + 27 = 42"
→ 단순 계산만

학습 중기 (Step 1000):
"Let me calculate step by step:
 15 + 27 = 15 + 20 + 7 = 35 + 7 = 42"
→ 단계별 풀이 등장

Aha Moment (Step ~5000):
"Wait, let me verify: 42 - 27 = 15 ✓
 The answer is correct."
→ 자기 검증이 자발적으로 등장!

명시적으로 가르치지 않았는데 스스로 발견!
```

---

## 3. DeepSeek-R1 학습 파이프라인

### 3.1 4단계 학습 과정

```
DeepSeek-R1 학습 파이프라인:

Stage 1: Cold Start
    DeepSeek-V3-Base
         ↓
    소량의 긴 CoT 데이터로 SFT
    (수천 개)
         ↓
Stage 2: Reasoning RL
    GRPO로 추론 능력 강화
    - 수학, 코드, 과학 문제
    - 정확도 기반 보상
         ↓
Stage 3: Rejection Sampling
    Stage 2 모델로 데이터 생성
    - 정답인 것만 선별
    - 다양한 도메인에 적용
         ↓
Stage 4: Final RL
    전체 도메인에서 RL
    - 추론 + 일반 능력
    - 형식 보상 추가
         ↓
    DeepSeek-R1 완성
```

### 3.2 Stage 1: Cold Start

```python
# Cold Start 데이터 예시
cold_start_example = {
    "prompt": "Solve: What is 15% of 240?",
    "response": """<think>
I need to calculate 15% of 240.

Step 1: Convert percentage to decimal
15% = 15/100 = 0.15

Step 2: Multiply
0.15 × 240 = 36

Let me verify:
36 / 240 = 0.15 = 15% ✓
</think>

The answer is 36.
"""
}

# 목적:
# - 기본적인 "생각하는" 형식 학습
# - <think> 태그 사용법
# - 이후 RL의 시작점 제공
```

### 3.3 Stage 2: Reasoning RL with GRPO

GRPO (Group Relative Policy Optimization) 사용:

```python
def reasoning_rl_reward(prompt, response):
    """추론 RL의 보상 함수"""

    # 1. 정확도 보상
    if is_correct_answer(prompt, response):
        accuracy_reward = 1.0
    else:
        accuracy_reward = 0.0

    # 2. 형식 보상 (선택적)
    format_reward = 0.0
    if has_think_tags(response):
        format_reward += 0.1
    if not has_language_mixing(response):
        format_reward += 0.1

    return accuracy_reward + format_reward

# GRPO 업데이트
# 같은 prompt에서 여러 응답 샘플링
# 상대적 보상으로 advantage 계산
# (자세한 내용은 GRPO 리뷰 참조)
```

### 3.4 Stage 3: Rejection Sampling

Stage 2 모델로 대규모 데이터 생성:

```python
def rejection_sampling(model, prompts, num_samples=10):
    """Rejection Sampling으로 학습 데이터 생성"""
    dataset = []

    for prompt in prompts:
        responses = []

        # 여러 응답 생성
        for _ in range(num_samples):
            response = model.generate(prompt, temperature=0.7)
            responses.append(response)

        # 정답인 것만 선별
        correct_responses = [
            r for r in responses
            if is_correct_answer(prompt, r)
        ]

        if correct_responses:
            # 가장 좋은 응답 선택 (또는 랜덤)
            best = select_best(correct_responses)
            dataset.append({
                "prompt": prompt,
                "response": best
            })

    return dataset
```

### 3.5 Stage 4: Final RL

전체 도메인에서 종합 RL:

```python
def final_rl_reward(prompt, response, domain):
    """최종 RL의 멀티 도메인 보상"""

    if domain == "reasoning":
        # 정확도 + 형식
        return accuracy_reward(prompt, response) + format_reward(response)

    elif domain == "writing":
        # 창의성 + 일관성
        return creativity_score(response) + coherence_score(response)

    elif domain == "coding":
        # 실행 결과 + 코드 품질
        return execution_reward(prompt, response) + code_quality(response)

    else:
        # 일반 도메인: 유용성 + 무해성
        return helpfulness(response) + harmlessness(response)
```

---

## 4. 핵심 기술: GRPO

### 4.1 PPO vs GRPO

| 측면 | PPO | GRPO |
|------|-----|------|
| Critic 모델 | 필요 | **불필요** |
| Advantage | 가치 함수로 추정 | **그룹 상대 보상** |
| 메모리 | 4개 모델 | **2개 모델** |
| 안정성 | 높음 | 높음 |

### 4.2 GRPO 수식

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}$$

```python
def grpo_advantage(rewards):
    """그룹 내 상대적 advantage 계산"""
    mean = rewards.mean()
    std = rewards.std() + 1e-8

    advantages = (rewards - mean) / std
    return advantages

# 예시
rewards = [1.0, 1.0, 0.0, 1.0]  # 4개 응답의 보상
advantages = grpo_advantage(rewards)
# [0.58, 0.58, -1.74, 0.58]
# → 오답은 강하게 패널티
```

---

## 5. 실험 결과

### 5.1 수학 벤치마크

| 모델 | AIME 2024 | MATH-500 | 방법 |
|------|-----------|----------|------|
| GPT-4o | 9.3% | 74.6% | - |
| Claude 3.5 Sonnet | 16.0% | 78.3% | - |
| OpenAI o1-mini | 63.6% | 90.0% | 비공개 |
| OpenAI o1 | **79.2%** | 96.4% | 비공개 |
| DeepSeek-R1-Zero | 71.0% | 85.5% | 순수 RL |
| **DeepSeek-R1** | 79.8% | **97.3%** | 4단계 학습 |

**DeepSeek-R1이 o1과 동등 또는 우수!**

### 5.2 코딩 벤치마크

| 모델 | Codeforces Rating | LiveCodeBench |
|------|-------------------|---------------|
| GPT-4o | 808 | 32.8% |
| Claude 3.5 | 717 | 36.3% |
| o1-mini | 1650 | 53.8% |
| o1 | **1891** | 63.4% |
| **DeepSeek-R1** | 1886 | **65.9%** |

### 5.3 일반 벤치마크

| 벤치마크 | GPT-4o | o1 | DeepSeek-R1 |
|----------|--------|----|-------------|
| MMLU | 87.2% | 91.8% | **90.8%** |
| GPQA Diamond | 49.9% | 75.7% | **71.5%** |
| SimpleQA | 39.0% | 42.7% | **30.1%** |
| IF-Eval | 86.5% | 87.1% | **83.3%** |

추론 외 일반 태스크에서도 강력!

### 5.4 Distillation 모델 성능

작은 모델로 지식 증류:

| 모델 | AIME 2024 | MATH-500 |
|------|-----------|----------|
| QwQ-32B-Preview | 50.0% | 90.6% |
| DeepSeek-R1-Distill-Qwen-32B | **72.6%** | **94.3%** |
| DeepSeek-R1-Distill-Llama-70B | **70.0%** | **94.5%** |
| DeepSeek-R1-Distill-Qwen-7B | 55.5% | 83.9% |

Distill 모델도 기존 모델들을 크게 능가!

---

## 6. 구현

### 6.1 DeepSeek-R1 사용

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 모델 로드
model_name = "deepseek-ai/DeepSeek-R1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# 추론
prompt = """A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?

Think step by step and verify your answer."""

messages = [{"role": "user", "content": prompt}]
input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    temperature=0.6,
    top_p=0.95,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 6.2 Distill 모델 사용 (더 가벼움)

```python
# 7B Distill 모델
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 수학 문제
prompt = """Solve the following problem step by step:

If a train travels at 60 mph for the first half of a journey
and 40 mph for the second half, what is the average speed
for the entire journey?"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.0,  # 결정론적
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 6.3 R1 스타일 RL 학습 (간소화)

```python
import torch
from transformers import AutoModelForCausalLM

class R1StyleTrainer:
    """DeepSeek-R1 스타일 RL 학습기"""

    def __init__(self, model_name, learning_rate=1e-6):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).cuda()

        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).cuda()
        self.ref_model.eval()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        self.group_size = 4
        self.kl_coef = 0.1

    def compute_reward(self, prompt, response):
        """정확도 + 형식 보상"""
        # 정확도 (실제로는 더 복잡한 검증 필요)
        accuracy = 1.0 if self.check_answer(prompt, response) else 0.0

        # 형식 보상
        format_score = 0.0
        if "<think>" in response and "</think>" in response:
            format_score += 0.2
        if "verify" in response.lower() or "check" in response.lower():
            format_score += 0.1

        return accuracy + format_score

    def train_step(self, prompts):
        """GRPO 스타일 학습 스텝"""
        total_loss = 0

        for prompt in prompts:
            # 그룹 샘플링
            responses = self.sample_responses(prompt, self.group_size)

            # 보상 계산
            rewards = torch.tensor([
                self.compute_reward(prompt, r) for r in responses
            ])

            # GRPO advantage
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # 각 응답에 대해 손실 계산
            for response, advantage in zip(responses, advantages):
                loss = self.compute_loss(prompt, response, advantage)
                total_loss += loss

        # 업데이트
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return total_loss.item()

    def sample_responses(self, prompt, num_samples):
        """여러 응답 샘플링"""
        responses = []
        for _ in range(num_samples):
            with torch.no_grad():
                output = self.model.generate(
                    prompt,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                )
            responses.append(output)
        return responses

    def compute_loss(self, prompt, response, advantage):
        """PPO-clip 손실"""
        # 구현 생략 (GRPO 리뷰 참조)
        pass
```

---

## 7. 쉬운 예시로 이해하기

### 7.1 자전거 타기 비유

**기존 방식 (SFT)**:
```
선생님: "이렇게 페달을 밟으세요"
        "핸들은 이렇게 잡으세요"
        "넘어지면 이렇게 하세요"

학생: 가르쳐준 대로만 함
      → 새로운 상황에 약함
```

**R1 방식 (순수 RL)**:
```
선생님: "저기까지 가면 사탕 줄게" (보상만 제공)

학생: 직접 시행착오
      → 넘어지고, 일어나고
      → 균형 잡는 법 스스로 발견
      → "아, 속도를 내면 더 안정적이네!" (창발)
```

### 7.2 수학 문제 풀이 예시

**R1-Zero의 창발적 추론**:

```
문제: "농부가 양 17마리를 가지고 있다.
       9마리를 제외하고 모두 도망갔다.
       남은 양은 몇 마리인가?"

일반 모델:
"17 - 9 = 8마리"  ← 오답!

R1-Zero (RL로 학습 후):
"<think>
Let me carefully read this problem.

'All but 9 run away' means 9 did NOT run away.
So 9 sheep remain.

Wait, let me verify:
- Total: 17 sheep
- Run away: 17 - 9 = 8 sheep
- Remain: 9 sheep ✓

The wording is tricky. 'All but 9' = 'except 9'.
</think>

The farmer has 9 sheep left."

→ 함정을 인식하고, 검증까지 자발적으로!
```

### 7.3 창발의 시각화

```
학습 진행에 따른 응답 변화:

Step 0:     "42"
            (단답)

Step 1000:  "15 + 27 = 42"
            (계산 과정)

Step 5000:  "Let me calculate:
             15 + 27 = 42
             Check: 42 - 27 = 15 ✓"
            (검증 등장!)

Step 10000: "I need to add 15 and 27.
             Method 1: Direct: 15 + 27 = 42
             Method 2: 15 + 30 - 3 = 42
             Both give 42, so the answer is 42."
            (다중 접근법!)

보상은 "정답 여부"만 주었는데
검증, 다중 접근법이 자발적으로 창발!
```

---

## 8. 한계점 및 후속 연구

### 8.1 현재 한계점

1. **언어 혼합**: 영어/중국어 혼용 발생
   - 형식 보상으로 완화했지만 완전 해결 안 됨

2. **가독성**: R1-Zero의 출력이 읽기 어려움
   - 반복, 구조 부족
   - Cold start로 개선

3. **일반 태스크**: 추론 외 태스크에서 상대적으로 약함
   - 롤플레이, 창의적 글쓰기 등

4. **긴 추론 비용**: 토큰 사용량이 많음
   - 간단한 문제에도 긴 사고

### 8.2 후속 연구 방향

1. **효율적 추론**: 필요할 때만 긴 사고
2. **멀티모달 추론**: 이미지, 비디오 포함
3. **도구 사용**: 계산기, 검색 등 외부 도구
4. **자기 개선**: 지속적인 자기 학습

### 8.3 영향

- **오픈소스 추론 모델**: 최초의 o1급 오픈 모델
- **RL만으로 충분**: SFT 의존도 낮춤
- **창발적 능력**: 명시적 학습 없이 능력 획득 가능 입증

---

## 9. 핵심 요약

### 기억해야 할 것들

1. **R1-Zero**: 순수 RL만으로 추론 능력 창발
2. **4단계 파이프라인**: Cold Start → RL → Rejection Sampling → Final RL
3. **GRPO**: Critic 없는 효율적 RL 알고리즘
4. **창발적 행동**: 자기 검증, 반성이 자발적으로 등장

### 핵심 성능

| 벤치마크 | o1 | DeepSeek-R1 |
|----------|-----|-------------|
| AIME 2024 | 79.2% | 79.8% |
| MATH-500 | 96.4% | 97.3% |
| Codeforces | 1891 | 1886 |

### 실무 체크리스트

```python
# 1. API 사용 (가장 쉬움)
# https://platform.deepseek.com

# 2. 로컬 사용 (고성능 GPU 필요)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1",
    trust_remote_code=True,
)

# 3. Distill 모델 (더 가벼움)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

# 4. 추론 시 temperature 설정
# 수학/코드: temperature=0 (결정론적)
# 창의적: temperature=0.6-0.7
```

---

## 참고 자료

1. [DeepSeek-R1 논문](https://arxiv.org/abs/2501.12948)
2. [GitHub 저장소](https://github.com/deepseek-ai/DeepSeek-R1)
3. [HuggingFace 모델](https://huggingface.co/deepseek-ai/DeepSeek-R1)
4. [DeepSeek 플랫폼](https://platform.deepseek.com)
5. [DeepSeekMath 논문 (GRPO)](https://arxiv.org/abs/2402.03300)

---

*이전 리뷰: [Quiet-STaR](./004_Quiet-STaR.md)*
*다음 리뷰: [Chain-of-Thought](./001_Chain-of-Thought.md)*
