# GRPO: Group Relative Policy Optimization

**논문 발표**: 2024년 (DeepSeekMath 논문에서 소개)
**저자**: Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo
**소속**: DeepSeek AI
**논문 링크**: [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) (DeepSeekMath)
**공식 구현**: [GitHub](https://github.com/deepseek-ai/DeepSeek-Math)

---

## 한 줄 요약
> PPO의 Critic 모델을 제거하고 그룹 내 상대적 보상으로 Advantage를 추정하여, 메모리 사용량을 절반으로 줄이면서 동등하거나 더 나은 성능을 달성하는 RL 알고리즘

---

## 1. 문제 정의

### 1.1 PPO의 문제점

PPO (Proximal Policy Optimization)는 LLM alignment의 표준이지만 심각한 한계가 있음:

```
PPO 학습 시 필요한 모델들:
┌────────────────────────────────────────┐
│ 1. Policy Model (Actor)    - 학습 대상 │
│ 2. Reference Model         - KL 계산   │
│ 3. Reward Model            - 보상 제공 │
│ 4. Value Model (Critic)    - 가치 추정 │
└────────────────────────────────────────┘

7B 모델 기준 메모리:
- 각 모델: ~14 GB (FP16)
- 총 필요: 56 GB + Optimizer states
→ 80GB A100도 부족!
```

### 1.2 Critic 모델의 비효율성

| 문제점 | 설명 |
|--------|------|
| **높은 메모리** | Policy와 같은 크기의 모델 필요 |
| **학습 불안정** | 두 모델이 동시에 학습되어 불안정 |
| **느린 수렴** | Value 추정이 정확해질 때까지 대기 |
| **하이퍼파라미터** | Critic 전용 설정 필요 |

### 1.3 핵심 질문

> Critic 모델 없이도 효과적인 RL이 가능할까?

GRPO의 답: **그룹 샘플링으로 상대적 보상을 계산하면 된다!**

---

## 2. 배경 지식

### 2.1 RLHF의 목표

사람의 선호도에 맞게 LLM을 정렬:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta(y|x)} [r(x, y)] - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

- $r(x, y)$: 보상 (사람 선호도 반영)
- $\text{KL}$: Reference 모델과의 거리 (너무 멀어지지 않게)
- $\beta$: KL 페널티 강도

### 2.2 PPO의 Advantage 추정

PPO는 Generalized Advantage Estimation (GAE)를 사용:

$$A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

여기서 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

**문제**: $V(s)$를 추정하기 위해 별도의 Critic 모델 필요!

### 2.3 기존 대안들

| 방법 | 장점 | 단점 |
|------|------|------|
| **DPO** | Critic 불필요 | 온라인 학습 불가 |
| **REINFORCE** | 간단함 | 높은 분산 |
| **RLOO** | 분산 감소 | 여전히 절대 보상 사용 |

---

## 3. 핵심 아이디어

### 3.1 그룹 상대적 보상

같은 prompt에서 여러 응답을 샘플링하고, **그룹 내 상대적 순위**로 advantage 계산:

```
기존 PPO:
Prompt → 응답 1개 → Critic이 절대 가치 추정

GRPO:
Prompt → 응답 G개 샘플링 → 그룹 내 상대 순위 계산
         ↓
     r₁, r₂, ..., rG (보상)
         ↓
     정규화: Â_i = (r_i - mean) / std
         ↓
     높은 보상 응답 → 양의 advantage
     낮은 보상 응답 → 음의 advantage
```

### 3.2 수학적 정의

GRPO의 advantage 추정:

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}$$

또는 더 간단히 (outcome supervision):

$$\hat{A}_i = r_i - \text{mean}(\{r_1, ..., r_G\})$$

### 3.3 왜 이게 작동하는가?

1. **상대적 신호**: "이 응답이 다른 응답보다 좋은지"가 중요
2. **자동 베이스라인**: 그룹 평균이 자연스러운 베이스라인
3. **분산 감소**: 같은 prompt 내에서 비교하므로 노이즈 감소

```
예시:
Prompt: "2 + 2 = ?"

응답 1: "4" (정답)     → 보상: +1.0
응답 2: "5" (오답)     → 보상: -1.0
응답 3: "4" (정답)     → 보상: +1.0
응답 4: "3" (오답)     → 보상: -1.0

평균: 0.0, 표준편차: 1.0

Advantage:
응답 1: (1.0 - 0) / 1.0 = +1.0  (강화)
응답 2: (-1.0 - 0) / 1.0 = -1.0 (약화)
```

---

## 4. 알고리즘 상세 설명

### 4.1 GRPO 손실 함수

PPO-clip과 유사하지만 그룹 기반:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} \hat{A}_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon \right) \hat{A}_i \right) \right]$$

### 4.2 전체 알고리즘

```
알고리즘: GRPO (Group Relative Policy Optimization)
────────────────────────────────────────────────────
입력: 초기 정책 π_θ, 프롬프트 데이터셋 D,
      그룹 크기 G, 클리핑 ε, KL 계수 β
출력: 최적화된 정책 π_θ

1. for iteration = 1 to max_iterations:
2.     # 데이터 수집
3.     for each prompt q in batch:
4.         # 그룹 샘플링
5.         {o_1, ..., o_G} ~ π_θ_old(·|q)
6.
7.         # 보상 계산
8.         for i = 1 to G:
9.             r_i = RewardModel(q, o_i)
10.
11.        # Advantage 계산 (그룹 정규화)
12.        mean_r = mean({r_1, ..., r_G})
13.        std_r = std({r_1, ..., r_G})
14.        for i = 1 to G:
15.            Â_i = (r_i - mean_r) / std_r
16.
17.    # 정책 업데이트
18.    for epoch = 1 to K:
19.        for each (q, o_i, Â_i) in buffer:
20.            # Importance ratio
21.            ratio = π_θ(o_i|q) / π_θ_old(o_i|q)
22.
23.            # Clipped surrogate objective
24.            loss_clip = min(ratio * Â_i,
25.                           clip(ratio, 1-ε, 1+ε) * Â_i)
26.
27.            # KL penalty
28.            kl = log(π_θ_old(o_i|q) / π_θ(o_i|q))
29.
30.            # Total loss
31.            loss = -loss_clip + β * kl
32.
33.        # Gradient update
34.        θ = θ - α * ∇_θ loss

35. return π_θ
```

### 4.3 변형: Outcome vs Process Supervision

**Outcome Supervision**: 최종 결과만 보상
```python
reward = 1.0 if final_answer_correct else 0.0
advantage = reward - group_mean_reward
```

**Process Supervision**: 각 단계에 보상
```python
# 수학 문제의 각 풀이 단계
rewards = [step_reward(step) for step in solution_steps]
advantages = compute_per_step_advantage(rewards, group_rewards)
```

### 4.4 토큰 레벨 KL Penalty

```python
def compute_token_kl(logits_policy, logits_ref):
    """토큰 레벨에서 KL divergence 계산"""
    # Policy 분포
    p = F.softmax(logits_policy, dim=-1)
    log_p = F.log_softmax(logits_policy, dim=-1)

    # Reference 분포
    log_q = F.log_softmax(logits_ref, dim=-1)

    # KL divergence
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return kl
```

---

## 5. 구현

### 5.1 PyTorch 기본 구현

```python
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

class GRPO:
    def __init__(
        self,
        model_name: str,
        group_size: int = 4,
        clip_epsilon: float = 0.2,
        kl_coef: float = 0.1,
        lr: float = 1e-6,
    ):
        self.device = torch.device("cuda")

        # Policy 모델
        self.policy = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(self.device)

        # Reference 모델 (frozen)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef

        self.optimizer = AdamW(self.policy.parameters(), lr=lr)

    def sample_group(self, prompt: str) -> list[str]:
        """같은 prompt에서 G개의 응답 샘플링"""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True
        ).to(self.device)

        responses = []
        for _ in range(self.group_size):
            with torch.no_grad():
                outputs = self.policy.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            responses.append(response)

        return responses

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """그룹 내 상대적 advantage 계산"""
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8  # 안정성

        advantages = (rewards - mean_reward) / std_reward
        return advantages

    def compute_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """시퀀스의 로그 확률 계산"""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, -1, shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask padding
        mask = (shift_labels != self.tokenizer.pad_token_id).float()
        sequence_log_prob = (token_log_probs * mask).sum(dim=-1)

        return sequence_log_prob

    def train_step(
        self,
        prompts: list[str],
        reward_fn,  # 보상 함수
    ) -> dict:
        """GRPO 학습 스텝"""
        total_loss = 0.0
        total_reward = 0.0

        for prompt in prompts:
            # 1. 그룹 샘플링
            responses = self.sample_group(prompt)

            # 2. 보상 계산
            rewards = torch.tensor([
                reward_fn(prompt, resp) for resp in responses
            ], device=self.device)

            # 3. Advantage 계산
            advantages = self.compute_advantages(rewards)

            # 4. 각 응답에 대해 손실 계산
            for response, advantage in zip(responses, advantages):
                full_text = prompt + response

                # 토큰화
                encoding = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)

                input_ids = encoding.input_ids
                attention_mask = encoding.attention_mask
                labels = input_ids.clone()

                # Prompt 부분 마스킹 (응답만 학습)
                prompt_length = len(self.tokenizer.encode(prompt))
                labels[:, :prompt_length] = -100

                # Old policy log prob (detached)
                with torch.no_grad():
                    old_log_prob = self.compute_log_probs(
                        self.policy, input_ids, attention_mask, labels
                    )

                # Current policy log prob
                new_log_prob = self.compute_log_probs(
                    self.policy, input_ids, attention_mask, labels
                )

                # Reference model log prob
                with torch.no_grad():
                    ref_log_prob = self.compute_log_probs(
                        self.ref_model, input_ids, attention_mask, labels
                    )

                # Importance ratio
                ratio = torch.exp(new_log_prob - old_log_prob)

                # Clipped objective
                clip_adv = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * advantage
                loss_clip = -torch.min(ratio * advantage, clip_adv)

                # KL penalty
                kl = old_log_prob - ref_log_prob
                loss_kl = self.kl_coef * kl

                # Total loss
                loss = loss_clip + loss_kl
                total_loss += loss.item()

                # Backward
                loss.backward()

            total_reward += rewards.mean().item()

        # Gradient step
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            "loss": total_loss / len(prompts),
            "reward": total_reward / len(prompts),
        }


# 사용 예시
def math_reward(prompt: str, response: str) -> float:
    """수학 문제 보상 함수 예시"""
    # 정답 추출 및 검증 (간소화)
    try:
        # response에서 숫자 추출
        answer = extract_number(response)
        correct = check_answer(prompt, answer)
        return 1.0 if correct else 0.0
    except:
        return 0.0


# 학습 루프
grpo = GRPO("meta-llama/Llama-2-7b-hf", group_size=4)

for epoch in range(100):
    prompts = get_batch_prompts(batch_size=8)
    metrics = grpo.train_step(prompts, math_reward)
    print(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, Reward={metrics['reward']:.4f}")
```

### 5.2 효율적인 배치 처리

```python
def batched_train_step(self, prompts: list[str], reward_fn):
    """배치로 효율적인 GRPO 학습"""

    # 모든 prompt에서 그룹 샘플링
    all_responses = []
    all_rewards = []

    for prompt in prompts:
        responses = self.sample_group(prompt)
        rewards = [reward_fn(prompt, r) for r in responses]

        all_responses.extend([(prompt, r) for r in responses])
        all_rewards.extend(rewards)

    # 그룹별 advantage 계산
    advantages = []
    idx = 0
    for _ in prompts:
        group_rewards = all_rewards[idx:idx + self.group_size]
        group_advantages = self.compute_advantages(
            torch.tensor(group_rewards)
        )
        advantages.extend(group_advantages.tolist())
        idx += self.group_size

    # 배치로 한 번에 forward/backward
    batch_texts = [p + r for p, r in all_responses]
    encodings = self.tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(self.device)

    # ... (배치 처리 계속)
```

### 5.3 DeepSeek-R1 스타일 학습

```python
class DeepSeekGRPO(GRPO):
    """DeepSeek-R1에서 사용된 GRPO 변형"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # R1은 추가적인 정규화 사용

    def compute_advantages(self, rewards, format_rewards=None):
        """
        DeepSeek-R1 스타일:
        - Accuracy reward + Format reward
        - 그룹 평균 빼기 (정규화 없이)
        """
        if format_rewards is not None:
            total_rewards = rewards + format_rewards
        else:
            total_rewards = rewards

        # 단순 평균 빼기 (std로 나누지 않음)
        baseline = total_rewards.mean()
        advantages = total_rewards - baseline

        return advantages

    def format_reward(self, response: str) -> float:
        """응답 형식 보상 (DeepSeek-R1)"""
        reward = 0.0

        # <think> 태그 사용 여부
        if "<think>" in response and "</think>" in response:
            reward += 0.5

        # 적절한 길이
        if 100 < len(response) < 2000:
            reward += 0.3

        # 언어 일관성 (영어/중국어 혼용 패널티)
        # ...

        return reward
```

---

## 6. 쉬운 예시로 이해하기

### 6.1 학교 시험 비유

**PPO (절대 평가)**:
```
선생님(Critic)이 각 학생의 점수를 예측:
"이 학생은 80점 받을 것 같아"

실제 점수: 85점
Advantage: 85 - 80 = +5 (예상보다 좋음, 보상)

문제: 선생님의 예측이 정확해야 함!
```

**GRPO (상대 평가)**:
```
한 반(Group)에서 시험:
학생 A: 90점
학생 B: 70점
학생 C: 85점
학생 D: 75점

평균: 80점

Advantage:
A: 90 - 80 = +10 (최고, 강하게 보상)
B: 70 - 80 = -10 (최하, 강하게 약화)
C: 85 - 80 = +5  (평균 이상, 보상)
D: 75 - 80 = -5  (평균 이하, 약화)

선생님(Critic) 필요 없음!
```

### 6.2 요리 대회 비유

**PPO**: 절대 점수로 평가
```
심사위원: "이 요리는 8점입니다"
예측: 7점
→ 예측보다 좋으므로 보상

문제: 절대적인 점수 기준 필요
```

**GRPO**: 같은 재료로 경쟁
```
같은 재료(prompt)로 4명이 요리:
요리사 A: 맛있음
요리사 B: 보통
요리사 C: 별로
요리사 D: 맛있음

상대 순위:
A, D: 평균 이상 → 보상
B, C: 평균 이하 → 약화

절대 점수 없이 상대 비교만으로 학습!
```

### 6.3 숫자 예시

```python
# 수학 문제: "15 + 27 = ?"

# 4개 응답 생성 (Group size = 4)
responses = [
    "42",           # 정답
    "42",           # 정답
    "41",           # 오답
    "Let me calculate... 15 + 27 = 42"  # 정답 (장황)
]

# 보상 (정확성)
rewards = [1.0, 1.0, 0.0, 1.0]

# GRPO Advantage 계산
mean = (1.0 + 1.0 + 0.0 + 1.0) / 4 = 0.75
std = sqrt(variance) ≈ 0.43

advantages = [
    (1.0 - 0.75) / 0.43 = +0.58,   # 정답: 양의 advantage
    (1.0 - 0.75) / 0.43 = +0.58,   # 정답: 양의 advantage
    (0.0 - 0.75) / 0.43 = -1.74,   # 오답: 음의 advantage
    (1.0 - 0.75) / 0.43 = +0.58,   # 정답: 양의 advantage
]

# 오답("41")의 확률은 크게 감소
# 정답들의 확률은 조금씩 증가
```

---

## 7. 실험 결과

### 7.1 DeepSeekMath 성능

GRPO로 학습한 DeepSeekMath-7B:

| 벤치마크 | DeepSeekMath-Base | +SFT | +GRPO | 개선 |
|----------|-------------------|------|-------|------|
| GSM8K | 64.1% | 82.9% | **88.2%** | +5.3% |
| MATH | 34.2% | 43.0% | **51.7%** | +8.7% |
| SVAMP | 74.0% | 85.3% | **87.8%** | +2.5% |
| SimulEq | 37.7% | 47.8% | **59.6%** | +11.8% |

### 7.2 PPO vs GRPO 비교

LLaMA-2 7B 기준:

| 지표 | PPO | GRPO |
|------|-----|------|
| MATH 정확도 | 48.3% | **51.7%** |
| GSM8K 정확도 | 86.1% | **88.2%** |
| GPU 메모리 | 4×14GB | **2×14GB** |
| 학습 시간 | 기준 | **-30%** |
| 하이퍼파라미터 | 복잡 | 간단 |

### 7.3 DeepSeek-R1에서의 성과

GRPO는 DeepSeek-R1의 핵심 학습 알고리즘:

| 모델 | 학습 방법 | AIME 2024 | MATH-500 |
|------|-----------|-----------|----------|
| GPT-4o | - | 9.3% | 74.6% |
| Claude 3.5 | - | 16.0% | 78.3% |
| DeepSeek-R1-Zero | GRPO only | 71.0% | 85.5% |
| **DeepSeek-R1** | SFT + GRPO | **79.8%** | **97.3%** |

### 7.4 Ablation Study

| 변형 | MATH | 설명 |
|------|------|------|
| Full GRPO | 51.7% | 기본 설정 |
| - Group normalization | 49.2% | 표준편차로 안 나눔 |
| - KL penalty | 47.8% | Reference 모델 없이 |
| Group size 2 | 50.1% | 그룹 크기 감소 |
| Group size 8 | 52.0% | 그룹 크기 증가 |

그룹 크기가 클수록 좋지만, 계산 비용도 증가.

---

## 8. 한계점 및 후속 연구

### 8.1 현재 한계점

1. **샘플링 비용**: 그룹 샘플링으로 추론 비용 증가
   - G=4이면 4배 더 많은 생성 필요

2. **보상 함수 의존**: 좋은 보상 함수 설계 필요
   - 스파스 보상에서 어려움

3. **그룹 크기 선택**: 최적 그룹 크기 찾기 어려움
   - 태스크마다 다름

4. **긴 응답**: 매우 긴 응답에서 분산 증가
   - 토큰 레벨 크레딧 할당 어려움

### 8.2 후속 연구

1. **GRPO + Process Reward**: 단계별 보상과 결합
   - DeepSeek-R1에서 활용

2. **적응형 그룹 크기**: 학습 중 그룹 크기 조정

3. **다중 보상 결합**: 정확성 + 형식 + 안전성

4. **Iterative GRPO**: 여러 라운드 반복 학습
   - DeepSeek-R1의 4단계 학습

### 8.3 관련 연구

| 방법 | 특징 | GRPO와의 관계 |
|------|------|---------------|
| RLOO | Leave-one-out 베이스라인 | 유사한 아이디어 |
| ReMax | 최대값을 베이스라인으로 | 다른 정규화 |
| RAFT | 거부 샘플링 | 비슷한 목표 |

---

## 9. 핵심 요약

### 기억해야 할 것들

1. **Critic 제거**: Value 모델 없이 RL 학습
2. **그룹 샘플링**: 같은 prompt에서 여러 응답 생성
3. **상대적 보상**: 그룹 내 순위로 advantage 계산
4. **메모리 절약**: PPO 대비 50% 메모리 감소

### 핵심 수식

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}$$

### 실무 체크리스트

```python
# 1. 그룹 크기 설정 (4-8 권장)
group_size = 4

# 2. 보상 함수 정의
def reward_fn(prompt, response):
    # 정확성, 형식 등 고려
    return accuracy_score + format_score

# 3. GRPO 학습
for batch in dataloader:
    # 그룹 샘플링
    responses = [sample_group(p, group_size) for p in batch]
    rewards = [[reward_fn(p, r) for r in resps]
               for p, resps in zip(batch, responses)]

    # Advantage 계산
    advantages = [compute_group_advantage(r) for r in rewards]

    # 정책 업데이트
    loss = grpo_loss(responses, advantages)
    loss.backward()
```

---

## 참고 자료

1. [DeepSeekMath 논문](https://arxiv.org/abs/2402.03300)
2. [DeepSeek-R1 논문](https://arxiv.org/abs/2501.12948)
3. [DeepSeek GitHub](https://github.com/deepseek-ai/DeepSeek-Math)
4. [PPO 논문](https://arxiv.org/abs/1707.06347)
5. [RLOO 논문](https://arxiv.org/abs/2402.14740)

---

*이전 리뷰: [SimPO](./007_SimPO.md)*
*다음 리뷰: [InstructGPT](./001_InstructGPT.md)*
