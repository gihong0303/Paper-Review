# GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection

**논문 발표**: 2024년 (ICML 2024 Oral)
**저자**: Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, Yuandong Tian
**소속**: UT Austin, Meta AI, Caltech, CMU
**논문 링크**: [arXiv:2403.03507](https://arxiv.org/abs/2403.03507)
**공식 구현**: [GitHub](https://github.com/jiaweizzhao/GaLore)

---

## 한 줄 요약
> 그래디언트의 저순위(low-rank) 구조를 활용하여 optimizer state를 저순위 부분 공간에 투영함으로써, 전체 파라미터 학습의 성능을 유지하면서 메모리 사용량을 최대 65.5% 절감

---

## 1. 문제 정의

### 1.1 LLM 학습의 메모리 문제

7B 모델 학습 시 메모리 사용량:

```
Full Fine-tuning 메모리 구성:
┌─────────────────────────────────────┐
│ 모델 파라미터 (FP16):    14 GB      │
│ Gradients (FP16):       14 GB      │
│ Optimizer State (Adam):  56 GB      │
│   - 1st moment (FP32):   28 GB     │
│   - 2nd moment (FP32):   28 GB     │
│ Activations:            ~10 GB      │
├─────────────────────────────────────┤
│ 총 필요 메모리:          ~94 GB     │
└─────────────────────────────────────┘

문제: 80GB A100으로도 7B 모델 full fine-tuning 불가!
```

### 1.2 기존 해결책들의 한계

| 방법 | 장점 | 단점 |
|------|------|------|
| **LoRA** | 메모리 효율 | 저순위 부분 공간 제한 |
| **Gradient Checkpointing** | Activation 절약 | Optimizer state 그대로 |
| **Mixed Precision** | 어느 정도 절약 | 한계 있음 |
| **Offloading** | 대용량 학습 | 매우 느림 |

### 1.3 핵심 관찰

**그래디언트 행렬은 학습 중 저순위(low-rank)가 된다!**

```
학습 중 그래디언트 특이값 분포:
┌─────────────────────────────┐
│   ●                         │
│   │●                        │
│   │ ●                       │
│   │  ●●                     │
│   │    ●●●●                 │
│   │        ●●●●●●●●●●●●●    │
│   └─────────────────────────│
│   Top singular values dominate!
└─────────────────────────────┘

상위 r개 특이값이 대부분의 정보를 담고 있음
```

---

## 2. 배경 지식

### 2.1 Optimizer State의 메모리 사용

Adam optimizer의 state:

```python
# Adam update
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t      # 1st moment
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²     # 2nd moment
θ_t = θ_{t-1} - lr * m_t / (√v_t + ε)

# 메모리: 각 파라미터당 2개의 state (m, v)
# 7B 모델 → 14B 개의 state 값 필요
```

### 2.2 LoRA의 접근법

LoRA는 **가중치 변화량**을 저순위로 제한:

$$W' = W_0 + BA$$

여기서 $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$, $r \ll \min(m, n)$

**한계**:
- 학습 가능한 공간이 저순위로 제한됨
- 사전학습에는 부적합

### 2.3 그래디언트의 저순위 구조

학습 중 그래디언트 행렬 $G \in \mathbb{R}^{m \times n}$의 특성:

$$G \approx U_r \Sigma_r V_r^T$$

여기서 $U_r \in \mathbb{R}^{m \times r}$, $\Sigma_r \in \mathbb{R}^{r \times r}$, $V_r \in \mathbb{R}^{r \times n}$

상위 $r$개 특이값만으로 대부분의 그래디언트 정보 보존 가능.

---

## 3. 핵심 아이디어

### 3.1 GaLore의 기본 개념

**그래디언트를 저순위 부분 공간에 투영하여 optimizer state 크기 감소:**

```
기존 Adam:
┌───────────────┐
│ Gradient G    │ → m, v 저장 (m×n 크기)
│ (m × n)       │
└───────────────┘

GaLore:
┌───────────────┐      ┌───────────────┐
│ Gradient G    │ --P→ │ 투영된 G̃     │ → m, v 저장 (m×r 크기!)
│ (m × n)       │      │ (m × r)       │
└───────────────┘      └───────────────┘
                   r << n
```

### 3.2 투영 행렬

두 가지 투영 방식:

1. **왼쪽 투영**: $\tilde{G} = P^T G$, $P \in \mathbb{R}^{m \times r}$
   - Optimizer state: $\mathbb{R}^{r \times n}$

2. **오른쪽 투영**: $\tilde{G} = G Q$, $Q \in \mathbb{R}^{n \times r}$
   - Optimizer state: $\mathbb{R}^{m \times r}$

### 3.3 투영 행렬 업데이트

주기적으로 투영 행렬을 SVD로 업데이트:

```
학습 과정:
Step 0-T:     P₁ 사용
Step T-2T:    G의 SVD → 새로운 P₂로 업데이트
Step 2T-3T:   P₂ 사용
...

투영 행렬이 현재 그래디언트 구조에 적응
```

---

## 4. 알고리즘 상세 설명

### 4.1 GaLore 업데이트 규칙

$m \leq n$인 경우 (일반적):

$$\tilde{G}_t = P_t^T G_t$$
$$\tilde{m}_t = \beta_1 \tilde{m}_{t-1} + (1 - \beta_1) \tilde{G}_t$$
$$\tilde{v}_t = \beta_2 \tilde{v}_{t-1} + (1 - \beta_2) \tilde{G}_t^2$$
$$\tilde{n}_t = \tilde{m}_t / (\sqrt{\tilde{v}_t} + \epsilon)$$
$$W_t = W_{t-1} - \eta P_t \tilde{n}_t$$

### 4.2 전체 알고리즘

```
알고리즘: GaLore (Gradient Low-Rank Projection)
────────────────────────────────────────────────
입력: 초기 가중치 W₀, 학습률 η, rank r,
      투영 업데이트 주기 T
출력: 학습된 가중치 W

1. P ← orthogonal_init(m, r)  # 초기 투영 행렬
2. m̃, ṽ ← 0                   # 저순위 optimizer state

3. for t = 1 to num_steps:
4.     # Forward & Backward
5.     G_t = ∇_W Loss(W_{t-1})
6.
7.     # 투영 행렬 업데이트 (주기적)
8.     if t mod T == 0:
9.         # SVD로 주요 부분 공간 추출
10.        U, S, V = SVD(G_t)
11.        P = U[:, :r]  # 상위 r개 특이 벡터
12.
13.    # 그래디언트 투영
14.    G̃_t = P^T @ G_t  # (r × n)
15.
16.    # Adam 업데이트 (저순위 공간에서)
17.    m̃ = β₁ * m̃ + (1 - β₁) * G̃_t
18.    ṽ = β₂ * ṽ + (1 - β₂) * G̃_t²
19.    ñ = m̃ / (√ṽ + ε)
20.
21.    # 업데이트를 원래 공간으로 복원
22.    ΔW = P @ ñ  # (m × n)
23.
24.    # 가중치 업데이트
25.    W_t = W_{t-1} - η * ΔW

26. return W
```

### 4.3 메모리 분석

| 구성 요소 | Full Adam | GaLore | 절약률 |
|-----------|-----------|--------|--------|
| Gradient | $m \times n$ | $m \times n$ | 0% |
| 1st moment | $m \times n$ | $r \times n$ | $1 - r/m$ |
| 2nd moment | $m \times n$ | $r \times n$ | $1 - r/m$ |
| 투영 행렬 | - | $m \times r$ | 추가 |

$r = 128$, $m = 4096$인 경우: **97% 절약**!

### 4.4 계산 오버헤드

```
추가 연산:
1. SVD (T 스텝마다): O(m²n) 또는 O(mn²)
2. 투영 (매 스텝): O(mrn)

SVD를 자주 하지 않으므로 (T=200) 오버헤드 작음
```

---

## 5. 구현

### 5.1 GaLore Optimizer

```python
import torch
from torch.optim import Optimizer
import torch.nn.functional as F

class GaLoreAdamW(Optimizer):
    """GaLore를 적용한 AdamW optimizer"""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        rank=128,
        update_proj_gap=200,
        scale=1.0,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay,
            rank=rank, update_proj_gap=update_proj_gap,
            scale=scale,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('GaLore does not support sparse gradients')

                state = self.state[p]

                # State 초기화
                if len(state) == 0:
                    state['step'] = 0
                    # 2D 텐서에만 GaLore 적용
                    if grad.dim() == 2:
                        state['projector'] = GaLoreProjector(
                            grad.shape[0], grad.shape[1],
                            rank=group['rank'],
                            update_proj_gap=group['update_proj_gap'],
                            scale=group['scale'],
                        )
                        # 저순위 optimizer state
                        state['exp_avg'] = torch.zeros(
                            group['rank'], grad.shape[1],
                            device=grad.device, dtype=grad.dtype
                        )
                        state['exp_avg_sq'] = torch.zeros(
                            group['rank'], grad.shape[1],
                            device=grad.device, dtype=grad.dtype
                        )
                    else:
                        # 1D 텐서는 일반 Adam
                        state['exp_avg'] = torch.zeros_like(grad)
                        state['exp_avg_sq'] = torch.zeros_like(grad)

                state['step'] += 1

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # GaLore 적용 여부
                if 'projector' in state:
                    projector = state['projector']

                    # 투영 행렬 업데이트 (주기적)
                    projector.update(grad, state['step'])

                    # 그래디언트 투영
                    grad_projected = projector.project(grad)

                    # Adam 업데이트 (저순위 공간)
                    exp_avg.mul_(beta1).add_(grad_projected, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(
                        grad_projected, grad_projected, value=1 - beta2
                    )

                    # Bias correction
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    # 정규화된 업데이트
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                    step_size = group['lr'] / bias_correction1
                    norm_grad = exp_avg / denom

                    # 원래 공간으로 복원
                    full_grad = projector.project_back(norm_grad)

                    # Weight decay
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                    # Update
                    p.add_(full_grad, alpha=-step_size)

                else:
                    # 일반 AdamW
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                    step_size = group['lr'] / bias_correction1

                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class GaLoreProjector:
    """SVD 기반 저순위 투영기"""

    def __init__(self, m, n, rank, update_proj_gap, scale):
        self.m = m
        self.n = n
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale

        # 투영 행렬 초기화
        self.ortho_matrix = None

    def update(self, grad, step):
        """투영 행렬 업데이트 (SVD)"""
        if step == 1 or step % self.update_proj_gap == 0:
            # SVD 계산
            if self.m <= self.n:
                # 왼쪽 특이 벡터 사용
                U, S, Vh = torch.linalg.svd(grad, full_matrices=False)
                self.ortho_matrix = U[:, :self.rank]
            else:
                # 오른쪽 특이 벡터 사용
                U, S, Vh = torch.linalg.svd(grad, full_matrices=False)
                self.ortho_matrix = Vh[:self.rank, :].T

    def project(self, grad):
        """그래디언트를 저순위 부분 공간으로 투영"""
        if self.m <= self.n:
            # P^T @ G: (r × m) @ (m × n) = (r × n)
            return self.ortho_matrix.T @ grad
        else:
            # G @ Q: (m × n) @ (n × r) = (m × r)
            return grad @ self.ortho_matrix

    def project_back(self, grad_projected):
        """저순위 업데이트를 원래 공간으로 복원"""
        if self.m <= self.n:
            # P @ G̃: (m × r) @ (r × n) = (m × n)
            return self.ortho_matrix @ grad_projected * self.scale
        else:
            # G̃ @ Q^T: (m × r) @ (r × n) = (m × n)
            return grad_projected @ self.ortho_matrix.T * self.scale
```

### 5.2 사용 예시

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from galore_optimizer import GaLoreAdamW

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
).cuda()

# GaLore를 적용할 파라미터 선택
galore_params = []
regular_params = []

for name, param in model.named_parameters():
    if param.requires_grad:
        # Linear 레이어의 2D 가중치에만 GaLore 적용
        if param.dim() == 2 and 'weight' in name:
            galore_params.append(param)
        else:
            regular_params.append(param)

# Optimizer 구성
optimizer = GaLoreAdamW(
    [
        {'params': galore_params, 'rank': 256, 'update_proj_gap': 200},
        {'params': regular_params, 'rank': None},  # 일반 AdamW
    ],
    lr=1e-5,
    weight_decay=0.01,
)

# 학습 루프
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 5.3 HuggingFace Trainer와 통합

```python
from transformers import Trainer, TrainingArguments

class GaLoreTrainer(Trainer):
    def create_optimizer(self):
        # GaLore 파라미터 분류
        galore_params = []
        regular_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.dim() == 2:
                    galore_params.append(param)
                else:
                    regular_params.append(param)

        self.optimizer = GaLoreAdamW(
            [
                {
                    'params': galore_params,
                    'rank': 256,
                    'update_proj_gap': 200,
                    'lr': self.args.learning_rate,
                },
                {
                    'params': regular_params,
                    'lr': self.args.learning_rate,
                },
            ],
            weight_decay=self.args.weight_decay,
        )

# 사용
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    num_train_epochs=3,
    bf16=True,
)

trainer = GaLoreTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

### 5.4 8-bit GaLore

```python
import bitsandbytes as bnb

class GaLoreAdamW8bit(GaLoreAdamW):
    """8-bit quantized GaLore optimizer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 8-bit state 사용
        self.use_8bit = True

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    if p.grad.dim() == 2:
                        # 8-bit state 초기화
                        state['exp_avg'] = bnb.functional.create_dynamic_map(
                            signed=True
                        )
                        # ... (8-bit 버전 구현)
```

---

## 6. 쉬운 예시로 이해하기

### 6.1 지도 압축 비유

**Full Adam**: 전체 고해상도 지도 저장

```
세계 지도 (7000 × 7000 픽셀)
┌─────────────────────┐
│ ■■■■■■■■■■■■■■■■■■■ │
│ ■■■■■■■■■■■■■■■■■■■ │
│ ■■■■■■■■■■■■■■■■■■■ │
│ ...                 │
└─────────────────────┘
저장 공간: 49 MB
```

**GaLore**: 주요 특징만 저장 + 복원 키

```
압축된 데이터:
┌──────┐     ┌─────────────────────┐
│ 키   │  +  │ 주요 특징 (128개)    │
│(P)   │     │ ■ ■ ■ ■ ■ ...       │
└──────┘     └─────────────────────┘
복원 시 키로 원래 지도 재구성

저장 공간: 1 MB (98% 절약)
```

### 6.2 학교 필기 비유

**Full Adam**: 모든 내용 완전 필기

```
수업 내용 전체:
- 중요한 공식 10개
- 예제 100개
- 설명 1000줄
→ 노트 10권 필요
```

**GaLore**: 핵심만 필기 + 구조 기억

```
핵심 개념 (투영):
- 주요 공식 10개
- 핵심 예제 10개
→ 노트 1권

나머지는 핵심에서 유추 가능!
```

### 6.3 숫자로 보는 절약

7B 모델, rank=256인 경우:

```
Linear 레이어 (4096 × 4096):

Full Adam state:
- exp_avg:    4096 × 4096 × 4 bytes = 67 MB
- exp_avg_sq: 4096 × 4096 × 4 bytes = 67 MB
- 총: 134 MB

GaLore state:
- exp_avg:    256 × 4096 × 4 bytes = 4.2 MB
- exp_avg_sq: 256 × 4096 × 4 bytes = 4.2 MB
- 투영 행렬:  4096 × 256 × 2 bytes = 2.1 MB
- 총: 10.5 MB

절약: 92%!

전체 모델 (200+ Linear 레이어):
Full Adam: ~56 GB
GaLore: ~6 GB
절약: ~89%
```

---

## 7. 실험 결과

### 7.1 Pre-training 성능

LLaMA 아키텍처, C4 데이터셋:

| 모델 | 방법 | Perplexity | 메모리 |
|------|------|------------|--------|
| 1B | Full-rank | 15.72 | 26.5 GB |
| 1B | LoRA (r=128) | 17.43 | 16.2 GB |
| 1B | **GaLore** | **15.89** | **15.1 GB** |
| 7B | Full-rank | OOM | - |
| 7B | LoRA (r=256) | 10.86 | 35.2 GB |
| 7B | **GaLore** | **10.52** | **22.0 GB** |

### 7.2 메모리 절약

| 설정 | Full Adam | GaLore | 절약률 |
|------|-----------|--------|--------|
| Optimizer state | 56 GB | 19.3 GB | 65.5% |
| + 8-bit | 28 GB | 4.9 GB | 82.5% |
| 총 학습 메모리 | 94 GB | 34.4 GB | 63.3% |

### 7.3 RTX 4090에서 7B 학습

**세계 최초로 24GB 소비자 GPU에서 7B 모델 전체 학습 성공!**

```
RTX 4090 (24GB) 설정:
- 8-bit GaLore
- Gradient checkpointing
- BF16 training

결과: Pre-training perplexity 10.52 달성
     (Full Adam과 동등)
```

### 7.4 LoRA와의 비교

Fine-tuning 태스크:

| 태스크 | Full FT | LoRA | GaLore | 메모리 |
|--------|---------|------|--------|--------|
| GLUE avg | 88.4 | 85.2 | **88.1** | -60% |
| SQuAD | 91.2 | 88.9 | **91.0** | -60% |
| CoNLL | 93.5 | 90.1 | **93.3** | -60% |

GaLore는 full fine-tuning 성능에 근접하면서 LoRA보다 우수!

### 7.5 Ablation Study

| 변형 | PPL | 설명 |
|------|-----|------|
| GaLore (기본) | 10.52 | T=200, r=256 |
| 투영 업데이트 없음 | 11.23 | 초기 P만 사용 |
| T=1000 | 10.89 | 덜 자주 업데이트 |
| r=128 | 10.78 | 더 낮은 rank |
| r=512 | 10.41 | 더 높은 rank |

---

## 8. 한계점 및 후속 연구

### 8.1 현재 한계점

1. **SVD 오버헤드**: 큰 행렬의 SVD는 비용이 높음
   - 현재: randomized SVD로 완화

2. **부분 공간 전환**: 투영 행렬 변경 시 momentum 손실
   - 초기화 문제 발생 가능

3. **최적 rank 선택**: 태스크/레이어마다 최적 rank 다름
   - 현재는 고정 rank 사용

4. **Activation 메모리**: Gradient checkpointing 필요
   - GaLore만으로는 activation 절약 안 됨

### 8.2 후속 연구

1. **GaLore 2 (2024)**:
   - 더 효율적인 SVD
   - 적응형 rank 선택

2. **Q-GaLore**:
   - Quantization과 결합
   - 4-bit training

3. **Distributed GaLore**:
   - 분산 학습에서 적용
   - 통신 효율 개선

### 8.3 관련 연구

| 방법 | 메모리 절약 위치 | GaLore와의 차이 |
|------|-----------------|-----------------|
| LoRA | 가중치 | GaLore는 그래디언트 |
| Gradient Checkpointing | Activation | 상호 보완적 |
| ZeRO | 분산 | 직교적 (함께 사용 가능) |
| Adafactor | Optimizer | 다른 근사 방식 |

---

## 9. 핵심 요약

### 기억해야 할 것들

1. **핵심 관찰**: 그래디언트 행렬은 학습 중 저순위
2. **투영**: 그래디언트를 저순위 부분 공간으로 투영
3. **메모리 절약**: Optimizer state가 r/m 비율로 축소
4. **성능 유지**: Full-rank 학습과 동등한 성능

### 핵심 수식

투영: $\tilde{G} = P^T G$

복원: $\Delta W = P \tilde{n}$

메모리: $O(rn)$ vs $O(mn)$

### 실무 체크리스트

```python
# 1. rank 선택 (128-512 권장)
rank = 256

# 2. 투영 업데이트 주기 (100-500)
update_proj_gap = 200

# 3. 적용할 레이어 선택
# - Linear weight에만 적용
# - Embedding, LayerNorm은 일반 Adam

# 4. 8-bit 버전 고려 (추가 절약)
# pip install bitsandbytes

# 5. Gradient checkpointing과 함께 사용
model.gradient_checkpointing_enable()
```

---

## 참고 자료

1. [GaLore 논문](https://arxiv.org/abs/2403.03507)
2. [공식 GitHub 저장소](https://github.com/jiaweizzhao/GaLore)
3. [ICML 2024 발표](https://icml.cc/virtual/2024/oral/35485)
4. [Hugging Face 통합](https://huggingface.co/papers/2403.03507)
5. [LoRA 논문](https://arxiv.org/abs/2106.09685)

---

*이전 리뷰: [LISA](./006_LISA.md)*
*다음 리뷰: [LoRA](./001_LoRA.md)*
