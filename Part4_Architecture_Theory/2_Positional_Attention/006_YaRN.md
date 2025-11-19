# YaRN: Efficient Context Window Extension of Large Language Models

**논문 발표**: 2023년 (ICLR 2024)
**저자**: Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole
**소속**: EleutherAI, Nous Research
**논문 링크**: [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
**공식 구현**: [GitHub](https://github.com/jquesnelle/yarn)

---

## 한 줄 요약
> RoPE의 주파수 성분을 차원별로 다르게 스케일링하여, 기존 방법 대비 10배 적은 토큰과 2.5배 적은 학습 스텝으로 효과적인 컨텍스트 확장을 달성

---

## 1. 문제 정의

### 1.1 컨텍스트 길이의 한계

LLM은 학습된 컨텍스트 길이를 넘어가면 성능이 급격히 저하:

```
LLaMA-2 (4K 컨텍스트) 성능:

위치 0-4K:   ████████████████████  100%
위치 4K-8K:  ████████░░░░░░░░░░░░  40%
위치 8K-16K: ██░░░░░░░░░░░░░░░░░░  10%

문제: 긴 문서, 대화 히스토리 처리 불가
```

### 1.2 기존 해결책들

| 방법 | 설명 | 한계 |
|------|------|------|
| **Position Interpolation** | 위치를 압축 | 근거리 정보 손실 |
| **NTK-aware** | 주파수 스케일링 | 고주파 성분 문제 |
| **더 긴 사전학습** | 처음부터 긴 컨텍스트 | 비용 막대 |

### 1.3 Position Interpolation의 문제

단순히 위치를 s배 압축하면:

$$\text{RoPE}(m, \theta) \rightarrow \text{RoPE}(m/s, \theta)$$

```
문제점:
원래: 위치 1, 2, 3, 4
압축: 위치 0.5, 1, 1.5, 2 (s=2)

→ 인접한 토큰들의 위치 차이가 줄어듦
→ 근거리 정보 구분 능력 저하
```

---

## 2. 배경 지식

### 2.1 RoPE (Rotary Position Embedding) 복습

RoPE는 위치 정보를 회전으로 인코딩:

$$\text{RoPE}(x_m, m, \theta_i) = x_m \cdot e^{im\theta_i}$$

여기서 $\theta_i = 10000^{-2i/d}$는 차원별 주파수.

```python
def rope_embedding(x, position, dim):
    """RoPE 적용"""
    theta = 10000 ** (-2 * torch.arange(dim // 2) / dim)
    freqs = position * theta

    # 회전 적용
    x_rotated = x * torch.cos(freqs) + rotate_half(x) * torch.sin(freqs)
    return x_rotated
```

### 2.2 주파수의 의미

RoPE의 각 차원은 다른 주파수를 가짐:

```
차원별 주파수:
차원 0-1:   θ₀ = 10000⁰ = 1        고주파 (로컬)
차원 2-3:   θ₁ = 10000^(-2/d)      ↓
차원 4-5:   θ₂ = 10000^(-4/d)      ↓
...                                 ↓
차원 d-2:   θ_n = 10000^(-1)       저주파 (글로벌)

고주파 차원: 가까운 토큰 구분
저주파 차원: 먼 토큰의 전체 위치
```

### 2.3 파장과 컨텍스트 관계

주파수 $\theta$의 파장:

$$\lambda_i = \frac{2\pi}{\theta_i}$$

| 차원 | 주파수 | 파장 | 역할 |
|------|--------|------|------|
| 처음 (고주파) | 1.0 | 6.28 | 6토큰 내 구분 |
| 중간 | 0.01 | 628 | 수백 토큰 패턴 |
| 마지막 (저주파) | 0.0001 | 62,832 | 전체 위치 |

---

## 3. 핵심 아이디어

### 3.1 차원별 다른 처리

YaRN의 핵심 통찰:
> **고주파 차원은 보존하고, 저주파 차원만 스케일링한다!**

```
Position Interpolation (기존):
모든 차원 동일하게 s배 압축

YaRN:
고주파 차원: 거의 그대로 (로컬 정보 보존)
중간 차원:   점진적 스케일링
저주파 차원: 강하게 스케일링 (확장 범위)

→ 근거리 정보도 보존하면서 컨텍스트 확장!
```

### 3.2 수학적 정의

YaRN의 스케일링 함수:

$$\theta'_i = \theta_i \cdot \frac{1}{s^{1 - \gamma(r_i)}}$$

여기서:
- $s$: 컨텍스트 확장 비율
- $r_i = \lambda_i / L$: 파장과 원래 컨텍스트 길이의 비
- $\gamma(r)$: 스케일링 정도를 결정하는 램프 함수

### 3.3 램프 함수

$$\gamma(r) = \begin{cases}
0 & \text{if } r < \alpha \\
1 & \text{if } r > \beta \\
\frac{r - \alpha}{\beta - \alpha} & \text{otherwise}
\end{cases}$$

```
γ(r) 램프 함수:
1.0 ─────────────────●───────
                    /
                   /
                  /
0.0 ─●───────────/
     α          β

r < α: γ=0 → 스케일링 없음 (고주파)
r > β: γ=1 → 완전 스케일링 (저주파)
중간:  점진적 전환
```

### 3.4 왜 이게 작동하는가?

```
예시: 4K → 64K 확장 (s=16)

고주파 차원 (파장 < 512):
- γ = 0, θ' = θ
- 인접 토큰 구분 능력 보존

저주파 차원 (파장 > 4096):
- γ = 1, θ' = θ/16
- 더 긴 범위를 커버

중간 차원:
- 부드럽게 전환
- 정보 손실 최소화
```

---

## 4. 알고리즘 상세

### 4.1 YaRN 주파수 계산

```python
import torch

def yarn_frequencies(
    dim: int,
    original_max_length: int,
    scale: float,
    alpha: float = 1.0,
    beta: float = 32.0,
    base: float = 10000.0,
):
    """YaRN 스케일링된 주파수 계산"""

    # 원래 RoPE 주파수
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    # 파장 계산
    wavelengths = 2 * torch.pi / freqs

    # 파장 비율
    ratio = wavelengths / original_max_length

    # 램프 함수 적용
    gamma = torch.zeros_like(ratio)
    gamma[ratio > beta] = 1.0
    mask = (ratio >= alpha) & (ratio <= beta)
    gamma[mask] = (ratio[mask] - alpha) / (beta - alpha)

    # YaRN 스케일링
    # θ' = θ / s^(1-γ)
    scaled_freqs = freqs / (scale ** (1 - gamma))

    return scaled_freqs
```

### 4.2 Attention 스케일링

YaRN은 temperature 스케일링도 추가:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d} \cdot t}\right)V$$

여기서 $t = 0.1 \ln(s) + 1$

```python
def yarn_attention_scale(scale: float) -> float:
    """YaRN attention temperature"""
    return 0.1 * math.log(scale) + 1.0

# 예시
# scale=4: t = 0.1 * ln(4) + 1 = 1.139
# scale=16: t = 0.1 * ln(16) + 1 = 1.277
```

### 4.3 완전한 YaRN RoPE 구현

```python
import torch
import torch.nn as nn
import math

class YaRNRotaryEmbedding(nn.Module):
    """YaRN Rotary Position Embedding"""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        scale: float = 1.0,
        original_max_position: int = 4096,
        alpha: float = 1.0,
        beta: float = 32.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position = original_max_position
        self.alpha = alpha
        self.beta = beta

        # YaRN 주파수 계산
        self.inv_freq = self._compute_yarn_frequencies()

        # Attention scale
        self.mscale = self._compute_mscale()

        # 캐시
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _compute_yarn_frequencies(self):
        # 기본 주파수
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )

        if self.scale == 1.0:
            return inv_freq

        # 파장
        wavelengths = 2 * math.pi / inv_freq

        # 비율
        ratio = wavelengths / self.original_max_position

        # 램프 함수
        low = max(math.floor(self.alpha * self.dim / 2), 0)
        high = min(math.ceil(self.beta * self.dim / 2), self.dim // 2)

        gamma = torch.zeros(self.dim // 2)

        for i in range(self.dim // 2):
            if ratio[i] < self.alpha:
                gamma[i] = 0
            elif ratio[i] > self.beta:
                gamma[i] = 1
            else:
                gamma[i] = (ratio[i] - self.alpha) / (self.beta - self.alpha)

        # YaRN 스케일링
        inv_freq_scaled = inv_freq / (self.scale ** (1 - gamma))

        return inv_freq_scaled

    def _compute_mscale(self):
        if self.scale <= 1:
            return 1.0
        return 0.1 * math.log(self.scale) + 1.0

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[2]

        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len

            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)

            # [seq_len, dim]
            emb = torch.cat((freqs, freqs), dim=-1)

            self._cos_cached = emb.cos() * self.mscale
            self._sin_cached = emb.sin() * self.mscale

        return (
            self._cos_cached[:seq_len].to(x.dtype),
            self._sin_cached[:seq_len].to(x.dtype),
        )


def apply_rotary_pos_emb(q, k, cos, sin):
    """Query와 Key에 RoPE 적용"""
    # q, k: [batch, heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim]

    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
```

### 4.4 HuggingFace 모델에 적용

```python
from transformers import LlamaForCausalLM, LlamaConfig

def extend_context_with_yarn(
    model_name: str,
    target_length: int,
    original_length: int = 4096,
):
    """기존 모델에 YaRN 적용"""

    config = LlamaConfig.from_pretrained(model_name)

    # 확장 비율 계산
    scale = target_length / original_length

    # YaRN 설정
    config.rope_scaling = {
        "type": "yarn",
        "factor": scale,
        "original_max_position_embeddings": original_length,
    }
    config.max_position_embeddings = target_length

    # 모델 로드
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    return model


# 사용 예시
model = extend_context_with_yarn(
    "meta-llama/Llama-2-7b-hf",
    target_length=32768,
    original_length=4096,
)
```

---

## 5. Dynamic YaRN

### 5.1 동적 스케일링

학습 없이 추론 시 동적으로 컨텍스트 확장:

```python
class DynamicYaRNRotaryEmbedding(YaRNRotaryEmbedding):
    """동적 YaRN - 시퀀스 길이에 따라 자동 스케일링"""

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[2]

        # 동적 스케일 계산
        if seq_len > self.original_max_position:
            # 원래 길이 초과 시 스케일링
            dynamic_scale = seq_len / self.original_max_position
            self.inv_freq = self._compute_yarn_frequencies_with_scale(
                dynamic_scale
            )
            self.mscale = 0.1 * math.log(dynamic_scale) + 1.0
        else:
            # 원래 범위 내면 스케일링 없음
            self.inv_freq = self._compute_yarn_frequencies_with_scale(1.0)
            self.mscale = 1.0

        # 나머지는 동일
        return super().forward(x, seq_len)
```

### 5.2 장점

| 특성 | Static YaRN | Dynamic YaRN |
|------|-------------|--------------|
| Fine-tuning | 필요 | **불필요** |
| 짧은 시퀀스 | 약간 저하 | **원래 성능** |
| 긴 시퀀스 | 좋음 | 좋음 |
| 사용 편의성 | 중간 | **높음** |

---

## 6. 쉬운 예시로 이해하기

### 6.1 라디오 주파수 비유

**Position Interpolation** (단순 압축):
```
원래 라디오:
- 88-108 MHz (FM)

2배 확장 시:
- 44-54 MHz로 전체 압축

문제: 저주파 라디오 신호와 간섭!
      음질 저하
```

**YaRN** (선택적 조정):
```
원래 라디오:
- 88-92 MHz: 뉴스 (고주파, 로컬)
- 92-100 MHz: 음악 (중간)
- 100-108 MHz: 국제방송 (저주파, 멀리)

2배 확장 시:
- 88-92 MHz: 그대로 (로컬 정보 보존)
- 92-100 MHz: 92-104 MHz로 조금 확장
- 100-108 MHz: 104-120 MHz로 크게 확장

→ 로컬 음질 유지하면서 더 먼 방송 수신!
```

### 6.2 줄자 비유

```
4K 컨텍스트 모델 = 4m 줄자

문제: 10m 방을 재야 함

Position Interpolation:
- 줄자의 모든 눈금을 0.4배로 축소
- 1cm 간격이 0.4cm로
- 근거리 측정이 부정확해짐!

YaRN:
- cm 눈금 (고주파): 거의 그대로
- 10cm 눈금 (중간): 약간 축소
- 1m 눈금 (저주파): 많이 축소

→ 작은 물건도 정확히 재면서 긴 거리 측정 가능!
```

### 6.3 숫자 예시

LLaMA-2 7B, 4K → 64K 확장:

```
차원 0 (고주파):
- 파장: 6.28 토큰
- 비율: 6.28 / 4096 = 0.0015 < α
- γ = 0 → 스케일링 없음
- θ' = θ

차원 64 (중주파):
- 파장: 628 토큰
- 비율: 628 / 4096 = 0.15 (α~β 사이)
- γ = 0.5 → 절반 스케일링
- θ' = θ / 16^0.5 = θ / 4

차원 127 (저주파):
- 파장: 62,832 토큰
- 비율: 62,832 / 4096 = 15.3 > β
- γ = 1 → 완전 스케일링
- θ' = θ / 16

결과: 근거리 정확도 유지 + 64K 컨텍스트 지원
```

---

## 7. 실험 결과

### 7.1 Perplexity 비교

다양한 컨텍스트 확장 방법 비교 (LLaMA-2 7B):

| 방법 | 8K PPL | 16K PPL | 32K PPL | 64K PPL |
|------|--------|---------|---------|---------|
| 기본 (4K) | 발산 | 발산 | 발산 | 발산 |
| PI | 5.89 | 6.45 | 7.21 | 8.12 |
| NTK | 5.72 | 6.31 | 7.15 | 8.34 |
| **YaRN** | **5.41** | **5.78** | **6.15** | **6.58** |

**YaRN이 모든 길이에서 최저 perplexity!**

### 7.2 학습 효율

같은 성능을 위한 학습 비용:

| 방법 | 토큰 수 | 학습 스텝 | GPU 시간 |
|------|---------|-----------|----------|
| PI | 4B | 10K | 기준 |
| NTK | 2B | 5K | 0.5× |
| **YaRN** | **0.4B** | **4K** | **0.1×** |

**YaRN은 PI 대비 10배 효율적!**

### 7.3 긴 컨텍스트 태스크

Passkey Retrieval (건초더미에서 바늘 찾기):

| 컨텍스트 | PI | NTK | YaRN |
|----------|-----|-----|------|
| 8K | 98% | 99% | 100% |
| 16K | 89% | 92% | 100% |
| 32K | 71% | 78% | 99% |
| 64K | 45% | 52% | 97% |
| 128K | 12% | 21% | **89%** |

### 7.4 실제 태스크 성능

GovReport 요약 (긴 문서):

| 모델 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------|---------|---------|---------|
| LLaMA-2 7B (4K) | 32.1 | 10.2 | 21.5 |
| + PI (32K) | 38.5 | 13.8 | 26.2 |
| + YaRN (32K) | **41.2** | **15.6** | **28.9** |

### 7.5 Dynamic YaRN 성능

Fine-tuning 없이 추론만으로:

| 설정 | 원래 범위 PPL | 확장 범위 PPL |
|------|---------------|---------------|
| Static YaRN | 5.41 (-3%) | 6.58 |
| **Dynamic YaRN** | **5.58** (0%) | **6.89** |

짧은 시퀀스에서 원래 성능 유지!

---

## 8. 한계점 및 후속 연구

### 8.1 현재 한계점

1. **여전히 Fine-tuning 필요**: 최적 성능을 위해서는 학습 필요
   - Dynamic YaRN은 성능 약간 저하

2. **하이퍼파라미터 튜닝**: α, β 최적값 찾기
   - 모델마다 다를 수 있음

3. **매우 긴 컨텍스트**: 128K+ 에서 성능 저하
   - 다른 기법과 결합 필요

4. **모델 특화**: LLaMA 계열에서 주로 검증
   - 다른 아키텍처에서 추가 검증 필요

### 8.2 후속 연구

1. **Self-Extend**: YaRN + attention 패턴 조작
2. **LongRoPE**: 더 긴 컨텍스트를 위한 개선
3. **PoSE**: Position Skip으로 추가 확장
4. **Landmark Attention**: 주요 위치만 기억

### 8.3 관련 연구

| 방법 | 접근 | YaRN과의 차이 |
|------|------|---------------|
| Position Interpolation | 균일 압축 | 차별화된 스케일링 |
| NTK-aware | base 수정 | 파장 기반 조정 |
| ALiBi | 선형 bias | RoPE 대체 |
| Sliding Window | 지역 attention | 전역 정보 손실 |

---

## 9. 핵심 요약

### 기억해야 할 것들

1. **핵심 아이디어**: 고주파는 보존, 저주파만 스케일링
2. **램프 함수**: α~β 사이에서 점진적 전환
3. **Attention 스케일**: $t = 0.1 \ln(s) + 1$
4. **Dynamic YaRN**: Fine-tuning 없이 적용 가능

### 핵심 수식

$$\theta'_i = \theta_i \cdot \frac{1}{s^{1 - \gamma(r_i)}}$$

### 실무 체크리스트

```python
# 1. HuggingFace에서 사용
from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained(
    "model_name",
    rope_scaling={
        "type": "yarn",
        "factor": 8.0,  # 4K → 32K
        "original_max_position_embeddings": 4096,
    },
    max_position_embeddings=32768,
)

# 2. 권장 파라미터
# alpha: 1.0 (기본값)
# beta: 32.0 (기본값)
# factor: target_length / original_length

# 3. Fine-tuning
# - 학습 데이터: 원래 데이터의 0.1% 정도
# - 학습 스텝: 400-1000
# - 학습률: 2e-5
```

---

## 참고 자료

1. [YaRN 논문](https://arxiv.org/abs/2309.00071)
2. [공식 GitHub](https://github.com/jquesnelle/yarn)
3. [EleutherAI 블로그](https://blog.eleuther.ai/yarn/)
4. [HuggingFace 문서](https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaConfig.rope_scaling)
5. [RoPE 논문](https://arxiv.org/abs/2104.09864)

---

*이전 리뷰: [Ring Attention](./005_Ring_Attention.md)*
*다음 리뷰: [RoPE](./001_RoPE.md)*
