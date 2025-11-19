# GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

**논문 발표**: 2022년 (ICLR 2023)
**저자**: Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
**소속**: IST Austria, ETH Zürich
**논문 링크**: [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
**공식 구현**: [GitHub](https://github.com/IST-DASLab/gptq)

---

## 한 줄 요약
> OBQ(Optimal Brain Quantization)를 대규모 모델에 효율적으로 적용할 수 있도록 개선하여, 175B 모델을 단일 GPU에서 3-4시간 만에 3-4bit까지 정확하게 양자화

---

## 1. 배경: Post-Training Quantization (PTQ)

### 1.1 PTQ vs QAT

| 방법 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **QAT** (Quantization-Aware Training) | 학습 중 양자화 시뮬레이션 | 높은 정확도 | 재학습 필요, 비용 큼 |
| **PTQ** (Post-Training Quantization) | 학습된 모델을 양자화 | 빠름, 데이터 적게 필요 | 정확도 저하 가능 |

GPTQ는 PTQ 방식 → **재학습 없이 빠르게 양자화**

### 1.2 양자화 대상

```
LLM의 구성 요소:

1. 가중치 (Weights) ← GPTQ가 양자화하는 부분
   - 고정된 값
   - 모델 크기의 대부분

2. 활성화 (Activations)
   - 입력에 따라 변함
   - GPTQ는 FP16 유지
```

### 1.3 왜 가중치만 양자화?

- 가중치: 미리 양자화 가능, 정적 분석 가능
- 활성화: 입력마다 다름, 동적 분석 필요 (더 어려움)

---

## 2. 기반 이론: Optimal Brain Quantization (OBQ)

### 2.1 문제 정의

가중치 행렬 W를 양자화할 때 출력 변화 최소화:

$$\arg\min_{\hat{W}} \|WX - \hat{W}X\|_2^2$$

여기서:
- W: 원본 FP16 가중치
- $\hat{W}$: 양자화된 가중치
- X: 입력 활성화 (calibration 데이터)

### 2.2 OBQ의 핵심 아이디어

가중치를 하나씩 양자화하면서, 아직 양자화되지 않은 가중치로 **오류를 보상**:

1. 가중치 $w_q$를 양자화
2. 발생한 오류를 계산
3. 나머지 가중치들을 조정하여 오류 보상

### 2.3 수학적 공식

가중치 $w_q$를 양자화할 때의 최적 업데이트:

$$\delta_F = -\frac{w_q - \text{quant}(w_q)}{[H_F^{-1}]_{qq}} \cdot (H_F^{-1})_{:,q}$$

여기서:
- $H_F = 2X_FX_F^T$: Hessian 행렬
- F: 아직 양자화되지 않은 가중치들의 집합
- $[H_F^{-1}]_{qq}$: 역 Hessian의 q번째 대각 원소

### 2.4 OBQ의 문제점

**시간 복잡도**: O(d_row · d_col³)

175B 모델에 적용하면:
- 예상 시간: **수백 GPU-시간**
- 비현실적!

---

## 3. GPTQ의 핵심 개선

### 3.1 개선 1: 고정된 양자화 순서

OBQ: 매 스텝마다 가장 좋은 가중치 선택 → O(d_col²) 탐색

GPTQ: **임의의 고정 순서**로 양자화
- 실험적으로 순서가 크게 중요하지 않음을 발견
- 탐색 비용 제거

### 3.2 개선 2: Lazy Batch Updates

Hessian 역행렬을 매번 업데이트하지 않고, **B개 열을 묶어서** 업데이트:

```python
# OBQ: 매 열마다 Hessian 업데이트
for i in range(d_col):
    quantize(W[:, i])
    update_hessian_inverse()  # 비용 큼

# GPTQ: B개 열마다 업데이트
for block in range(d_col // B):
    for i in range(B):
        quantize(W[:, block*B + i])
    update_hessian_inverse()  # B번에 한 번만
```

B = 128이 최적

### 3.3 개선 3: Cholesky 분해 활용

Hessian 역행렬 업데이트를 Cholesky 분해로 효율화:

$$H^{-1} = (H^{-1} - H^{-1}_{:,q} H^{-1}_{q,:} / H^{-1}_{qq})$$

이를 Cholesky 분해로 안정적이고 빠르게 계산

### 3.4 시간 복잡도 개선

| 알고리즘 | 시간 복잡도 | 175B 모델 |
|----------|-------------|-----------|
| OBQ | O(d_row · d_col³) | 수백 GPU-시간 |
| **GPTQ** | O(d_row · d_col²) | **3-4 GPU-시간** |

**100배 이상 빨라짐!**

---

## 4. GPTQ 알고리즘 상세

### 4.1 전체 알고리즘

```
알고리즘: GPTQ
─────────────────────────
입력: W (가중치), X (calibration 데이터), B (블록 크기)
출력: Q (양자화된 가중치)

1. H = 2XX^T + λI  # Hessian 계산
2. H^{-1} = Cholesky(H)^{-1}  # 역행렬

3. for i = 0, B, 2B, ... do:
4.     # 블록 내 양자화
5.     for j = i to i+B-1 do:
6.         q_j = quant(W[:, j])  # 열 양자화
7.         Q[:, j] = q_j
8.
9.         # 오류 계산
10.        error = (W[:, j] - q_j) / H^{-1}[j, j]
11.
12.        # 블록 내 나머지 열에 오류 보상
13.        W[:, j+1:i+B] -= error × H^{-1}[j, j+1:i+B]
14.    end for
15.
16.    # 블록 이후 열들에 오류 보상
17.    W[:, i+B:] -= E × H^{-1}[i:i+B, i+B:]
18.
19. end for

return Q
```

### 4.2 단계별 시각화

```
초기 가중치 행렬:
┌─────────────────────────────────┐
│ w11  w12  w13  w14  w15  w16 ...│
│ w21  w22  w23  w24  w25  w26 ...│
│ w31  w32  w33  w34  w35  w36 ...│
└─────────────────────────────────┘

Block 1 (열 1-3) 처리:
┌─────────────────────────────────┐
│[Q11][Q12][Q13] w14' w15' w16'...│  ← 양자화 완료
│[Q21][Q22][Q23] w24' w25' w26'...│  ← 오류 보상됨
│[Q31][Q32][Q33] w34' w35' w36'...│
└─────────────────────────────────┘
     양자화됨     보상됨
```

### 4.3 오류 보상의 직관

```python
# 예: w1을 양자화할 때
w1 = 0.73
q1 = 1.0  # 가장 가까운 양자화 값
error = 0.73 - 1.0 = -0.27

# 이 오류를 w2, w3 등에 분배
# 가장 관련 있는 (Hessian 값 큰) 가중치에 더 많이
w2_new = w2 + α * error
w3_new = w3 + β * error
...

# 결과: 전체 출력 오차가 최소화됨
```

---

## 5. 양자화 형식

### 5.1 Uniform Quantization

```python
# n-bit 균일 양자화
def quantize(x, n_bits):
    qmin = 0
    qmax = 2**n_bits - 1

    scale = (x.max() - x.min()) / qmax
    zero_point = round(-x.min() / scale)

    q = clamp(round(x / scale) + zero_point, qmin, qmax)
    return q, scale, zero_point
```

### 5.2 Group-wise Quantization

행 전체가 아닌 **그룹(128개)** 단위로 scale 적용:

```
가중치 행: [w1 w2 ... w128 | w129 w130 ... w256 | ...]
           ↓                ↓
         scale1           scale2

장점: 더 정확한 양자화
단점: scale 저장 공간 필요
```

일반적으로 group size = 128

### 5.3 비트 수 선택

| 비트 | 메모리 절약 | 정확도 손실 |
|------|-------------|-------------|
| 8bit | 2× | 거의 없음 |
| **4bit** | **4×** | **약간** |
| 3bit | 5.3× | 있음 |
| 2bit | 8× | 상당함 |

**4bit가 가장 실용적인 균형점**

---

## 6. 쉬운 예시로 이해하기

### 6.1 사진 압축 비유

고해상도 사진을 저용량으로 압축:

**단순 양자화**: 모든 픽셀을 독립적으로 압축
- 각 픽셀에 오류 발생
- 전체적으로 품질 저하

**GPTQ 방식**: 순차적으로 압축하며 보정
- 픽셀 1 압축 → 오류 발생
- 오류를 인접 픽셀에 분배 (dithering과 유사)
- 결과: 전체 이미지 특성 보존

### 6.2 팀 점수 조정 비유

팀원 10명의 점수를 반올림해야 함 (합계 유지):

**단순 반올림**:
- 모든 점수 독립적 반올림
- 합계가 크게 변할 수 있음

**GPTQ 방식**:
1. 첫 번째 점수 반올림: 오류 = -0.3
2. 두 번째 점수에 +0.3 보정 후 반올림
3. 계속 진행...
4. 결과: 합계 유지됨

---

## 7. 구현 세부사항

### 7.1 Calibration 데이터

```python
# C4 데이터셋에서 128개 샘플 사용
calibration_data = load_c4_samples(n=128)

# 각 레이어의 입력 활성화 수집
activations = {}
for batch in calibration_data:
    with torch.no_grad():
        # Forward pass하며 활성화 저장
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                activations[name].append(
                    module.input_activation
                )
```

128개 샘플로 충분! (더 많아도 큰 개선 없음)

### 7.2 Layer-wise 양자화

```python
def quantize_model(model, calibration_data):
    for layer in model.layers:
        # 이 레이어의 calibration 활성화 수집
        X = collect_activations(layer, calibration_data)

        for linear in layer.linear_modules:
            # GPTQ로 가중치 양자화
            W_quant = gptq_quantize(linear.weight, X)
            linear.weight = W_quant

        # 다음 레이어를 위해 활성화 업데이트
        calibration_data = layer(calibration_data)
```

### 7.3 CUDA 커널 (추론)

```python
# AutoGPTQ의 효율적 커널
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-GPTQ",
    use_safetensors=True,
    device="cuda:0",
    use_triton=False  # CUDA 커널 사용
)
```

---

## 8. 실험 결과

### 8.1 Perplexity 비교 (OPT 모델)

| 모델 | FP16 | RTN 4bit | **GPTQ 4bit** | GPTQ 3bit |
|------|------|----------|---------------|-----------|
| 125M | 27.65 | 34.22 | **27.99** | 30.33 |
| 350M | 22.00 | 25.15 | **22.28** | 23.78 |
| 1.3B | 14.63 | 15.84 | **14.79** | 15.52 |
| 6.7B | 10.86 | 11.39 | **10.95** | 11.29 |
| 13B | 10.13 | 10.56 | **10.22** | 10.50 |
| **175B** | 8.34 | 8.68 | **8.37** | 8.68 |

**RTN** = Round-to-Nearest (단순 반올림)
**GPTQ 4bit ≈ FP16**

### 8.2 Zero-shot 정확도

| 모델 | 정밀도 | LAMBADA | HellaSwag | ARC |
|------|--------|---------|-----------|-----|
| BLOOM-176B | FP16 | 67.4 | 73.0 | 57.0 |
| BLOOM-176B | GPTQ 4bit | **67.1** | **72.8** | **57.3** |

### 8.3 양자화 시간

| 모델 | 파라미터 | 시간 (1 GPU) |
|------|----------|--------------|
| OPT-13B | 13B | 10분 |
| OPT-66B | 66B | 1시간 |
| BLOOM-176B | 176B | 4시간 |

RTX 3090 24GB 또는 A100에서 측정

### 8.4 메모리 및 속도

| 모델 | FP16 메모리 | GPTQ 4bit | 추론 속도 |
|------|-------------|-----------|-----------|
| LLaMA-7B | 14GB | 4GB | 1.5-2× faster |
| LLaMA-13B | 26GB | 7GB | 1.5-2× faster |
| LLaMA-30B | 60GB | 16GB | 1.5-2× faster |

---

## 9. 실무 사용법

### 9.1 AutoGPTQ 사용

```python
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

# 양자화된 모델 로드
model_name = "TheBloke/Llama-2-13B-GPTQ"
model = AutoGPTQForCausalLM.from_quantized(
    model_name,
    device="cuda:0",
    use_triton=True,
    quantize_config=None
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# 추론
prompt = "The meaning of life is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### 9.2 직접 양자화하기

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# 원본 모델 로드
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 양자화 설정
quantize_config = BaseQuantizeConfig(
    bits=4,           # 4bit 양자화
    group_size=128,   # 그룹 크기
    desc_act=False    # Activation order 비활성화
)

# Calibration 데이터 준비
examples = [
    tokenizer("The capital of France is", return_tensors="pt"),
    # ... 더 많은 예시
]

# 양자화 수행
model = AutoGPTQForCausalLM.from_pretrained(model, quantize_config)
model.quantize(examples)

# 저장
model.save_quantized("llama-2-7b-gptq")
```

### 9.3 vLLM에서 사용

```python
from vllm import LLM

# GPTQ 모델 자동 지원
llm = LLM(
    model="TheBloke/Llama-2-13B-GPTQ",
    quantization="gptq"
)

outputs = llm.generate(["Hello, my name is"])
```

---

## 10. 다른 방법들과 비교

### 10.1 vs LLM.int8()

| 측면 | LLM.int8() | GPTQ |
|------|------------|------|
| 비트 수 | 8bit | 4bit |
| 메모리 절약 | 2× | **4×** |
| 속도 | 비슷 | **더 빠름** |
| 복잡도 | 간단 | 중간 |

### 10.2 vs AWQ

| 측면 | GPTQ | AWQ |
|------|------|-----|
| 양자화 시간 | 빠름 | 더 빠름 |
| 정확도 | 좋음 | **더 좋음** |
| 원리 | 오류 보상 | 활성화 기반 |

### 10.3 선택 가이드

- **빠른 양자화 필요**: AWQ
- **가장 높은 정확도**: GPTQ (신중한 튜닝)
- **간단한 사용**: LLM.int8()

---

## 11. 한계점 및 후속 연구

### 11.1 한계점

1. **Calibration 데이터 의존성**:
   - 도메인에 따라 성능 차이

2. **양자화 시간**:
   - 큰 모델은 여전히 수 시간

3. **3bit 이하**:
   - 정확도 손실 있음

### 11.2 후속 연구

- **AWQ** (2023): Activation-aware로 더 정확
- **SqueezeLLM** (2023): Sparse + Dense 하이브리드
- **QuIP** (2023): Incoherence processing으로 2bit까지

---

## 12. 핵심 요약

### 기억해야 할 것들

1. **핵심 아이디어**: 순차적 양자화 + 오류 보상
2. **혁신**: OBQ를 O(d_col²)로 최적화
3. **결과**: 175B 모델을 4bit로, 성능 유지
4. **실용성**: 3-4시간 양자화, 4배 메모리 절약

### GPTQ의 핵심 공식

오류 보상:
$$\delta_F = -\frac{w_q - \text{quant}(w_q)}{[H_F^{-1}]_{qq}} \cdot (H_F^{-1})_{:,q}$$

### 실무 체크리스트

- [ ] Calibration 데이터 준비 (128개면 충분)
- [ ] Group size 선택 (128 권장)
- [ ] Bit 수 선택 (4bit가 최적)
- [ ] 효율적 커널 사용 (Triton 또는 CUDA)

---

## 참고 자료

1. [GPTQ 논문](https://arxiv.org/abs/2210.17323)
2. [공식 GitHub](https://github.com/IST-DASLab/gptq)
3. [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
4. [TheBloke의 GPTQ 모델들](https://huggingface.co/TheBloke)

---

*이전 리뷰: [LLM.int8()](./001_LLM_int8.md)*
*다음 리뷰: [AWQ](./003_AWQ.md)*
