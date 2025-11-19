# Part 1. System & Inference Optimization

시스템 및 추론 최적화에 관한 핵심 논문들을 다룹니다.

## 개요

LLM의 실제 서비스 배포를 위해서는 추론 속도와 메모리 효율성이 매우 중요합니다. 이 파트에서는 GPU 메모리 최적화, 양자화, 추측적 디코딩 등 실용적인 최적화 기술들을 다룹니다.

## 섹션 구성

### 1. Kernel & Attention Optimization
GPU 커널 수준에서의 어텐션 연산 최적화

| 논문 | 핵심 기여 |
|------|-----------|
| [FlashAttention](./1_Kernel_and_Attention_Optimization/001_FlashAttention.md) | IO-aware 어텐션으로 메모리 접근 최적화 |
| [FlashAttention-2](./1_Kernel_and_Attention_Optimization/002_FlashAttention-2.md) | 병렬 처리 및 워크 파티셔닝 개선 |
| [PagedAttention (vLLM)](./1_Kernel_and_Attention_Optimization/003_PagedAttention.md) | OS 페이징 기법으로 KV Cache 관리 |
| [Flash-Decoding](./1_Kernel_and_Attention_Optimization/004_Flash-Decoding.md) | 긴 문맥 디코딩 가속화 |
| [Splitwise](./1_Kernel_and_Attention_Optimization/005_Splitwise.md) | Prefill/Decode 단계 분리 |

### 2. Quantization (양자화)
모델 가중치 및 활성화 값의 비트 수 감소를 통한 압축

| 논문 | 핵심 기여 |
|------|-----------|
| [LLM.int8()](./2_Quantization/001_LLM_int8.md) | 성능 저하 없는 8bit 추론 |
| [GPTQ](./2_Quantization/002_GPTQ.md) | 가중치 양자화의 표준 기술 |
| [AWQ](./2_Quantization/003_AWQ.md) | Activation-aware 양자화 |
| [SmoothQuant](./2_Quantization/004_SmoothQuant.md) | Activation outlier 문제 해결 |
| [SqueezeLLM](./2_Quantization/005_SqueezeLLM.md) | Dense-and-Sparse 하이브리드 |
| [BitNet](./2_Quantization/006_BitNet.md) | 1비트 양자화의 시작 |
| [BitNet b1.58](./2_Quantization/007_BitNet_b158.md) | {-1, 0, 1} 삼진법 양자화 |
| [Q-GaLore](./2_Quantization/008_Q-GaLore.md) | 학습 시 양자화 적용 |

### 3. Speculative Decoding & Speed
추측적 디코딩을 통한 생성 속도 향상

| 논문 | 핵심 기여 |
|------|-----------|
| [Speculative Decoding](./3_Speculative_Decoding/001_Speculative_Decoding.md) | 작은 모델로 초안, 큰 모델이 검수 |
| [Speculative Sampling](./3_Speculative_Decoding/002_Speculative_Sampling.md) | DeepMind의 병렬적 접근 |
| [Medusa](./3_Speculative_Decoding/003_Medusa.md) | Multiple Decoding Heads |
| [Eagle](./3_Speculative_Decoding/004_Eagle.md) | SOTA급 Speculative Decoding |

## 선수 지식

이 파트를 이해하기 위해 다음 개념들에 대한 기본적인 이해가 필요합니다:
- Transformer 아키텍처 (특히 Self-Attention)
- GPU 메모리 계층 구조 (HBM, SRAM)
- 부동소수점 표현 (FP32, FP16, INT8)
- 행렬 연산의 기초
