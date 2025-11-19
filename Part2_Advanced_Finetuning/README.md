# Part 2. Advanced Fine-tuning & Training Methodologies

PEFT, SFT, Alignment(RLHF/DPO), 분산 학습에 관한 핵심 논문들을 다룹니다.

## 개요

대규모 언어 모델을 특정 태스크나 선호도에 맞게 조정하는 방법론들입니다. 효율적인 파인튜닝부터 인간 선호도 학습까지 다양한 기법을 다룹니다.

## 섹션 구성

### 1. PEFT (Parameter-Efficient Fine-Tuning)
전체 파라미터 대신 일부만 학습하는 효율적 튜닝

| 날짜 | 논문 | 핵심 기여 |
|------|------|-----------|
| 2025-03-15 | [LoRA](./1_PEFT/001_LoRA.md) | Low-rank 행렬 분해로 파라미터 절감 |
| 2025-03-18 | [QLoRA](./1_PEFT/002_QLoRA.md) | 4bit 양자화 + LoRA로 메모리 혁신 |
| 2025-03-20 | [DoRA](./1_PEFT/003_DoRA.md) | Weight decomposition으로 성능 향상 |
| 2025-03-23 | [Prefix-Tuning](./1_PEFT/004_Prefix-Tuning.md) | 연속적 프롬프트 학습 |
| 2025-03-25 | [P-Tuning v2](./1_PEFT/005_P-Tuning_v2.md) | Deep Prompt Tuning |
| 2025-03-28 | [LISA](./1_PEFT/006_LISA.md) | 레이어별 중요도 샘플링 |

### 2. Alignment & Preference Learning
인간 선호도에 맞춘 모델 정렬

| 날짜 | 논문 | 핵심 기여 |
|------|------|-----------|
| 2025-03-30 | [InstructGPT](./2_Alignment/001_InstructGPT.md) | RLHF의 대중화 |
| 2025-04-02 | [DPO](./2_Alignment/002_DPO.md) | RL 없이 선호도 학습 |
| 2025-04-04 | [IPO](./2_Alignment/003_IPO.md) | DPO 과적합 해결 |
| 2025-04-07 | [KTO](./2_Alignment/004_KTO.md) | Good/Bad 라벨만으로 학습 |
| 2025-04-09 | [ORPO](./2_Alignment/005_ORPO.md) | SFT+DPO 통합 |
| 2025-04-12 | [SPIN](./2_Alignment/006_SPIN.md) | Self-play 반복 학습 |
| 2025-04-14 | [SimPO](./2_Alignment/007_SimPO.md) | Reference-free 단순화 |

### 3. Instruction Tuning & Data Selection
데이터 품질과 학습 방법론

| 날짜 | 논문 | 핵심 기여 |
|------|------|-----------|
| 2025-04-17 | [LIMA](./3_Instruction_Tuning/001_LIMA.md) | 1000개 고품질 데이터의 힘 |
| 2025-04-19 | [Orca](./3_Instruction_Tuning/002_Orca.md) | 사고 과정 학습 |
| 2025-04-22 | [NEFTune](./3_Instruction_Tuning/003_NEFTune.md) | 노이즈로 일반화 향상 |

### 4. Distributed Training
대규모 모델 분산 학습

| 날짜 | 논문 | 핵심 기여 |
|------|------|-----------|
| 2025-04-24 | [ZeRO](./4_Distributed_Training/001_ZeRO.md) | 메모리 최적화의 교과서 |
| 2025-04-27 | [Megatron-LM](./4_Distributed_Training/002_Megatron-LM.md) | 모델 병렬화의 기초 |

## 선수 지식

- Transformer 아키텍처
- 기본적인 딥러닝 학습 과정
- Gradient descent 및 Optimizer
- (Alignment) 강화학습 기초 (선택)
