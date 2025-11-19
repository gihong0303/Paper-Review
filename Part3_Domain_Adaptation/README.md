# Part 3. Domain Adaptation & Korean/Data Engineering

도메인 적응, 데이터 엔지니어링, 토크나이저 확장, 합성 데이터에 관한 핵심 논문들을 다룹니다.

## 개요

LLM을 특정 도메인이나 언어에 적응시키는 방법론과 고품질 데이터 구축 전략을 다룹니다. 특히 한국어 LLM 개발 시 필수적인 지식들을 포함합니다.

## 섹션 구성

### 1. Domain Adaptation & CPT (Continual Pre-training)
도메인 특화 모델 개발 방법론

| 논문 | 핵심 기여 |
|------|-----------|
| [Don't Stop Pretraining](./1_Domain_Adaptation_CPT/001_Dont_Stop_Pretraining.md) | DAPT, TAPT 개념 정립 |
| [Pythia](./1_Domain_Adaptation_CPT/002_Pythia.md) | 학습 데이터 순서와 구성 영향 분석 |
| [ChipNeMo](./1_Domain_Adaptation_CPT/003_ChipNeMo.md) | 도메인 특화 토크나이저 전략 |
| [Code Llama](./1_Domain_Adaptation_CPT/004_Code_Llama.md) | 코드 도메인 Long Context 전략 |

### 2. Data Engineering & Synthetic Data
데이터 품질과 합성 데이터 생성

| 논문 | 핵심 기여 |
|------|-----------|
| [Phi-1 (Textbooks)](./2_Data_Engineering/001_Phi-1.md) | 데이터 품질 > 모델 크기 증명 |
| [Phi-1.5](./2_Data_Engineering/002_Phi-1.5.md) | 합성 데이터 활용의 정석 |
| [FineWeb](./2_Data_Engineering/003_FineWeb.md) | 대규모 웹 데이터 필터링 |
| [Cosmopedia](./2_Data_Engineering/004_Cosmopedia.md) | 합성 데이터 대량 생성 방법론 |
| [Self-Instruct](./2_Data_Engineering/005_Self-Instruct.md) | LLM으로 학습 데이터 생성 |

### 3. Multilingual & Tokenizer (한국어 적용 시 필수)
다국어 모델과 토크나이저

| 논문 | 핵심 기여 |
|------|-----------|
| [BPE](./3_Multilingual_Tokenizer/001_BPE.md) | 모든 LLM 토크나이저의 시조 |
| [SentencePiece](./3_Multilingual_Tokenizer/002_SentencePiece.md) | 언어 독립 토크나이저 |
| [XGLM](./3_Multilingual_Tokenizer/003_XGLM.md) | 다국어 few-shot 능력 |
| [BLOOM](./3_Multilingual_Tokenizer/004_BLOOM.md) | 다국어 대규모 학습 |
| [NLLB](./3_Multilingual_Tokenizer/005_NLLB.md) | 200개 언어 번역 모델 |
| [Swallow](./3_Multilingual_Tokenizer/006_Swallow.md) | 일본어 CPT 방법론 (한국어 참고) |

## 선수 지식

- Transformer 아키텍처
- 토크나이저 기초 (Vocabulary, Token)
- Pre-training과 Fine-tuning 차이
- (선택) 정보 이론 기초

## 한국어 LLM 개발 로드맵

이 파트를 기반으로 한국어 LLM을 개발할 때:

1. **토크나이저 확장**: BPE/SentencePiece로 한국어 토큰 추가
2. **Continual Pre-training**: Don't Stop Pretraining + Swallow 참고
3. **데이터 구축**: FineWeb 스타일 필터링 + 합성 데이터
4. **평가**: 다국어 벤치마크 구성
