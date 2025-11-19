# ChipNeMo: Domain-Adapted LLMs for Chip Design

**논문 발표**: 2023년
**저자**: Mingjie Liu, Teodor-Dumitru Ene, Robert Kirber, Chris Cheng, et al.
**소속**: NVIDIA
**논문 링크**: [arXiv:2311.00176](https://arxiv.org/abs/2311.00176)

---

## 한 줄 요약
> 칩 설계 도메인에 특화된 LLM을 만들기 위해 DAPT, 토크나이저 확장, RAG를 결합하여 GPT-4 수준의 도메인 성능을 5배 작은 모델로 달성

---

## 1. 문제: 일반 LLM의 한계

### 1.1 칩 설계 도메인의 특수성

```
특수 언어:
- Verilog, VHDL (하드웨어 기술 언어)
- SystemVerilog
- 내부 설계 문서, 버그 리포트

일반 LLM 문제:
- 도메인 용어 이해 부족
- 코드 생성 품질 낮음
- 내부 지식 없음
```

### 1.2 해결 접근법

```
3가지 기술 결합:
1. Domain-Adaptive Pre-Training (DAPT)
2. Tokenizer 확장
3. Retrieval-Augmented Generation (RAG)
```

---

## 2. 데이터 준비

### 2.1 학습 데이터 구성

| 출처 | 토큰 수 | 비율 |
|------|---------|------|
| 내부 설계 문서 | 5.1B | 35% |
| 코드 (Verilog 등) | 6.7B | 46% |
| 공개 칩 데이터 | 2.8B | 19% |
| **총계** | **14.6B** | 100% |

### 2.2 데이터 전처리

```python
# 품질 필터링
def filter_chip_data(doc):
    # 1. 중복 제거
    if is_duplicate(doc):
        return False

    # 2. 최소 길이
    if len(doc) < 100:
        return False

    # 3. 도메인 관련성
    if domain_score(doc) < threshold:
        return False

    return True
```

---

## 3. 토크나이저 확장

### 3.1 왜 필요한가?

```
일반 토크나이저 문제:
"SystemVerilog" → ["System", "Ver", "ilog"]  # 3 토큰
"assign" → ["as", "sign"]  # 2 토큰

→ 토큰 효율성 낮음, 의미 손실
```

### 3.2 확장 방법

```python
# 도메인 특화 토큰 추가
new_tokens = [
    "SystemVerilog",
    "assign",
    "always_ff",
    "always_comb",
    ...
]

# 토크나이저 확장
tokenizer.add_tokens(new_tokens)

# 임베딩 초기화 (평균 방식)
for token in new_tokens:
    subword_ids = original_tokenizer(token)
    new_embedding = mean(embeddings[subword_ids])
    model.add_embedding(token, new_embedding)
```

### 3.3 효과

```
토큰 수 감소: 평균 3.3%
→ 학습/추론 속도 향상
→ 더 긴 컨텍스트 처리 가능
```

---

## 4. Domain-Adaptive Pre-Training

### 4.1 학습 설정

```python
# DAPT 설정
training_config = {
    "base_model": "LLaMA-2-70B",
    "learning_rate": 5e-5,  # 낮은 학습률
    "epochs": 1,
    "batch_size": 128,
    "warmup_ratio": 0.01
}
```

### 4.2 학습률 스케줄

```
Warmup → Cosine Decay → 최종 lr

초기 lr의 10%까지 감소
→ Catastrophic forgetting 방지
```

---

## 5. Retrieval-Augmented Generation

### 5.1 RAG 구성

```
질문 → Retriever → 관련 문서 검색
              ↓
        [문서 + 질문] → LLM → 답변
```

### 5.2 구현

```python
# Dense Retriever 학습
retriever = DenseRetriever(
    encoder="e5-small",  # 임베딩 모델
    index="FAISS",
    top_k=5
)

# RAG 파이프라인
def generate_with_rag(query):
    # 관련 문서 검색
    docs = retriever.retrieve(query)

    # 컨텍스트 구성
    context = "\n".join([d.text for d in docs])

    # LLM 생성
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    return llm.generate(prompt)
```

### 5.3 효과

특히 사실 기반 질문에서 큰 향상:
- 내부 문서 기반 QA
- API 사용법 질문
- 설계 규칙 확인

---

## 6. 평가 및 결과

### 6.1 평가 태스크

1. **Engineering Assistant Chatbot**: 기술 질문 응답
2. **EDA Script Generation**: 자동화 스크립트 생성
3. **Bug Summarization**: 버그 리포트 요약

### 6.2 결과

| 모델 | Chatbot | Script Gen | Bug Summary |
|------|---------|------------|-------------|
| LLaMA-2-70B | 37.6 | 42.1 | 54.3 |
| GPT-4 | 62.3 | 58.7 | 71.2 |
| **ChipNeMo-70B** | **59.1** | **61.4** | **68.5** |

**핵심**: 70B 모델이 GPT-4에 근접!

### 6.3 모델 크기별 비교

| 모델 크기 | DAPT 효과 |
|-----------|-----------|
| 7B | +18.2% |
| 13B | +15.7% |
| 70B | +12.4% |

→ 작은 모델일수록 DAPT 효과 큼

---

## 7. 각 기술의 기여도

### 7.1 Ablation Study

```
Base LLaMA-2: 37.6

+ Tokenizer 확장: 39.2 (+1.6)
+ DAPT: 52.8 (+13.6)
+ RAG: 59.1 (+6.3)

총 개선: +21.5
```

### 7.2 핵심 인사이트

1. **DAPT가 가장 중요** (60%+ 기여)
2. **RAG는 사실 기반 태스크에서 핵심**
3. **토크나이저 확장은 효율성 개선**

---

## 8. 실무 적용 가이드

### 8.1 도메인 적응 체크리스트

- [ ] 도메인 코퍼스 수집 (최소 1B+ 토큰)
- [ ] 도메인 특화 토큰 식별
- [ ] 토크나이저 확장 여부 결정
- [ ] DAPT 학습률 설정 (base의 1/10~1/100)
- [ ] RAG 시스템 구축 여부 결정

### 8.2 한국어 도메인 적용

```python
# 예: 한국어 법률 도메인
domain_config = {
    "corpus": [
        "법령 전문",
        "판례",
        "법률 뉴스",
        "법률 상담 QA"
    ],
    "special_tokens": [
        "제○조", "항", "호",
        "원고", "피고", "판결"
    ],
    "dapt_epochs": 1,
    "learning_rate": 1e-5
}
```

---

## 9. 쉬운 예시

### 9.1 의사 전문화 비유

```
일반의 (Base LLM)
→ 외과 전공의 수련 (DAPT)
→ 병원 내부 시스템 학습 (Tokenizer)
→ 의료 DB 참조 (RAG)
→ 전문 외과의 (ChipNeMo)

각 단계가 전문성을 더함!
```

### 9.2 언어 학습 비유

```
영어 기본 (Base)
→ 프로그래밍 용어 학습 (Tokenizer)
→ 개발 문서 대량 읽기 (DAPT)
→ Stack Overflow 검색 (RAG)
→ 개발자 영어 마스터
```

---

## 10. 핵심 요약

### 기억해야 할 것들

1. **3가지 결합**: DAPT + Tokenizer 확장 + RAG
2. **DAPT 가장 중요**: 전체 개선의 60%+
3. **토크나이저**: 도메인 용어 효율화
4. **RAG**: 사실 기반 질문에 필수

### 수식 요약

$$\text{Performance} = f(\text{DAPT}) + g(\text{RAG}) + h(\text{Tokenizer})$$

여기서 $f >> g > h$

### 실무 팁

- 데이터: 최소 1B 토큰
- 학습률: 원본의 1/10~1/100
- RAG: 도메인 문서 10만+ 권장
- 평가: 도메인 특화 벤치마크 필수

---

## 참고 자료

1. [ChipNeMo 논문](https://arxiv.org/abs/2311.00176)
2. [NVIDIA Technical Blog](https://developer.nvidia.com/blog/)

---

*이전 리뷰: [Pythia](./002_Pythia.md)*
*다음 리뷰: [Code Llama](./004_Code_Llama.md)*
