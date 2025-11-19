# Don't Stop Pretraining: Adapt Language Models to Domains and Tasks

**논문 발표**: 2020년 (ACL 2020)
**저자**: Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, Noah A. Smith
**소속**: Allen Institute for AI, University of Washington
**논문 링크**: [arXiv:2004.10964](https://arxiv.org/abs/2004.10964)

---

## 한 줄 요약
> 사전 학습된 모델을 타겟 도메인 데이터로 추가 학습(DAPT)하고, 태스크 데이터로 한 번 더 학습(TAPT)하면 일관되게 성능이 향상됨을 체계적으로 증명

---

## 1. 핵심 개념

### 1.1 DAPT (Domain-Adaptive Pre-Training)

```
일반 코퍼스로 학습된 모델
         ↓
도메인 특화 코퍼스로 추가 학습
(의료, 법률, 과학 등)
         ↓
도메인 태스크에서 성능 향상
```

### 1.2 TAPT (Task-Adaptive Pre-Training)

```
도메인 적응된 모델
         ↓
태스크의 unlabeled 데이터로 추가 학습
(downstream task와 동일한 분포)
         ↓
해당 태스크에서 추가 성능 향상
```

### 1.3 전체 파이프라인

```
RoBERTa (일반) → DAPT → TAPT → Fine-tuning
                      ↓
               최고 성능!
```

---

## 2. 실험 설계

### 2.1 도메인

| 도메인 | 코퍼스 크기 | 출처 |
|--------|-------------|------|
| Biomedical | 7.55B tokens | PubMed |
| Computer Science | 8.10B tokens | S2ORC |
| News | 6.66B tokens | RealNews |
| Reviews | 2.11B tokens | Amazon |

### 2.2 태스크

각 도메인당 2개 태스크 (총 8개)

---

## 3. 주요 결과

### 3.1 DAPT 효과

| 도메인 | RoBERTa | +DAPT | 개선 |
|--------|---------|-------|------|
| Biomedical | 84.2 | **86.5** | +2.3 |
| CS | 79.7 | **82.5** | +2.8 |
| News | 93.3 | **94.1** | +0.8 |
| Reviews | 95.7 | **96.2** | +0.5 |

**모든 도메인에서 일관된 향상!**

### 3.2 TAPT 효과

| 방법 | ChemProt | RCT | ACL-ARC |
|------|----------|-----|---------|
| RoBERTa | 81.9 | 87.2 | 63.0 |
| +DAPT | 84.2 | 87.6 | 75.4 |
| +TAPT | 82.6 | 87.7 | 67.4 |
| **+DAPT+TAPT** | **84.5** | **88.1** | **76.4** |

### 3.3 핵심 발견

1. **DAPT는 항상 도움**: 도메인 데이터가 있으면 무조건 이득
2. **TAPT는 추가 이득**: 적은 데이터로도 효과적
3. **DAPT+TAPT 최고**: 두 방법 결합이 최적

---

## 4. 도메인 관련성 분석

### 4.1 Cross-domain 실험

```
News DAPT → CS 태스크: 성능 저하
CS DAPT → CS 태스크: 성능 향상

→ 도메인이 일치해야 효과적
```

### 4.2 도메인 거리 측정

Vocabulary overlap으로 도메인 유사도 측정:
- 유사한 도메인: 작은 이득
- 다른 도메인: 큰 이득

---

## 5. Curated TAPT

### 5.1 아이디어

Task 데이터가 적을 때, **유사한 unlabeled 데이터**를 찾아서 추가

### 5.2 방법

```python
# kNN으로 task 데이터와 유사한 문서 검색
similar_docs = knn_search(
    task_data,
    domain_corpus,
    k=50 * len(task_data)
)

# 유사 문서로 추가 학습
model.continue_pretraining(similar_docs)
```

### 5.3 결과

TAPT보다 추가 1-2% 향상

---

## 6. 한국어 적용 가이드

### 6.1 DAPT for Korean

```python
# 예: 한국어 의료 도메인
# 1. 한국어 의료 코퍼스 수집
medical_corpus = [
    "건강보험심사평가원 데이터",
    "의학 논문",
    "의료 뉴스",
    ...
]

# 2. DAPT 수행
model.continue_pretraining(
    medical_corpus,
    epochs=1-2,
    lr=1e-4
)
```

### 6.2 주의사항

1. **토크나이저**: 한국어 토큰이 충분한지 확인
2. **데이터 양**: 최소 수십만~수백만 문서
3. **학습률**: 원래 pre-training보다 낮게

---

## 7. 쉬운 예시

### 7.1 언어 학습 비유

```
일반 영어 (RoBERTa)
→ 의학 영어 추가 학습 (DAPT)
→ 병리학 논문 읽기 (TAPT)
→ 병리학 시험 (Fine-tuning)

단계마다 더 전문적으로!
```

### 7.2 요리사 비유

```
기본 요리 기술 (Pre-training)
→ 한식 요리 연습 (DAPT)
→ 김치찌개 집중 연습 (TAPT)
→ 김치찌개 대회 출전 (Fine-tuning)
```

---

## 8. 실무 체크리스트

### 8.1 DAPT 준비

- [ ] 타겟 도메인 코퍼스 수집 (최소 수억 토큰)
- [ ] 데이터 전처리 및 정제
- [ ] 도메인 특화 vocabulary 분석
- [ ] 학습률, 에폭 수 결정

### 8.2 TAPT 준비

- [ ] Task unlabeled 데이터 수집
- [ ] (선택) kNN으로 유사 데이터 확장
- [ ] 짧은 학습 (100 steps 정도)

---

## 9. 핵심 요약

### 기억해야 할 것들

1. **DAPT**: 도메인 코퍼스로 추가 학습
2. **TAPT**: 태스크 데이터로 추가 학습
3. **결합**: DAPT + TAPT가 최고
4. **교훈**: "학습을 멈추지 마라!"

### 수식 요약

$$\text{Final Model} = \text{Pretrain} \xrightarrow{\text{DAPT}} \xrightarrow{\text{TAPT}} \xrightarrow{\text{FT}}$$

### 실무 팁

- DAPT: 1-2 에폭, lr = 1e-4~5e-5
- TAPT: 100 steps, lr = 1e-4
- 데이터가 많을수록 효과 큼

---

## 참고 자료

1. [논문](https://arxiv.org/abs/2004.10964)
2. [GitHub](https://github.com/allenai/dont-stop-pretraining)
3. [Allen AI Blog](https://blog.allenai.org/)

---

*다음 리뷰: [Pythia](./002_Pythia.md)*
