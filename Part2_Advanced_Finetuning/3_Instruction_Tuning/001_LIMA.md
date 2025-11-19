# LIMA: Less Is More for Alignment

**논문 발표**: 2023년 (NeurIPS 2023)
**저자**: Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, Omer Levy
**소속**: Meta AI
**논문 링크**: [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)

---

## 한 줄 요약
> 단 1,000개의 신중하게 선별된 고품질 데이터만으로도 GPT-4에 필적하는 alignment 성능을 달성할 수 있음을 증명, "데이터 품질 > 양" 가설 검증

---

## 1. 핵심 가설: Superficial Alignment Hypothesis

> "모델의 지식과 능력은 대부분 사전학습에서 학습됨. Alignment는 단지 사용자와 상호작용하는 **형식/스타일**만 학습하면 됨"

따라서 적은 양의 고품질 데이터로 충분!

---

## 2. 데이터 구성

### 2.1 1,000개 예시의 구성

| 출처 | 개수 | 설명 |
|------|------|------|
| Stack Exchange | 200 | 고품질 Q&A |
| wikiHow | 200 | 방법론적 설명 |
| Pushshift Reddit | 200 | 다양한 주제 |
| 저자 직접 작성 | 250 | 고품질 시범 |
| Super-Natural Inst | 150 | NLP 태스크 |

### 2.2 선별 기준

1. **Diversity**: 다양한 주제와 형식
2. **Quality**: 도움이 되고 정확한 응답
3. **Style**: 일관된 톤과 형식

---

## 3. 실험 결과

### 3.1 인간 평가

| 비교 | LIMA 승률 |
|------|-----------|
| vs Alpaca 65B | **43% 동등, 50% 승리** |
| vs GPT-4 | **19% 동등, 43% 승리** |
| vs Bard | **42% 동등, 58% 승리** |

### 3.2 데이터 양 실험

| 데이터 수 | 성능 |
|-----------|------|
| 1,000 | 기준 |
| 2,000 | +2% |
| 7,000 (low quality) | -5% |

**품질이 양보다 중요!**

---

## 4. 핵심 교훈

### 4.1 데이터 엔지니어링의 중요성

```
나쁜 접근: 대량의 저품질 데이터 수집
좋은 접근: 소량의 고품질 데이터 신중히 선별
```

### 4.2 실무 적용

1. 품질 높은 시범 데이터 작성에 투자
2. 다양성 확보 (주제, 형식, 난이도)
3. 일관된 스타일 유지

---

## 5. 핵심 요약

1. **핵심**: 1,000개 고품질 > 대량 저품질
2. **가설**: Alignment는 스타일만 학습
3. **결과**: GPT-4와 경쟁 가능
4. **교훈**: 데이터 품질에 투자

---

## 참고 자료

1. [LIMA 논문](https://arxiv.org/abs/2305.11206)

---

*다음 리뷰: [Orca](./002_Orca.md)*
