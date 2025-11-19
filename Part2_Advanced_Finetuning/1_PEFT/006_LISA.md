# LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning

**논문 발표**: 2024년
**저자**: Rui Pan, Xiang Liu, Shizhe Diao, Renjie Pi, Jipeng Zhang, Chi Han, Tong Zhang
**소속**: UIUC, HKUST
**논문 링크**: [arXiv:2403.17919](https://arxiv.org/abs/2403.17919)

---

## 한 줄 요약
> 각 iteration에서 중요도 기반으로 일부 레이어만 선택적으로 학습하여, LoRA보다 메모리 효율적이면서 Full Fine-tuning에 더 가까운 성능 달성

---

## 1. 핵심 아이디어

### 1.1 관찰

모든 레이어가 동일하게 중요하지 않음:
- 하위 레이어: 일반적 특징
- 상위 레이어: 태스크 특화

### 1.2 LISA 방법

매 iteration마다 N개 레이어만 랜덤 샘플링하여 학습:

```python
def lisa_training_step(model, batch, num_layers_to_train=2):
    # 중요도 기반 레이어 샘플링
    # (보통 하위 레이어와 상위 레이어에 높은 확률)
    selected_layers = sample_layers(model, num_layers_to_train)

    # 선택된 레이어만 gradient 계산
    for layer in model.layers:
        layer.requires_grad = (layer in selected_layers)

    loss = model(batch)
    loss.backward()
```

---

## 2. 메모리 효율

```
Full FT:  모든 레이어 활성화 저장
LoRA:     Adapter 파라미터만
LISA:     선택된 N개 레이어만 활성화 저장

메모리: LISA < LoRA < Full FT
성능:   Full FT ≈ LISA > LoRA
```

---

## 3. 실험 결과

### 3.1 vs LoRA

| 모델 | 메트릭 | LoRA | LISA |
|------|--------|------|------|
| LLaMA-2-7B | MT-Bench | 5.14 | **5.74** |
| LLaMA-2-7B | GSM8K | 20.9 | **42.5** |

### 3.2 메모리 사용량

| 방법 | LLaMA-7B | LLaMA-70B |
|------|----------|-----------|
| Full FT | 63GB | OOM |
| LoRA | 21GB | 88GB |
| **LISA** | **20GB** | **72GB** |

---

## 4. 핵심 요약

1. **핵심**: 레이어별 중요도 샘플링
2. **장점**: LoRA보다 성능 좋고 메모리 유사
3. **특징**: Full FT의 대안으로 적합

---

## 참고 자료

1. [LISA 논문](https://arxiv.org/abs/2403.17919)

---

*이전 리뷰: [P-Tuning v2](./005_P-Tuning_v2.md)*
*다음 섹션: [Alignment](../2_Alignment/)*
