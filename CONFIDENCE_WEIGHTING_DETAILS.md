# Confidence 가중치 적용 상세 정리

## 개요

Confidence map을 사용하여 각 픽셀별 loss에 신뢰도를 가중치로 곱하여 계산합니다.
- **높은 confidence (1.0)**: 해당 픽셀의 loss에 더 큰 가중치
- **낮은 confidence (0.0)**: 해당 픽셀의 loss에 작은 가중치

---

## 1. L1 Loss with Confidence (`l1_loss_confidence`)

### 입력
- `network_output`: 예측 이미지 `[C, H, W]` 또는 `[B, C, H, W]`
- `gt`: Ground truth 이미지 `[C, H, W]` 또는 `[B, C, H, W]`
- `confidence`: Confidence map `[H, W]` 또는 `[B, H, W]` 또는 `[B, 1, H, W]`
  - 값 범위: `[0, 1]` (1 = 높은 신뢰도)

### 계산 과정

#### Step 1: 픽셀별 L1 Loss 계산
```python
pixel_loss = torch.abs(network_output - gt)  # [C, H, W] or [B, C, H, W]
```
- 각 픽셀의 절대 차이 계산

#### Step 2: Confidence Shape 정규화
```python
# [H, W] → [1, H, W] → [B, 1, H, W] 또는 [C, H, W]
if len(confidence.shape) == 2:  # [H, W]
    confidence = confidence.unsqueeze(0)  # [1, H, W]
if len(confidence.shape) == 3 and confidence.shape[1] != 1:  # [B, H, W]
    confidence = confidence.unsqueeze(1)  # [B, 1, H, W]
```

#### Step 3: 공간 크기 일치
```python
if confidence.shape[-2:] != pixel_loss.shape[-2:]:
    confidence = F.interpolate(confidence, size=pixel_loss.shape[-2:], 
                               mode='bilinear', align_corners=False)
```
- Confidence map 크기가 이미지와 다를 경우 bilinear interpolation

#### Step 4: 채널 확장
```python
# [B, 1, H, W] → [B, C, H, W] 또는 [1, H, W] → [C, H, W]
if len(pixel_loss.shape) == 4:  # [B, C, H, W]
    if confidence.shape[1] == 1:
        confidence = confidence.expand(-1, pixel_loss.shape[1], -1, -1)
elif len(pixel_loss.shape) == 3:  # [C, H, W]
    if confidence.shape[0] == 1:
        confidence = confidence.expand(pixel_loss.shape[0], -1, -1)
```
- Confidence를 이미지 채널 수만큼 확장

#### Step 5: 가중치 적용 및 정규화
```python
weighted_loss = (pixel_loss * confidence).sum() / (confidence.sum() + 1e-8)
```

**수식**:
```
L1_loss_confidence = Σ(pixel_loss × confidence) / Σ(confidence)
```

**의미**:
- 각 픽셀의 loss에 confidence를 곱함
- Confidence 합으로 나누어 정규화 (scale 유지)
- Confidence가 높은 픽셀의 loss에 더 큰 가중치

---

## 2. L2 Loss with Confidence (`l2_loss_confidence`)

### 입력
- 동일 (L1과 동일)

### 계산 과정

#### Step 1: 픽셀별 L2 Loss 계산
```python
pixel_loss = ((network_output - gt) ** 2)  # [C, H, W] or [B, C, H, W]
```
- 각 픽셀의 제곱 차이 계산

#### Step 2-4: Confidence Shape 정규화
- L1과 동일한 과정

#### Step 5: 가중치 적용
```python
weighted_loss = (pixel_loss * confidence).sum() / (confidence.sum() + 1e-8)
```

**수식**:
```
L2_loss_confidence = Σ((pred - gt)² × confidence) / Σ(confidence)
```

---

## 3. SSIM Loss with Confidence (`ssim_confidence`)

### 입력
- `img1`: 첫 번째 이미지 `[C, H, W]` 또는 `[B, C, H, W]`
- `img2`: 두 번째 이미지 `[C, H, W]` 또는 `[B, C, H, W]`
- `confidence`: Confidence map `[H, W]` 또는 `[B, H, W]` 또는 `[B, 1, H, W]`

### 계산 과정

#### Step 1: SSIM Map 계산
```python
ssim_map = _ssim(img1, img2, window, window_size, channel, size_average=False)
# [C, H, W] or [B, C, H, W]
```
- 각 픽셀 위치에서 SSIM 값 계산 (window 기반)
- SSIM 값 범위: `[0, 1]` (1 = 완전히 유사)

#### Step 2: Confidence Shape 정규화
```python
# [H, W] → [1, 1, H, W] 또는 [B, H, W] → [B, 1, H, W]
if len(confidence.shape) == 2:  # [H, W]
    confidence = confidence.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
elif len(confidence.shape) == 3:  # [B, H, W]
    confidence = confidence.unsqueeze(1)  # [B, 1, H, W]
```

#### Step 3: 공간 크기 일치
```python
if confidence.shape[-2:] != ssim_map.shape[-2:]:
    confidence = F.interpolate(confidence, size=ssim_map.shape[-2:], 
                               mode='bilinear', align_corners=False)
```

#### Step 4: 채널 확장
```python
# SSIM map과 동일한 shape로 확장
if len(ssim_map.shape) == 4:  # [B, C, H, W]
    if confidence.shape[1] == 1:
        confidence = confidence.expand(-1, ssim_map.shape[1], -1, -1)
elif len(ssim_map.shape) == 3:  # [C, H, W]
    if confidence.shape[0] == 1:
        confidence = confidence.expand(ssim_map.shape[0], -1, -1)
```

#### Step 5: 가중치 적용
```python
weighted_ssim = (ssim_map * confidence).sum() / (confidence.sum() + 1e-8)
```

**수식**:
```
SSIM_confidence = Σ(SSIM_map × confidence) / Σ(confidence)
SSIM_loss = 1.0 - SSIM_confidence
```

**의미**:
- SSIM 값이 높을수록 유사도가 높음
- Confidence가 높은 영역의 SSIM에 더 큰 가중치
- 최종 loss는 `1.0 - weighted_ssim`

---

## 4. 공통 특징

### Shape 처리
모든 confidence 가중치 함수는 다음을 자동 처리:
1. **다양한 입력 shape 지원**: `[H, W]`, `[B, H, W]`, `[B, 1, H, W]`
2. **크기 불일치 처리**: Bilinear interpolation으로 자동 조정
3. **채널 불일치 처리**: Confidence를 이미지 채널 수만큼 확장

### 정규화 방식
```python
weighted_loss = Σ(loss × confidence) / Σ(confidence)
```
- Confidence 합으로 나누어 정규화
- Confidence가 0인 픽셀은 loss 계산에서 제외됨
- `1e-8` 추가로 division by zero 방지

### 수학적 의미

**일반 Loss (confidence 없음)**:
```
Loss = (1/N) × Σ(pixel_loss)
```
- 모든 픽셀에 동일한 가중치

**Confidence 가중치 Loss**:
```
Loss = Σ(pixel_loss × confidence) / Σ(confidence)
```
- Confidence가 높은 픽셀에 더 큰 가중치
- Confidence가 낮은 픽셀의 영향 감소

---

## 5. train.py에서의 사용

### L1 Loss
```python
if confidence_map is not None:
    Ll1 = l1_loss_confidence(image, gt_image, confidence_map)
else:
    Ll1 = l1_loss_mask(image, gt_image)
```

### SSIM Loss
```python
if confidence_map is not None:
    Lssim = ssim_confidence(image, gt_image, confidence_map)
else:
    Lssim = ssim(image, gt_image)
```

### 최종 Loss
```python
loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - Lssim))
```

---

## 6. 예시

### Confidence Map 예시
```
Confidence Map (3x3):
[0.9, 0.8, 0.7]
[0.6, 0.5, 0.4]
[0.3, 0.2, 0.1]
```
- 좌상단: 높은 confidence (0.9)
- 우하단: 낮은 confidence (0.1)

### Loss 계산 예시
```
Pixel Loss (3x3):
[0.1, 0.2, 0.3]
[0.4, 0.5, 0.6]
[0.7, 0.8, 0.9]

Weighted Loss = Σ(pixel_loss × confidence) / Σ(confidence)
              = (0.1×0.9 + 0.2×0.8 + ... + 0.9×0.1) / (0.9+0.8+...+0.1)
              = 1.35 / 4.5
              = 0.3
```

**일반 Loss (confidence 없음)**:
```
Loss = mean(pixel_loss) = 0.5
```

**차이점**:
- Confidence 가중치 적용 시: 높은 confidence 영역(좌상단)의 loss에 더 큰 가중치
- 일반 loss: 모든 픽셀에 동일한 가중치

---

## 요약

1. **픽셀별 loss 계산**: 각 픽셀의 차이 계산
2. **Confidence shape 정규화**: 다양한 입력 shape 처리
3. **크기/채널 일치**: Interpolation 및 expansion
4. **가중치 적용**: `Σ(loss × confidence) / Σ(confidence)`
5. **정규화**: Confidence 합으로 나누어 scale 유지

**핵심 아이디어**: Confidence가 높은 픽셀의 loss에 더 큰 가중치를 부여하여, 신뢰할 수 있는 영역에 더 집중하여 학습


