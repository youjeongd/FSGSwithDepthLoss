# Loss 계산 과정 정리

## 단일 이미지 입력부터 Loss 계산까지의 흐름

### 1. 뷰포인트 선택 (Viewpoint Selection)
```python
# 랜덤하게 하나의 카메라 뷰포인트 선택
viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
```
- **입력**: Training camera 리스트에서 랜덤 선택
- **출력**: `viewpoint_cam` (카메라 정보 및 원본 이미지 포함)

---

### 2. 렌더링 (Rendering)
```python
render_pkg = render(viewpoint_cam, gaussians, pipe, background)
image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], ...
```
- **입력**: 
  - `viewpoint_cam`: 선택된 카메라 뷰포인트
  - `gaussians`: 3D Gaussian 모델
  - `pipe`: 렌더링 파이프라인 설정
  - `background`: 배경 색상
- **출력**: 
  - `image`: 렌더링된 이미지 `[C, H, W]`
  - `rendered_depth`: 렌더링된 depth map
  - 기타 렌더링 정보

---

### 3. Ground Truth 이미지 준비
```python
gt_image = viewpoint_cam.original_image.cuda()
```
- **입력**: `viewpoint_cam.original_image` (원본 이미지)
- **출력**: `gt_image` (CUDA 텐서, `[C, H, W]`)

---

### 4. Confidence Map 계산/로드 (선택적)

#### 4-1. 사전 생성된 Confidence Map 사용
```python
if confidence_dir is not None:
    confidence_map = load_confidence_map_for_camera(
        viewpoint_cam, dataset, confidence_dir, use_confidence, confidence_file_mapping
    )
```
- 파일에서 confidence map 로드
- 이미지 이름 기반으로 파일 경로 찾기

#### 4-2. 런타임 Confidence Map 계산
```python
else:
    # 캐시 확인
    if image_name in confidence_cache:
        confidence_map = confidence_cache[image_name].clone()
    else:
        # 새로 계산
        confidence_map = compute_confidence_map_runtime(
            gt_image, 
            num_augmentations=4,
            method='variance'
        )
        # 캐시에 저장
        confidence_cache[image_name] = confidence_map.clone()
```
- **입력**: `gt_image`
- **과정**: 
  1. 여러 augmentation 적용 (flip, grayscale, noise 등)
  2. 각 augmentation에 대해 MiDaS로 depth 추정
  3. Depth variance 계산
  4. Variance를 uncertainty로, uncertainty를 confidence로 변환
- **출력**: `confidence_map` `[H, W]` (값 범위: [0, 1], 1 = 높은 신뢰도)

---

### 5. Loss 계산

#### 5-1. L1 Loss 계산
```python
if confidence_map is not None:
    Ll1 = l1_loss_confidence(image, gt_image, confidence_map)
else:
    Ll1 = l1_loss_mask(image, gt_image)
```

**Confidence 가중치 사용 시 (`l1_loss_confidence`)**:
1. 픽셀별 L1 차이 계산: `pixel_loss = |image - gt_image|` `[C, H, W]`
2. Confidence 가중치 적용: `weighted_loss = (pixel_loss * confidence).sum() / confidence.sum()`
3. 최종 L1 Loss: 스칼라 값

**Confidence 없을 시 (`l1_loss_mask`)**:
1. 픽셀별 L1 차이 계산: `pixel_loss = |image - gt_image|`
2. 평균 계산: `Ll1 = pixel_loss.mean()`

---

#### 5-2. SSIM Loss 계산
```python
if confidence_map is not None:
    Lssim = ssim_confidence(image, gt_image, confidence_map)
else:
    Lssim = ssim(image, gt_image)
```

**Confidence 가중치 사용 시 (`ssim_confidence`)**:
1. SSIM map 계산: `ssim_map = _ssim(image, gt_image, ...)` `[C, H, W]`
2. Confidence 가중치 적용: `weighted_ssim = (ssim_map * confidence).sum() / confidence.sum()`
3. 최종 SSIM Loss: `1.0 - weighted_ssim` (스칼라)

**Confidence 없을 시 (`ssim`)**:
1. SSIM map 계산
2. 평균 계산: `Lssim = ssim_map.mean()`
3. 최종 SSIM Loss: `1.0 - Lssim`

---

#### 5-3. 최종 이미지 Loss
```python
loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - Lssim))
```
- **L1 Loss 가중치**: `(1.0 - opt.lambda_dssim)`
- **SSIM Loss 가중치**: `opt.lambda_dssim`
- **최종 Loss**: 두 loss의 가중 합

---

### 6. Depth Loss 추가 (선택적)

```python
rendered_depth = render_pkg["depth"][0]  # [H, W]
midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda()  # [H, W]

# Reshape to [N, 1]
rendered_depth = rendered_depth.reshape(-1, 1)
midas_depth = midas_depth.reshape(-1, 1)

# Pearson correlation 기반 depth loss
depth_loss = min(
    (1 - pearson_corrcoef(-midas_depth, rendered_depth)),
    (1 - pearson_corrcoef(1 / (midas_depth + 200.), rendered_depth))
)

loss += args.depth_weight * depth_loss
```
- **입력**: 
  - `rendered_depth`: 렌더링된 depth
  - `midas_depth`: MiDaS로 추정한 depth (ground truth)
- **계산**: 두 depth map 간의 Pearson correlation
- **최종 Loss에 추가**: `loss += depth_weight * depth_loss`

---

### 7. Pseudo Depth Loss (조건부)

```python
if iteration % args.sample_pseudo_interval == 0 and 
   iteration > args.start_sample_pseudo and 
   iteration < args.end_sample_pseudo:
    
    # Pseudo camera 선택 및 렌더링
    pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))
    render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background)
    
    # Depth 추정
    rendered_depth_pseudo = render_pkg_pseudo["depth"][0]
    midas_depth_pseudo = estimate_depth(render_pkg_pseudo["render"], mode='train')
    
    # Depth loss 계산
    depth_loss_pseudo = (1 - pearson_corrcoef(rendered_depth_pseudo, -midas_depth_pseudo)).mean()
    
    # Loss에 추가
    if not torch.isnan(depth_loss_pseudo).sum():
        loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
        loss += loss_scale * args.depth_pseudo_weight * depth_loss_pseudo
```
- 특정 iteration 범위에서만 실행
- Pseudo view에서 추가 depth loss 계산

---

## 최종 Loss 구성

```
Total Loss = Image Loss + Depth Loss + Pseudo Depth Loss

Image Loss = (1 - λ_dssim) × L1_Loss + λ_dssim × (1 - SSIM_Loss)
Depth Loss = depth_weight × depth_loss
Pseudo Depth Loss = loss_scale × depth_pseudo_weight × depth_loss_pseudo (조건부)
```

---

## 요약

1. **뷰포인트 선택** → 랜덤 카메라 선택
2. **렌더링** → Gaussian 모델로 이미지 생성
3. **GT 이미지 준비** → 원본 이미지 로드
4. **Confidence Map** → 계산 또는 로드 (선택적)
5. **L1 Loss** → Confidence 가중치 적용 (선택적)
6. **SSIM Loss** → Confidence 가중치 적용 (선택적)
7. **Image Loss** → L1과 SSIM의 가중 합
8. **Depth Loss** → Depth correlation 기반 loss 추가
9. **Pseudo Depth Loss** → 조건부 추가 loss
10. **최종 Loss** → 모든 loss의 합


