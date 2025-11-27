# generate_maps.py 활용을 위한 필요한 데이터 정리

## 1. 입력 데이터

### 필수 입력:
- **RGB 이미지**: Ground truth 이미지 또는 렌더링된 이미지
  - 형식: `torch.Tensor` 또는 `numpy.ndarray`
  - Shape: `[C, H, W]` 또는 `[H, W, C]` (C=3, RGB)
  - 값 범위: `[0, 1]` (정규화된 이미지) 또는 `[0, 255]` (원본 이미지)
  - 위치: `viewpoint_cam.original_image` 또는 `render_pkg["render"]`

### 모델 관련:
- **Encoder 가중치**: `encoder.pth`
- **Decoder 가중치**: `depth.pth`
- **모델 설정**: 
  - `num_layers`: ResNet 레이어 수
  - `height`, `width`: 이미지 크기

## 2. generate_maps.py 출력

### Uncertainty Map:
- 형식: `numpy.ndarray` 또는 `torch.Tensor`
- Shape: `[H, W]` (단일 채널)
- 값 범위: `[0, 1]` (정규화된 uncertainty)
- 저장 형식: 16-bit PNG (값 범위: `[0, 65535]`)

### Confidence 변환:
Uncertainty를 Confidence로 변환하는 방법:
```python
# 방법 1: 역변환 (1 - uncertainty)
confidence = 1.0 - uncertainty

# 방법 2: 지수 변환 (exp(-uncertainty))
confidence = torch.exp(-uncertainty)

# 방법 3: 정규화된 역변환
confidence = (1.0 - uncertainty) / (1.0 + uncertainty)
```

## 3. generate_maps.py 출력 경로

### 저장 위치:
`generate_maps.py`는 다음 경로에 uncertainty map을 저장합니다:
- **기본 경로**: `opt.output_dir/raw/uncert/` (또는 `opt.output_dir/raw/uncert_.../`)
- **파일명 형식**: `%06d_10.png` (인덱스 기반, 예: `000000_10.png`, `000001_10.png`)

### 폴더명 변형:
- 기본: `uncert/`
- Gradient 기반: `uncert_{gref}_{gloss}_layer_{...}/`
- Dropout 기반: `uncert_p_{infer_p}/`

## 4. train.py에서 활용하기 위한 데이터 흐름

### Step 1: generate_maps.py로 Uncertainty Map 생성
```bash
# generate_maps.py 실행하여 uncertainty map 생성
python utils/DepthLoss/generate_maps.py \
    --output_dir /path/to/output \
    --load_weights_folder /path/to/weights \
    ...
# 결과: /path/to/output/raw/uncert/000000_10.png, 000001_10.png, ...
```

### Step 2: train.py 실행 시 confidence 경로 지정
```bash
# --confidence_dir로 generate_maps.py의 출력 경로 지정
python train.py \
    --use_confidence \
    --confidence_dir /path/to/output/raw/uncert \
    ...
```

### Step 3: (선택) 파일 매핑 생성
`generate_maps.py`가 인덱스 기반 파일명을 사용하는 경우, 이미지 이름과 인덱스 매핑이 필요합니다:
```json
// confidence_mapping.json
{
    "image_001.jpg": 0,
    "image_002.jpg": 1,
    ...
}
```
```bash
python train.py \
    --use_confidence \
    --confidence_dir /path/to/output/raw/uncert \
    --confidence_mapping_file confidence_mapping.json \
    ...
```

## 5. 구현 고려사항

### 이미지 크기 일치:
- Confidence map의 크기가 loss 계산에 사용되는 이미지와 일치해야 함
- 필요시 interpolation 사용: `F.interpolate()` 또는 `cv2.resize()`

### 메모리 효율성:
- Confidence map을 미리 계산하여 저장하거나
- 필요할 때만 계산 (캐싱 고려)

### 배치 처리:
- 여러 이미지에 대한 confidence map을 배치로 처리 가능
- Shape: `[B, H, W]` 또는 `[B, 1, H, W]`

## 6. 사용 예시

### 예시 1: generate_maps.py 실행
```bash
# Uncertainty map 생성
python utils/DepthLoss/generate_maps.py \
    --output_dir ./output/uncertainty_maps \
    --load_weights_folder /path/to/monodepth2/weights \
    --eval_mono \
    --data_path /path/to/dataset \
    ...

# 결과: ./output/uncertainty_maps/raw/uncert/000000_10.png, 000001_10.png, ...
```

### 예시 2: train.py에서 confidence 사용
```bash
# 기본 사용 (이미지 이름과 파일명이 일치하는 경우)
python train.py \
    --use_confidence \
    --confidence_dir ./output/uncertainty_maps/raw/uncert \
    --source_path /path/to/dataset \
    ...
```

### 예시 3: 파일 매핑 사용 (인덱스 기반 파일명)
```bash
# 1. 매핑 파일 생성 (이미지 이름 -> 인덱스)
# generate_maps.py의 출력 순서와 dataset의 이미지 순서를 매칭
python create_mapping.py > confidence_mapping.json

# 2. train.py 실행
python train.py \
    --use_confidence \
    --confidence_dir ./output/uncertainty_maps/raw/uncert \
    --confidence_mapping_file confidence_mapping.json \
    ...
```

### 예시 4: Python 코드에서 직접 사용
```python
from utils.loss_utils import l1_loss_confidence, ssim_confidence, load_confidence_map, uncertainty_to_confidence

# generate_maps.py로 생성한 uncertainty map 로드
uncertainty_map = load_confidence_map("path/to/uncert/000000_10.png", normalize=True)

# Confidence로 변환
confidence_map = uncertainty_to_confidence(uncertainty_map, method='inverse').cuda()

# Loss 계산
Ll1 = l1_loss_confidence(image, gt_image, confidence_map)
Lssim = ssim_confidence(image, gt_image, confidence_map)
loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - Lssim))
```

