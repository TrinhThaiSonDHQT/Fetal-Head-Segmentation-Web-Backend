# Backend Architecture Documentation

Detailed technical documentation of the backend system architecture.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Component Breakdown](#component-breakdown)
4. [Data Flow](#data-flow)
5. [Model Architecture](#model-architecture)
6. [Inference Pipeline](#inference-pipeline)
7. [Error Handling Strategy](#error-handling-strategy)
8. [Design Patterns](#design-patterns)
9. [Performance Considerations](#performance-considerations)

---

## Overview

The backend is built as a **modular Flask REST API** following separation of concerns principles. Each component has a specific responsibility, making the codebase maintainable, testable, and scalable.

### Key Design Principles

- **Modularity:** Clear separation between routes, services, core logic, and utilities
- **Single Responsibility:** Each module handles one specific concern
- **Dependency Injection:** Services injected into Flask app config
- **Error Isolation:** Comprehensive error handling at each layer
- **Stateless Design:** No session state between requests

---

## System Architecture

### High-Level Architecture

```
┌────────────────────────────────────────────────────────────┐
│                        Client Layer                         │
│                    (React Frontend)                         │
└─────────────────────┬──────────────────────────────────────┘
                      │ HTTP/REST API
┌─────────────────────┴──────────────────────────────────────┐
│                    Flask Application                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Routes Layer (Blueprints)               │  │
│  │  - health_bp    - segmentation_bp    - benchmark_bp │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────┴───────────────────────────────┐  │
│  │              Services Layer                          │  │
│  │              SegmentationService                     │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────┴───────────────────────────────┐  │
│  │              Core Engine Layer                       │  │
│  │  - ModelLoader        - InferenceEngine             │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────┴───────────────────────────────┐  │
│  │              PyTorch Model Layer                     │  │
│  │       MobileNetV2ASPPResidualSEUNet                 │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

1. **Routes Layer:** HTTP request/response handling, validation
2. **Services Layer:** Business logic orchestration
3. **Core Engine Layer:** Model loading, inference pipeline
4. **Model Layer:** Neural network architecture

---

## Component Breakdown

### 1. Flask Application (`app.py`)

**Purpose:** Application entry point and initialization.

**Responsibilities:**

- Initialize Flask app
- Configure CORS
- Register blueprints
- Initialize segmentation service
- Register error handlers
- Start development server

**Key Code:**

```python
app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
app.config['SECRET_KEY'] = 'dev-secret-key'

# CORS
CORS(app, resources={r'/api/*': {'origins': 'http://localhost:3000'}})

# Initialize service
segmentation_service = SegmentationService(model_path)
segmentation_service.initialize_model()
app.config['SEGMENTATION_SERVICE'] = segmentation_service

# Register blueprints
app.register_blueprint(health_bp, url_prefix='/api')
app.register_blueprint(segmentation_bp, url_prefix='/api')
app.register_blueprint(benchmark_bp, url_prefix='/api')
```

---

### 2. Routes (`routes/`)

**Purpose:** Define API endpoints and handle HTTP requests.

#### Health Route (`routes/health.py`)

```python
@health_bp.route('/health', methods=['GET'])
def health_check():
    segmentation_service = current_app.config['SEGMENTATION_SERVICE']
    model_status = segmentation_service.is_model_loaded()
    device_info = segmentation_service.get_device_info()

    return jsonify({
        'status': 'healthy',
        'model_loaded': model_status,
        'device': device_info
    })
```

#### Segmentation Route (`routes/segmentation.py`)

**Flow:**

1. Validate request (check for image file)
2. Parse TTA flag
3. Call service to process image
4. Return JSON response

**Error Handling:**

- `400` - Validation errors (no file, invalid format)
- `500` - Processing errors (inference failed)

#### Benchmark Route (`routes/benchmark.py`)

**Flow:**

1. Parse query parameters
2. Call service benchmark method
3. Return performance statistics

---

### 3. Services (`services/`)

**Purpose:** Business logic orchestration.

#### SegmentationService (`services/segmentation_service.py`)

**Responsibilities:**

- Initialize model (lazy loading)
- Validate uploaded images
- Process images through inference engine
- Format results for API response

**Key Methods:**

```python
def initialize_model(self):
    """Load model and inference engine."""
    if self.model_loader is None:
        self.model_loader = ModelLoader(self.model_path)
        self.inference_engine = InferenceEngine(
            self.model_loader.model,
            self.model_loader.device
        )

def process_uploaded_image(self, image_file, use_tta=True):
    """Process uploaded image and return segmentation results."""
    # 1. Validate image
    image = Image.open(image_file.stream)
    image.verify()

    # 2. Convert to numpy
    image_np = pil_to_numpy(image)

    # 3. Run inference
    result = self.inference_engine.process_image(image_np, use_tta)

    # 4. Convert results to base64
    result['original'] = image_to_base64(result['original_with_overlay'])
    result['segmentation'] = image_to_base64(result['mask'])

    return result
```

---

### 4. Core Engine (`core/`)

**Purpose:** Model loading and inference pipeline.

#### ModelLoader (`core/model_loader.py`)

**Responsibilities:**

- Detect device (CUDA/CPU)
- Load model architecture
- Load trained weights
- Set model to evaluation mode

**Key Code:**

```python
def _detect_device(self):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("✓ No GPU detected, using CPU")
    return device

def _load_model(self):
    # Load checkpoint
    checkpoint = torch.load(self.model_path, map_location=self.device)

    # Create model
    model = MobileNetV2ASPPResidualSEUNet(...)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to eval mode
    model.eval()
    model.to(self.device)

    return model
```

#### InferenceEngine (`core/inference.py`)

**Responsibilities:**

- Preprocess images
- Run inference (with optional TTA)
- Post-process predictions
- Validate results

**Pipeline:**

```python
def process_image(self, image, use_tta=True):
    # 1. Validate ultrasound
    is_us, us_conf = self.us_detector.is_ultrasound(image)

    # 2. Preprocess
    input_tensor, original_size = self.preprocess(image)

    # 3. Run inference (with TTA if enabled)
    if use_tta:
        mask, tta_variance, tta_conf = self.infer_with_tta(input_tensor)
    else:
        mask = self.infer(input_tensor)

    # 4. Post-process
    mask = self.postprocess(mask, original_size)

    # 5. Validate quality
    quality = self.quality_checker.analyze_mask(mask)

    # 6. Create visualization
    overlay = create_overlay(image, mask)

    return {
        'mask': mask,
        'original_with_overlay': overlay,
        'inference_time': inference_time,
        'is_ultrasound': is_us,
        'ultrasound_confidence': us_conf,
        'validation': quality,
        ...
    }
```

---

### 5. Model Architecture (`model/`)

**Purpose:** Define neural network architecture.

#### MobileNetV2ASPPResidualSEUNet (`model/mobinet_aspp_residual_se.py`)

**Components:**

1. **Encoder (MobileNetV2):**

   - Pre-trained on ImageNet
   - Frozen weights (transfer learning)
   - Extracts features at 5 scales

2. **ASPP Bottleneck:**

   - Multi-scale context aggregation
   - Atrous convolutions at rates [6, 12, 18]
   - Global pooling branch

3. **Decoder:**

   - 4 upsampling stages
   - Residual blocks with SE attention
   - Skip connections from encoder

4. **SE Blocks:**
   - Channel-wise attention
   - Applied to skip connections
   - Reduction ratio: 16

**Architecture Diagram:**

```
Input (256x256x1)
      ↓
┌─────────────────┐
│  MobileNetV2    │ ← Encoder (Frozen)
│  Encoder        │
└────┬────────────┘
     │
     ├───→ Skip1 (256x256x32) ──→ SE ──┐
     ├───→ Skip2 (128x128x16) ──→ SE ──┤
     ├───→ Skip3 (64x64x24)   ──→ SE ──┤
     ├───→ Skip4 (32x32x32)   ──→ SE ──┤
     └───→ Skip5 (16x16x96)   ──→ SE ──┤
           ↓                            │
      ┌───────────┐                     │
      │   ASPP    │ ← Bottleneck        │
      │ (8x8x320) │                     │
      └─────┬─────┘                     │
            ↓                           │
      ┌─────────────┐                   │
      │  Decoder    │ ← Upsampling      │
      │  Stage 1    │ ←─────────────────┘
      └──────┬──────┘   (Skip Concat)
             ↓
      ┌─────────────┐
      │  Decoder    │
      │  Stage 2    │
      └──────┬──────┘
             ↓
      ┌─────────────┐
      │  Decoder    │
      │  Stage 3    │
      └──────┬──────┘
             ↓
      ┌─────────────┐
      │  Decoder    │
      │  Stage 4    │
      └──────┬──────┘
             ↓
      ┌─────────────┐
      │ Final Conv  │
      └──────┬──────┘
             ↓
    Output (256x256x1)
```

---

### 6. Utilities (`utils/`)

**Purpose:** Helper functions and validation modules.

#### image_utils.py

- `numpy_to_pil()` - Convert numpy array to PIL Image
- `pil_to_numpy()` - Convert PIL Image to numpy
- `create_overlay()` - Draw segmentation boundary on original image
- `image_to_base64()` - Encode image to base64 string

#### quality_checker.py

**Validates segmentation quality:**

```python
def analyze_mask(self, mask):
    area_ratio = self._calculate_area_ratio(mask)
    circularity = self._calculate_circularity(mask)
    edge_sharpness = self._calculate_edge_sharpness(mask)
    is_valid = self._is_valid_fetal_head_shape(...)

    return {
        'mask_area_ratio': area_ratio,      # 0.05 - 0.60
        'mask_circularity': circularity,    # > 0.60
        'edge_sharpness': edge_sharpness,   # > 0.002
        'is_valid_shape': is_valid
    }
```

#### ultrasound_detector.py

**Detects if image is ultrasound:**

```python
def is_ultrasound(self, image):
    features = {
        'dark_mid_ratio': ...,      # Predominantly dark/mid tones
        'contrast': ...,            # Moderate contrast
        'edge_density': ...,        # Specific edge patterns
        'corner_darkness': ...,     # Dark corners (cone shape)
        'texture_score': ...        # Speckle noise pattern
    }

    confidence = self._compute_confidence(features)
    is_us = confidence >= 0.5

    return is_us, confidence
```

---

### 7. Middleware (`middleware/`)

**Purpose:** Centralized error handling.

#### error_handlers.py

```python
def register_error_handlers(app):
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            'success': False,
            'error': 'File too large. Maximum size is 16 MB.'
        }), 413

    # ... more handlers
```

---

## Data Flow

### Segmentation Request Flow

```
1. Client uploads image
         ↓
2. Flask receives POST /api/upload
         ↓
3. segmentation_bp validates request
         ↓
4. SegmentationService.process_uploaded_image()
         ↓
5. Image validation (PIL Image.verify())
         ↓
6. Convert to numpy array
         ↓
7. InferenceEngine.process_image()
         ↓
8. Ultrasound validation
         ↓
9. Preprocess (resize, normalize)
         ↓
10. Model inference (with TTA)
         ↓
11. Post-process (resize to original size)
         ↓
12. Quality validation
         ↓
13. Create overlay visualization
         ↓
14. Convert to base64
         ↓
15. Return JSON response
         ↓
16. Client receives results
```

---

## Model Architecture

### MobileNetV2 Encoder

**Feature Extraction Points:**

| Block        | Output Channels | Resolution  | Use   |
| ------------ | --------------- | ----------- | ----- |
| features[1]  | 16              | H/2 × W/2   | Skip1 |
| features[3]  | 24              | H/4 × W/4   | Skip2 |
| features[6]  | 32              | H/8 × W/8   | Skip3 |
| features[13] | 96              | H/16 × W/16 | Skip4 |
| features[18] | 1280            | H/32 × W/32 | Skip5 |

### ASPP Module

**Branches:**

1. **1×1 Convolution** - Captures point-wise features
2. **3×3 Conv, rate=6** - Small receptive field
3. **3×3 Conv, rate=12** - Medium receptive field
4. **3×3 Conv, rate=18** - Large receptive field
5. **Global Average Pooling** - Image-level context

All branches concatenated → 1×1 conv → Output

### Decoder

**4 Upsampling Stages:**

Each stage:

1. Upsample (bilinear interpolation, 2×)
2. Concatenate with skip connection (with SE)
3. ResidualBlockSE (2× conv blocks)
4. Batch normalization + ReLU

### SE Block

**Squeeze-and-Excitation:**

```
Input
  ↓
Global Average Pooling (Squeeze)
  ↓
FC → ReLU (Excite)
  ↓
FC → Sigmoid (Scale)
  ↓
Channel-wise multiplication
  ↓
Output
```

**Reduction ratio:** 16

---

## Inference Pipeline

### Standard Inference (No TTA)

```python
def infer(self, input_tensor):
    with torch.no_grad():
        output = self.model(input_tensor)
        mask = torch.sigmoid(output)
        mask = (mask > 0.5).float()
    return mask
```

**Time:** ~200-250ms (GPU)

### Test-Time Augmentation (TTA)

```python
def infer_with_tta(self, input_tensor):
    predictions = []

    for transform in self.tta_transforms:
        # Apply forward transform
        augmented = transform['forward'](input_tensor)

        # Predict
        output = self.model(augmented)
        mask = torch.sigmoid(output)

        # Reverse transform
        mask = transform['reverse'](mask)

        predictions.append(mask)

    # Average predictions
    final_mask = torch.mean(torch.stack(predictions), dim=0)

    # Calculate variance (stability metric)
    variance = torch.var(torch.stack(predictions), dim=0).mean()
    confidence = 1 - variance

    # Threshold
    final_mask = (final_mask > 0.5).float()

    return final_mask, variance, confidence
```

**TTA Transforms:**

1. Original
2. Horizontal flip
3. Vertical flip
4. Both flips

**Time:** ~800-1000ms (GPU)

---

## Error Handling Strategy

### Layer-Based Error Handling

**1. Route Layer:**

- Validate HTTP request format
- Check for required fields
- Return 400 for invalid requests

**2. Service Layer:**

- Validate image format
- Handle PIL errors (corrupted files)
- Catch and wrap inference errors

**3. Core Layer:**

- Handle CUDA out of memory
- Handle model inference failures
- Validate tensor shapes

**4. Middleware:**

- Catch all unhandled exceptions
- Return generic 500 errors
- Log errors for debugging

### Error Response Format

```json
{
  "success": false,
  "error": "User-friendly error message",
  "warnings": []
}
```

## Performance Considerations

### Optimization Techniques

1. **Frozen Encoder:**

   - MobileNetV2 weights frozen
   - Reduces computation
   - Maintains pre-trained features

2. **Efficient Architecture:**

   - MobileNetV2 uses depthwise separable convolutions
   - ~3.4M parameters (vs ~30M for ResNet50)
   - Faster inference

3. **GPU Acceleration:**

   - Automatic CUDA detection
   - All tensors moved to GPU
   - Parallel computation

4. **Batch Normalization:**

   - Faster convergence
   - Better generalization
   - Reduced internal covariate shift

5. **GroupNorm in ASPP:**
   - Better than BatchNorm for small batches
   - Inference uses batch size = 1

---
