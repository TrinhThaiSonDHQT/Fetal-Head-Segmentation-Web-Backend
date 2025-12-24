# Fetal Head Segmentation - Backend API

**Version:** 1.0.0  
**Python:** 3.7+ | **Framework:** Flask 3.x

## Overview

Backend REST API for Fetal Head Segmentation using deep learning. Upload ultrasound images and receive AI-generated segmentation results.

### Key Features

- 🤖 AI-powered segmentation with MobileNetV2-based U-Net (~3.4M parameters)
- ⚡ GPU/CPU support with Test-Time Augmentation
- 📊 Automatic quality validation
- 🛡️ Comprehensive error handling

---

## Quick Links

- **[Quick Reference](QUICK_REFERENCE.md)** - Common commands
- **[API Reference](API_REFERENCE.md)** - Complete endpoint documentation
- **[Architecture](ARCHITECTURE.md)** - Technical details and design

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [API Endpoints](#api-endpoints)
4. [Configuration](#configuration)
5. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.7+
- pip
- (Optional) CUDA-compatible GPU

### Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify model file exists (~13-14 MB)
ls -lh best_model_mobinet_aspp_residual_se_v2.pth
```

---

## Quick Start

### Run Development Server

```bash
python app.py
```

Server will start at: **http://localhost:5000**

### Test API

```bash
# Health check
curl http://localhost:5000/api/health

# Upload image
curl -X POST http://localhost:5000/api/upload \
  -F "image=@ultrasound.jpg"
```

---

## API Endpoints

### 1. Health Check

```
GET /api/health
```

Check API status and model state.

### 2. Image Segmentation

```
POST /api/upload
```

Upload ultrasound image for segmentation.

**Parameters:**

- `image` (file, required): Image file (JPEG, PNG, max 16 MB)
- `use_tta` (boolean, optional): Enable Test-Time Augmentation (default: true)

**Response includes:**

- Base64 encoded images (original + segmentation)
- Inference time and confidence metrics
- Quality validation results
  GET /api/benchmark?num_images=100&use_tta=false
  ``

GET /api/benchmark?num_images=100&use_tta=false

````
Test performance on multiple images.

**📖 See [API_REFERENCE.md](API_REFERENCE.md) for complete documentation.**

---

## Configuration

### Key Settings

**File:** `backend/config.py`
```python
MODEL_PATH = Path('best_model_mobinet_aspp_residual_se_v2.pth')
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
````

\*\*F

**File:** `backend/app.py`

```python
# CORS - Update for production
CORS(app, resources={r'/api/*': {'origins': 'http://localhost:3000'}})
```

### Performance Options

- **TTA enabled:** More accurate, ~4x slower (800-1000ms)
- **TTA disabled:** Faster inference (200-250ms)
- **GPU:** 3-4x faster than CPU (auto-detected)

---

## Troubleshooting

-|
| Model file not found | Verify `best_model_mobinet_aspp_residual_se_v2.pth` exists in backend/ |
| CUDA out of memory | Disable TTA or use CPU mode |
| CORS error | Update `CORS` origins in `app.py` to match frontend URL |
| File too large | Compress image (max: 16 MB) |
| Slow inference | Disable TTA for 4x speed boost |

**📖 See [DEPLOYMENT.md](DEPLOYMENT.md) for production setup and monitoring.**

---

## Project Structure

```

backend/
├── app.py # Flask app entry point
├── config.py # Configuration
├── requirements.txt # Dependencies
├── best*model*\*.pth # Model weights (~13-14 MB)
│
├── core/ # Inference engine
├── model/ # Neural network architecture
├── routes/ # API endpoints
├── services/ # Business logic
├── utils/ # Helper functions
├── middleware/ # Error handling
└── docs/ # Documentation

```

---

## Key Dependencies

- Flask 3.x - Web framework
- PyTorch 2.x - Deep learning
- OpenCV 4.x - Image processing
- Pillow 10.x - Image manipulation
- Albumentations 1.3.x - Data augmentation

---

## License

Educational project for graduation purposes.

---

**Version:** 1.0.0 | **Last Updated:** December 24, 2025

```
backend/
├── app.py                    # Flask app entry point
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
├── best_model_*.pth         # Model weights (~13-14 MB)
│
├── core/                     # Inference engine
├── model/                    # Neural network architecture
├── routes/                   # API endpoints
├── services/                 # Business logic
├── utils/                    # Helper functions
├── middleware/               # Error handling
└── docs/                     # Documentation
```
