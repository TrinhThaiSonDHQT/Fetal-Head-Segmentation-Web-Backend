# API Reference

Complete reference documentation for all API endpoints.

---

## Base URL

```
http://localhost:5000/api
```

---

## Authentication

Currently, no authentication is required. All endpoints are publicly accessible.

---

## Endpoints

### Health Check

Check API and model status.

#### Request

```http
GET /api/health
```

#### Response

**Status:** `200 OK`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

#### Response Fields

| Field          | Type    | Description                                    |
| -------------- | ------- | ---------------------------------------------- |
| `status`       | string  | API status: `"healthy"` or `"error"`           |
| `model_loaded` | boolean | Whether model is loaded successfully           |
| `device`       | string  | Device used for inference: `"cuda"` or `"cpu"` |

#### Example

```bash
curl http://localhost:5000/api/health
```

---

### Image Segmentation

Upload and segment fetal head ultrasound image.

#### Request

```http
POST /api/upload
Content-Type: multipart/form-data
```

#### Request Parameters

| Parameter | Type   | Required | Default  | Description                   |
| --------- | ------ | -------- | -------- | ----------------------------- |
| `image`   | file   | Yes      | -        | Image file (JPEG, PNG, BMP)   |
| `use_tta` | string | No       | `"true"` | Enable Test-Time Augmentation |

#### Response (Success)

**Status:** `200 OK`

```json
{
  "success": true,
  "original": "data:image/png;base64,iVBORw0KGgoAAAA...",
  "segmentation": "data:image/png;base64,iVBORw0KGgoAAAA...",
  "inference_time": 234.56,
  "tta_variance": 0.0234,
  "tta_confidence": 0.9234,
  "is_ultrasound": true,
  "ultrasound_confidence": 0.8765,
  "validation": {
    "is_valid_shape": true,
    "mask_area_ratio": 0.2345,
    "mask_circularity": 0.8234,
    "edge_sharpness": 0.0456
  },
  "warnings": []
}
```

#### Response Fields

| Field                         | Type    | Description                                      |
| ----------------------------- | ------- | ------------------------------------------------ |
| `success`                     | boolean | Whether operation succeeded                      |
| `original`                    | string  | Base64-encoded original image with overlay       |
| `segmentation`                | string  | Base64-encoded segmentation mask                 |
| `inference_time`              | float   | Processing time in milliseconds                  |
| `tta_variance`                | float   | TTA prediction variance (lower = more stable)    |
| `tta_confidence`              | float   | TTA-based confidence score (0-1)                 |
| `is_ultrasound`               | boolean | Whether image appears to be ultrasound           |
| `ultrasound_confidence`       | float   | Confidence that image is ultrasound (0-1)        |
| `validation`                  | object  | Segmentation quality metrics                     |
| `validation.is_valid_shape`   | boolean | Whether mask has valid fetal head shape          |
| `validation.mask_area_ratio`  | float   | Ratio of mask area to image area                 |
| `validation.mask_circularity` | float   | Circularity score (0-1, 1=perfect circle)        |
| `validation.edge_sharpness`   | float   | Edge quality metric                              |
| `warnings`                    | array   | Warning messages (empty if all validations pass) |

#### Response (Error)

**Status:** `400 Bad Request` or `500 Internal Server Error`

```json
{
  "success": false,
  "error": "No image file provided",
  "warnings": []
}
```

#### Error Codes

| Status | Error Message                             | Description                |
| ------ | ----------------------------------------- | -------------------------- |
| 400    | `"No image file provided"`                | Request missing image file |
| 400    | `"No file selected"`                      | Empty filename             |
| 400    | `"Corrupted or invalid image file"`       | Invalid or corrupted image |
| 413    | `"File too large. Maximum size is 16 MB"` | File exceeds size limit    |
| 500    | `"Model inference failed"`                | GPU/processing error       |
| 500    | `"An unexpected error occurred"`          | Unhandled exception        |

#### Examples

**cURL:**

```bash
# With TTA (slower, more accurate)
curl -X POST http://localhost:5000/api/upload \
  -F "image=@ultrasound.jpg" \
  -F "use_tta=true"

# Without TTA (faster)
curl -X POST http://localhost:5000/api/upload \
  -F "image=@ultrasound.jpg" \
  -F "use_tta=false"
```

**Python:**

```python
import requests

# With TTA
url = 'http://localhost:5000/api/upload'
files = {'image': open('ultrasound.jpg', 'rb')}
data = {'use_tta': 'true'}
response = requests.post(url, files=files, data=data)
result = response.json()

# Access results
if result['success']:
    print(f"Inference time: {result['inference_time']} ms")
    print(f"TTA confidence: {result['tta_confidence']}")
    print(f"Valid shape: {result['validation']['is_valid_shape']}")
else:
    print(f"Error: {result['error']}")
```

**JavaScript (Fetch API):**

```javascript
const formData = new FormData();
formData.append('image', imageFile);
formData.append('use_tta', 'true');

const response = await fetch('http://localhost:5000/api/upload', {
  method: 'POST',
  body: formData,
});

const result = await response.json();

if (result.success) {
  console.log('Inference time:', result.inference_time);
  console.log('Original image:', result.original);
  console.log('Segmentation:', result.segmentation);
} else {
  console.error('Error:', result.error);
}
```

---

### Benchmark Performance

Test inference performance on multiple images.

#### Request

```http
GET /api/benchmark?num_images=100&use_tta=false
```

#### Query Parameters

| Parameter    | Type    | Required | Default   | Description                      |
| ------------ | ------- | -------- | --------- | -------------------------------- |
| `num_images` | integer | No       | 100       | Number of test images (max: 500) |
| `use_tta`    | string  | No       | `"false"` | Enable TTA for benchmark         |

#### Response (Success)

**Status:** `200 OK`

```json
{
  "success": true,
  "avg_inference_time": 234.56,
  "total_images": 100,
  "total_time": 23.456,
  "min_time": 187.23,
  "max_time": 345.67,
  "std_dev": 45.23,
  "use_tta": false
}
```

#### Response Fields

| Field                | Type    | Description                      |
| -------------------- | ------- | -------------------------------- |
| `success`            | boolean | Whether benchmark completed      |
| `avg_inference_time` | float   | Average time per image (ms)      |
| `total_images`       | integer | Number of images processed       |
| `total_time`         | float   | Total processing time (seconds)  |
| `min_time`           | float   | Fastest inference time (ms)      |
| `max_time`           | float   | Slowest inference time (ms)      |
| `std_dev`            | float   | Standard deviation of times (ms) |
| `use_tta`            | boolean | Whether TTA was enabled          |

#### Response (Error)

**Status:** `404 Not Found` or `500 Internal Server Error`

```json
{
  "success": false,
  "error": "Benchmark dataset not found"
}
```

#### Examples

**cURL:**

```bash
# Standard benchmark (100 images, no TTA)
curl "http://localhost:5000/api/benchmark"

# Custom benchmark (50 images with TTA)
curl "http://localhost:5000/api/benchmark?num_images=50&use_tta=true"
```

**Python:**

```python
import requests

url = 'http://localhost:5000/api/benchmark'
params = {
    'num_images': 50,
    'use_tta': 'false'
}

response = requests.get(url, params=params)
result = response.json()

if result['success']:
    print(f"Average time: {result['avg_inference_time']:.2f} ms")
    print(f"Min time: {result['min_time']:.2f} ms")
    print(f"Max time: {result['max_time']:.2f} ms")
    print(f"Std dev: {result['std_dev']:.2f} ms")
```

---

## Data Types

### Image Format

**Supported formats:** JPEG, PNG, BMP

**Requirements:**

- Maximum file size: 16 MB
- Recommended size: 256x256 to 1024x1024 pixels
- Grayscale or RGB (automatically converted to grayscale)

### Base64 Image Encoding

Images in responses are encoded as base64 data URLs:

```
data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...
```

**Decoding in JavaScript:**

```javascript
// Create Image element
const img = new Image();
img.src = result.original;
document.body.appendChild(img);
```

**Decoding in Python:**

```python
import base64
from io import BytesIO
from PIL import Image

# Remove data URL prefix
base64_str = result['original'].split(',')[1]

# Decode
image_data = base64.b64decode(base64_str)
image = Image.open(BytesIO(image_data))
image.save('output.png')
```

---

## Rate Limiting

Currently, no rate limiting is implemented.

**Recommendations for production:**

- Implement rate limiting (e.g., 60 requests/minute per IP)
- Use caching for repeated requests
- Monitor resource usage

---

## Error Handling

### Error Response Format

All errors follow this format:

```json
{
  "success": false,
  "error": "Error message describing what went wrong",
  "warnings": []
}
```

### Common HTTP Status Codes

| Code | Meaning               | When It Occurs                     |
| ---- | --------------------- | ---------------------------------- |
| 200  | OK                    | Request successful                 |
| 400  | Bad Request           | Invalid parameters or missing data |
| 404  | Not Found             | Endpoint or resource not found     |
| 413  | Payload Too Large     | File exceeds 16 MB limit           |
| 500  | Internal Server Error | Server-side processing error       |
| 408  | Request Timeout       | Request took too long (>120s)      |

---

## Best Practices

### Image Upload

1. **Validate image client-side** before upload
2. **Resize large images** to 512x512 or smaller
3. **Use JPEG format** for smaller file sizes
4. **Handle errors gracefully** with user-friendly messages

### Performance Optimization

1. **Disable TTA for real-time applications** (4x faster)
2. **Use GPU** when available for inference
3. **Implement client-side caching** for repeated images
4. **Monitor inference times** and adjust TTA accordingly

### Error Handling

1. **Check `success` field** before processing results
2. **Display warning messages** to users
3. **Implement retry logic** for transient errors
4. **Log errors** for debugging

---

## CORS Configuration

### Allowed Origins

**Development:**

```
http://localhost:3000
```

**Production:**
Configure in `app.py`:

```python
CORS(app, resources={r'/api/*': {'origins': 'https://your-domain.com'}})
```

### Allowed Methods

- GET
- POST
- OPTIONS (preflight)

---
