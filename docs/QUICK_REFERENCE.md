# Quick Reference Guide

Fast reference for common tasks and commands.

---

## Quick Start

### Run Development Server

```bash
cd backend
python app.py
```

**Server:** http://localhost:5000  
**Health Check:** http://localhost:5000/api/health

---

## API Endpoints

### Health Check

```bash
curl http://localhost:5000/api/health
```

### Upload Image

```bash
curl -X POST http://localhost:5000/api/upload \
  -F "image=@ultrasound.jpg" \
  -F "use_tta=true"
```

### Benchmark

```bash
curl "http://localhost:5000/api/benchmark?num_images=50"
```

---

## Common Commands

### Virtual Environment

**Create:**

```bash
python -m venv venv
```

**Activate:**

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

**Deactivate:**

```bash
deactivate
```

### Dependencies

**Install:**

```bash
pip install -r requirements.txt
```

**Update:**

```bash
pip install --upgrade -r requirements.txt
```

---

## Configuration

### CORS Origins

**Development:**

```python
CORS(app, resources={r'/api/*': {'origins': 'http://localhost:3000'}})
```

### Max Upload Size

**Default:** 16 MB

**Change in app.py:**

```python
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB
```

---

## Performance Tips

### Speed Up Inference

1. **Disable TTA:** `use_tta=false` (4× faster)
2. **Use GPU:** Automatic if available (3-4× faster)
3. **Reduce workers:** Lower memory usage
4. **Increase worker timeout:** For large images

### Memory Optimization

1. **Limit max workers:** Prevent OOM
2. **Clear cache:** `torch.cuda.empty_cache()`
3. **Use CPU:** If GPU memory limited
4. **Reduce image size:** Resize before upload

---

## Error Codes

| Code | Meaning | Solution |
| 400 | Bad Request | Check request format |
| 404 | Not Found | Check endpoint URL |
| 413 | File Too Large | Reduce image size |
| 500 | Server Error | Check logs |
| 408 | Timeout | Increase timeout |

---

## API Response Fields

### Segmentation Response

```json
{
  "success": true,
  "original": "base64...",
  "segmentation": "base64...",
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

---

## Resources

- **Main README:** [README.md](README.md)
- **API Reference:** [API_REFERENCE.md](API_REFERENCE.md)
- **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)

---
