# Railway Deployment Guide

## Why Railway?

- **8 GB RAM** (vs Render's 512 MB)
- **$5 free credit/month** (enough for 100-500 hours)
- **Better for ML models** with PyTorch

## Deployment Steps

### 1. Sign up for Railway

1. Go to https://railway.app/
2. Sign up with GitHub
3. Connect your repository

### 2. Create New Project

1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose: `TrinhThaiSonDHQT/Fetal-Head-Segmentation-Web-Backend`
4. Railway will auto-detect Python and deploy

### 3. Configure Environment Variables (Optional)

In Railway dashboard → Variables tab:

```bash
FLASK_ENV=production
PYTHONUNBUFFERED=1

# To disable TTA for faster inference (optional):
# DISABLE_TTA=1

# To use quantized model (optional):
# MODEL_PATH=best_model_mobinet_aspp_residual_se_v2_quantized.pth
```

### 4. Set Port

Railway auto-detects the port from Gunicorn. No changes needed.

### 5. Deploy

1. Railway automatically builds and deploys
2. Get your public URL from Railway dashboard
3. Test: `https://your-app.railway.app/health`

## Cost Estimate

- **Free tier**: $5 credit/month
- **Usage**: ~$0.01-0.05/hour depending on traffic
- **Monthly cost**: $3-10 for moderate use

## Frontend Configuration

Update frontend API URL to Railway:

```typescript
// frontend/src/store/api/segmentationApi.ts
const BASE_URL = 'https://your-app.railway.app';
```

## Comparison: Railway vs Render

| Feature    | Railway         | Render       |
| ---------- | --------------- | ------------ |
| RAM        | 8 GB ✅         | 512 MB ❌    |
| Free tier  | $5 credit/month | Limited      |
| ML support | Excellent       | Poor         |
| Your model | Works ✅        | OOM error ❌ |

## Troubleshooting

**If deployment fails:**

1. Check Railway logs: Dashboard → Deployments → Logs
2. Verify `requirements.txt` is correct
3. Ensure `railway.toml` is in backend folder

**If model loads slowly:**

- Railway downloads model on first start
- Takes 1-2 minutes for 272 MB model
- Subsequent requests are fast

## Alternative: Use Quantized Model on Railway

If you want faster startup and lower costs:

```bash
# Set environment variable in Railway:
MODEL_PATH=best_model_mobinet_aspp_residual_se_v2_quantized.pth
```

This reduces:

- Model size: 272 MB → 97 MB
- Startup time: 2 min → 30 sec
- Memory usage: ~400 MB → ~200 MB
