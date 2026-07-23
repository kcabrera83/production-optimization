# Deployment Guide - Production Optimization

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python train.py

EXPOSE 5014

CMD ["python", "app.py"]
```

### Build and Run

```bash
docker build -t production-optimization .
docker run -p 5014:5014 production-optimization
```

## Docker Compose

```yaml
version: '3.8'
services:
  production-optimization:
    build: .
    ports:
      - "5014:5014"
    environment:
      - FLASK_ENV=production
    volumes:
      - model-data:/app/outputs

volumes:
  model-data:
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| FLASK_ENV | Flask environment mode | development |
| PORT | Server port | 5014 |

## Production Considerations

- Use gunicorn for production serving:
  ```bash
  gunicorn -w 4 -b 0.0.0.0:5014 app:app
  ```
- Set `debug=False` in `app.py` (already set)
- Configure reverse proxy (nginx) for SSL termination
- Set up health check monitoring on `/api/health`
- Use a process manager (systemd, supervisor) for auto-restart
- Models serialized with joblib for efficient loading

## Training Pipeline

1. `python train.py` generates synthetic field data
2. Separate preprocessors fitted for each target
3. Models trained with cross-validation
4. Artifacts saved to `outputs/models/`:
   - `field_optimizer.joblib` - Net profit predictor
   - `allocation_model.joblib` - Efficiency predictor
   - `preprocessor_opt.joblib` - Profit preprocessor
   - `preprocessor_alloc.joblib` - Efficiency preprocessor
   - `training_results.json` - CV scores summary

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):
- Runs on push to main
- Installs dependencies
- Runs training pipeline
- Executes API tests (pytest)

---

*Elaborado por Ing. Kelvin Cabrera*
