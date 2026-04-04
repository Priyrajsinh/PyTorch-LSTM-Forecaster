# ── Stage 1: builder — install dependencies ───────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ── Stage 2: runtime — lean image with non-root user ─────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Create non-root user
RUN adduser --disabled-password --gecos "" appuser

# Copy installed packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy only what the application needs at runtime
COPY src/    src/
COPY models/ models/
COPY config/ config/

# Hand ownership to appuser
RUN chown -R appuser:appuser /app

USER appuser

# Ensure local pip-installed scripts are on PATH
ENV PATH="/home/appuser/.local/bin:${PATH}" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
