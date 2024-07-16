# Use Ubuntu as the base image
FROM python:3.12-slim AS base
RUN apt-get update \
    && apt-get install ffmpeg libsm6 libxext6 poppler-utils tesseract-ocr -y
RUN pip install -U pdm

FROM base AS install
WORKDIR /app/
ENV PDM_CHECK_UPDATE=false
COPY pdm.lock pyproject.toml ./
RUN pdm install --check --prod --no-editable --skip=post_install
ENV PATH="/app/.venv/bin:$PATH"

FROM install AS app
WORKDIR /app/
ARG PORT=8000
EXPOSE $PORT
ENV PATH="/project/.venv/bin:$PATH"
COPY src src
COPY prompts prompts
CMD ["pdm", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "$PORT"]
