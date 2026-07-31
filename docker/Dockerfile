FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        chromium \
        curl \
        ffmpeg \
        fonts-freefont-ttf \
        fonts-ipafont-gothic \
        fonts-khmeros \
        fonts-thai-tlwg \
        fonts-wqy-zenhei \
        git \
        gnupg \
        kbd \
        libgl1 \
        libglib2.0-0 \
        libgtk-3-0 \
        libsm6 \
        libx11-6 \
        libxext6 \
        libxkbcommon-x11-0 \
        libxrender1 \
        libxss1 \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /frodobots-earth-rover

# Build context is the repository root (configured in docker-compose.yml).
COPY docker/requirements.txt /tmp/requirements.txt

# Keep the virtual environment outside /frodobots-earth-rover. The development
# bind mount replaces the project directory without hiding dependencies.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHROME_EXECUTABLE_PATH=/usr/bin/chromium \
    PYTHONPATH=/frodobots-earth-rover/earth-rovers-sdk:/frodobots-earth-rover

# This makes the image usable without Compose as well. During development,
# the bind mount in Compose overlays this copy with the live host project.
COPY . /frodobots-earth-rover

EXPOSE 8000 5050 9222

WORKDIR /frodobots-earth-rover

CMD ["sh", "-c", "cd earth-rovers-sdk && exec python3 -m hypercorn main:app --bind 0.0.0.0:8000"]
