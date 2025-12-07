FROM python:3.10-slim-bookworm

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# Copie des fichiers requirements d'abord (pour le cache Docker)
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie de l'application
COPY app.py .

# Variables d'environnement
ENV PORT=5000
ENV OCR_LANG=fr

# Exposition du port
EXPOSE 5000

# Healthcheck (start-period élevé car le premier chargement du modèle prend du temps)
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Démarrage avec Gunicorn
# Timeout élevé car le premier chargement du modèle est long
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "300", "--preload", "app:app"]
