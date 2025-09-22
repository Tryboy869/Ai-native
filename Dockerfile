
# =====================================
# DOCKERFILE - IA WEBAPP COMPLÈTE
# =====================================
# Build: docker build -t ia-native .
# Run: docker run -p 8000:8000 ia-native

FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Anzize Daouda"
LABEL description="IA Native 800 lignes + Stack Perplexity"
LABEL version="1.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système (minimal)
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de dépendances
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Installation des modèles spaCy
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download fr_core_news_sm || echo "Modèle français optionnel non installé"

# Copie du code source
COPY app.py .

# Création d'un utilisateur non-root pour la sécurité
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Exposition du port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Commande de démarrage
CMD ["python", "app.py"]

# =====================================
# INSTRUCTIONS DE DÉPLOIEMENT
# =====================================
# 
# 1. BUILD LOCAL:
#    docker build -t ia-native .
#    docker run -p 8000:8000 ia-native
#    
# 2. PUSH VERS REGISTRY:
#    docker tag ia-native your-registry/ia-native:latest
#    docker push your-registry/ia-native:latest
#    
# 3. DÉPLOIEMENT CLOUD:
#    - AWS ECS/Fargate
#    - Google Cloud Run  
#    - Azure Container Instances
#    - Kubernetes
#    
# 4. VARIABLES D'ENVIRONNEMENT OPTIONNELLES:
#    docker run -e PORT=5000 -p 5000:5000 ia-native
#    
# TAILLE IMAGE ESTIMÉE: ~500MB
# (Python:3.11-slim ~150MB + Dépendances ~200MB + Modèles spaCy ~150MB)
# 
# VS Framework traditionnel: 2-5GB+ 
# = 4-10x plus léger ! 🚀