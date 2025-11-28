# 🔍 PaddleOCR API

API REST performante pour extraire du texte depuis des images avec PaddleOCR.

> **PaddleOCR** est plus précis que Tesseract, surtout pour les documents complexes, le texte manuscrit et les langues asiatiques.

## 🚀 Déploiement rapide sur Coolify

1. Créez un nouveau projet "Public Repository" dans Coolify
2. Entrez l'URL de ce repo
3. Coolify détectera automatiquement le Dockerfile
4. **Important** : Augmentez les ressources (minimum 2GB RAM recommandé)
5. Déployez !

⚠️ **Note** : Le premier démarrage peut prendre 1-2 minutes (chargement des modèles).

## 📡 Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Informations sur l'API |
| GET | `/health` | État de santé du service |
| GET | `/languages` | Langues supportées |
| POST | `/ocr` | **Extraction de texte** |

## 💡 Utilisation

### Avec un fichier image (multipart/form-data)

```bash
curl -X POST \
  -F "image=@mon_document.png" \
  -F "lang=fr" \
  http://votre-url/ocr
```

### Avec une image en Base64 (JSON)

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"image": "BASE64_DE_VOTRE_IMAGE"}' \
  http://votre-url/ocr
```

### Paramètres optionnels

| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| `lang` | Langue OCR (fr, en, ch, german, etc.) | `fr` |

## 📋 Réponse

```json
{
  "success": true,
  "text": "Le texte extrait de l'image...",
  "language": "fr",
  "confidence": 95.2,
  "lines_count": 5,
  "words_count": 42,
  "details": [
    {
      "text": "Première ligne",
      "confidence": 97.5,
      "bbox": {
        "top_left": [10, 20],
        "top_right": [200, 20],
        "bottom_right": [200, 50],
        "bottom_left": [10, 50]
      }
    }
  ]
}
```

## 🌍 Langues supportées

| Code | Langue |
|------|--------|
| `fr` | Français |
| `en` | English |
| `ch` | Chinese (Simplified) |
| `german` | Deutsch |
| `japan` | Japanese |
| `korean` | Korean |
| `es` | Spanish |
| `it` | Italian |
| `pt` | Portuguese |
| `ru` | Russian |
| `ar` | Arabic |
| `latin` | Latin languages |

## 🔧 Variables d'environnement

| Variable | Description | Défaut |
|----------|-------------|--------|
| `PORT` | Port d'écoute | `5000` |
| `OCR_LANG` | Langue par défaut | `fr` |

## 📦 Développement local

```bash
# Construire l'image (peut prendre quelques minutes)
docker build -t paddleocr-api .

# Lancer le container
docker run -p 5000:5000 paddleocr-api

# Tester
curl http://localhost:5000/health
```

## ⚡ Performances

- **Premier appel** : ~2-5 secondes (chargement du modèle en cache)
- **Appels suivants** : ~0.5-2 secondes selon la taille de l'image

## 🔄 Comparaison avec Tesseract

| Critère | Tesseract | PaddleOCR |
|---------|-----------|-----------|
| Vitesse | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Précision | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Texte manuscrit | ⭐⭐ | ⭐⭐⭐⭐ |
| Documents complexes | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Taille image Docker | ~200MB | ~1.5GB |
| RAM requise | ~512MB | ~2GB |

## 📄 Licence

MIT
