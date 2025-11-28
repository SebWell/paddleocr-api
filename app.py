"""
API OCR avec PaddleOCR
Endpoint simple pour extraire du texte depuis des images
Plus performant que Tesseract pour les documents complexes
"""

from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import io
import base64
import os

app = Flask(__name__)

# Initialisation de PaddleOCR (français)
# use_angle_cls: détection de l'orientation du texte
# lang: 'fr' pour français, 'en' pour anglais
DEFAULT_LANG = os.getenv("OCR_LANG", "fr")

# Cache des instances OCR par langue
ocr_instances = {}

def get_ocr_instance(lang):
    """Récupère ou crée une instance PaddleOCR pour la langue demandée"""
    if lang not in ocr_instances:
        ocr_instances[lang] = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=False,
            show_log=False
        )
    return ocr_instances[lang]

# Pré-charger l'instance par défaut au démarrage
print(f"Chargement du modèle PaddleOCR ({DEFAULT_LANG})...")
get_ocr_instance(DEFAULT_LANG)
print("Modèle chargé !")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "PaddleOCR API",
        "version": "1.0.0",
        "endpoints": {
            "/ocr": "POST - Envoyer une image pour extraction de texte",
            "/health": "GET - Vérifier l'état du service",
            "/languages": "GET - Lister les langues supportées"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "engine": "paddleocr"})


@app.route("/languages", methods=["GET"])
def languages():
    """Liste les langues supportées par PaddleOCR"""
    supported = {
        "fr": "Français",
        "en": "English",
        "ch": "Chinese (Simplified)",
        "german": "Deutsch",
        "japan": "Japanese",
        "korean": "Korean",
        "es": "Spanish",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "ar": "Arabic",
        "latin": "Latin languages"
    }
    return jsonify({
        "default": DEFAULT_LANG,
        "supported": supported
    })


@app.route("/ocr", methods=["POST"])
def ocr():
    """
    Extrait le texte d'une image avec PaddleOCR
    
    Accepte:
    - multipart/form-data avec fichier 'image'
    - JSON avec 'image' en base64
    
    Paramètres optionnels:
    - lang: langue pour l'OCR (défaut: fr)
    - det: activer la détection de texte (défaut: true)
    - rec: activer la reconnaissance de texte (défaut: true)
    """
    try:
        # Récupérer les paramètres
        lang = request.form.get("lang") or request.args.get("lang") or DEFAULT_LANG
        
        # Récupérer l'image
        image = None
        
        # Option 1: Fichier uploadé
        if "image" in request.files:
            file = request.files["image"]
            image = Image.open(file.stream).convert("RGB")
        
        # Option 2: Base64 dans JSON
        elif request.is_json and "image" in request.json:
            image_data = request.json["image"]
            # Retirer le préfixe data:image si présent
            if "," in image_data:
                image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        else:
            return jsonify({
                "error": "Aucune image fournie",
                "usage": "Envoyez une image via 'image' (multipart) ou en base64 (JSON)"
            }), 400
        
        # Convertir en numpy array pour PaddleOCR
        img_array = np.array(image)
        
        # Obtenir l'instance OCR
        ocr_engine = get_ocr_instance(lang)
        
        # Extraction du texte
        result = ocr_engine.ocr(img_array, cls=True)
        
        # Parser les résultats
        lines = []
        total_confidence = 0
        word_count = 0
        
        if result and result[0]:
            for line in result[0]:
                bbox = line[0]  # Coordonnées du texte
                text = line[1][0]  # Texte extrait
                confidence = line[1][1]  # Score de confiance
                
                lines.append({
                    "text": text,
                    "confidence": round(confidence * 100, 2),
                    "bbox": {
                        "top_left": bbox[0],
                        "top_right": bbox[1],
                        "bottom_right": bbox[2],
                        "bottom_left": bbox[3]
                    }
                })
                
                total_confidence += confidence
                word_count += len(text.split())
        
        # Texte complet
        full_text = "\n".join([l["text"] for l in lines])
        avg_confidence = (total_confidence / len(lines) * 100) if lines else 0
        
        return jsonify({
            "success": True,
            "text": full_text,
            "language": lang,
            "confidence": round(avg_confidence, 2),
            "lines_count": len(lines),
            "words_count": word_count,
            "details": lines  # Détails par ligne avec positions
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
