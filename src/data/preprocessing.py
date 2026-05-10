"""
Script de preprocessing pour FaceForensics++
Détection de visages, alignement, resize
"""

import cv2
import os 
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FacePreprocessor:
    """Classe pour préprocesser les images de visages"""
    
    def __init__(self, target_size=(224, 224)):
        """
        Args:
            target_size: Taille cible des images (width, height)
        """
        self.target_size = target_size
        
        # Charger le détecteur de visages (Haar Cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            logger.error("❌ Impossible de charger le détecteur de visages")
            raise ValueError("Haar Cascade non chargé")
        
        logger.info("✅ Détecteur de visages chargé")
    
    def detect_face(self, image):
        """
        Détecter le visage dans une image
        
        Args:
            image: Image numpy array (BGR)
        
        Returns:
            (x, y, w, h) du visage ou None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Détecter les visages
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Prendre le plus grand visage
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        return largest_face
    
    def crop_face(self, image, face_coords, margin=0.2):
        """
        Recadrer le visage avec une marge
        
        Args:
            image: Image numpy array
            face_coords: (x, y, w, h)
            margin: Marge à ajouter (pourcentage)
        
        Returns:
            Image recadrée
        """
        x, y, w, h = face_coords
        
        # Ajouter une marge
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        return image[y1:y2, x1:x2]
    
    def resize_image(self, image):
        """
        Redimensionner l'image à la taille cible
        
        Args:
            image: Image numpy array
        
        Returns:
            Image redimensionnée
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
    
    def preprocess_image(self, image_path):
        """
        Pipeline complet de preprocessing
        
        Args:
            image_path: Chemin vers l'image
        
        Returns:
            Image preprocessée ou None si échec
        """
        try:
            # Charger l'image
            image = cv2.imread(str(image_path))
            
            if image is None:
                logger.warning(f"⚠️  Impossible de lire : {image_path}")
                return None
            
            # Détecter le visage
            face_coords = self.detect_face(image)
            
            if face_coords is None:
                # Si pas de visage détecté, utiliser l'image entière
                logger.debug(f"Pas de visage détecté dans {image_path.name}, utilisation image entière")
                preprocessed = self.resize_image(image)
            else:
                # Recadrer le visage
                face = self.crop_face(image, face_coords)
                
                # Redimensionner
                preprocessed = self.resize_image(face)
            
            return preprocessed
        
        except Exception as e:
            logger.error(f"❌ Erreur preprocessing {image_path}: {e}")
            return None


def preprocess_dataset(input_dir, output_dir, categories, target_size=(224, 224)):
    """
    Préprocesser tout le dataset
    
    Args:
        input_dir: Dossier contenant les images brutes
        output_dir: Dossier de sortie
        categories: Liste des catégories
        target_size: Taille cible
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialiser le preprocessor
    preprocessor = FacePreprocessor(target_size=target_size)
    
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'no_face': 0
    }
    
    for category in categories:
        logger.info(f"\n📁 Traitement catégorie : {category}")
        
        category_input = input_path / category
        category_output = output_path / category
        category_output.mkdir(parents=True, exist_ok=True)
        
        if not category_input.exists():
            logger.warning(f"⚠️  Catégorie {category} introuvable, skip")
            continue
        
        # Trouver toutes les images
        images = list(category_input.rglob('*.jpg'))
        
        if len(images) == 0:
            logger.warning(f"⚠️  Aucune image dans {category}")
            continue
        
        logger.info(f"   Images trouvées : {len(images)}")
        
        # Traiter chaque image
        for img_path in tqdm(images, desc=f"  {category}"):
            stats['total'] += 1
            
            # Préprocesser
            preprocessed = preprocessor.preprocess_image(img_path)
            
            if preprocessed is not None:
                # Sauvegarder
                # Créer un nom de fichier unique
                relative_path = img_path.relative_to(category_input)
                
                # Créer les sous-dossiers si nécessaire
                unique_name = str(relative_path).replace(os.sep, '_')
                output_file = category_output / unique_name
                
                success = cv2.imwrite(str(output_file), preprocessed)
                
                if success:
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                    logger.warning(f"⚠️  Échec sauvegarde : {output_file}")
            else:
                stats['failed'] += 1
    
    # Résumé
    logger.info("\n" + "="*60)
    logger.info("📊 RÉSUMÉ DU PREPROCESSING")
    logger.info("="*60)
    logger.info(f"Total traité    : {stats['total']:,}")
    logger.info(f"✅ Succès       : {stats['success']:,} ({stats['success']/stats['total']*100:.1f}%)")
    logger.info(f"❌ Échecs       : {stats['failed']:,} ({stats['failed']/stats['total']*100:.1f}%)")
    logger.info("="*60)
    
    return stats


if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "C:/Users/DELL/Desktop/IRM_2/Semestre2/deepfake-detection-system/data/raw/FaceForensics"
    OUTPUT_DIR = "C:/Users/DELL/Desktop/IRM_2/Semestre2/deepfake-detection-system/data/processed/FaceForensics_224"
    CATEGORIES = ['real','deepfakes', 'face2face', 'faceswap', 'neuraltextures']
    TARGET_SIZE = (224, 224)
    
    logger.info("🚀 Début du preprocessing...")
    logger.info(f"Input  : {INPUT_DIR}")
    logger.info(f"Output : {OUTPUT_DIR}")
    logger.info(f"Taille : {TARGET_SIZE}")
    
    # Lancer le preprocessing
    stats = preprocess_dataset(INPUT_DIR, OUTPUT_DIR, CATEGORIES, TARGET_SIZE)
    
    logger.info("\n✅ PREPROCESSING TERMINÉ !")