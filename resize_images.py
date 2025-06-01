import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

TARGET_SIZE = (150, 150)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

CLASS_MAP = {
    'COVID': 'COVID/images',
    'NORMAL': 'Normal/images',
    'VIRAL_PNEUMONIA': 'Viral Pneumonia/images'
}

def preprocess_images():
    for output_class, input_rel_path in CLASS_MAP.items():
        in_path = RAW_DIR / input_rel_path
        out_path = PROCESSED_DIR / output_class
        out_path.mkdir(parents=True, exist_ok=True)

        if not in_path.exists():
            print(f"❌ No se encontró la carpeta: {in_path}")
            continue

        print(f"📁 Procesando clase: {output_class}")
        for img_name in tqdm(os.listdir(in_path), desc=output_class):
            img_path = in_path / img_name

            if not img_path.is_file():
                continue
            if not img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                continue

            try:
                # 👇 Para evitar errores con rutas en Windows
                img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"⚠️ No se pudo leer {img_path.name}")
                    continue

                img_resized = cv2.resize(img, TARGET_SIZE)

                # Guardar respetando compatibilidad de escritura en Windows
                out_img_path = out_path / img_name
                _, encoded_img = cv2.imencode('.png', img_resized)
                with open(out_img_path, mode='wb') as f:
                    encoded_img.tofile(f)

            except Exception as e:
                print(f"❌ Error procesando {img_path.name}: {e}")

if __name__ == "__main__":
    preprocess_images()
