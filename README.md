# 🧠 Clasificador de COVID-19 por Radiografías (CASO INNOVACIÓN 2025)

Este proyecto permite clasificar imágenes de rayos X en tres categorías:
- **COVID-19**
- **Neumonía Viral**
- **Pulmones Normales**

Diseñado para funcionar en entornos con recursos limitados (máximo 12GB RAM, sin GPU), utilizando modelos optimizados con TensorFlow Lite.

---

## 📂 Estructura del Proyecto

```
.
├── data/
│   ├── raw/              # Imágenes originales
│   └── processed/        # Imágenes redimensionadas (150x150)
├── model/                # Modelos entrenados (.h5 y .tflite)
├── resize_images.py      # Preprocesamiento
├── train_model.py        # Entrenamiento + conversión TFLite
├── classify_tflite.py    # Clasificación usando modelo .tflite
├── dashboard_streamlit.py # Dashboard web interactivo
├── requirements.txt      # Dependencias
├── README.md
```

---

## ⚙️ Requisitos

- Python 3.10
- RAM ≤ 12GB (sin GPU)
- Tiempo de procesamiento ≤ 18h

---

## 🚀 Instalación

1. Clona el repositorio y crea un entorno virtual:

```bash
python -m venv venv
source venv/bin/activate  # o venv\Scripts\activate en Windows
```

2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Uso

### 1. Redimensionar imágenes:

```bash
python resize_images.py
```

Esto procesa las imágenes de `data/raw/` y guarda las nuevas versiones en `data/processed/`.

---

### 2. Entrenar y exportar el modelo:

```bash
python train_model.py
```

Esto entrenará una MobileNetV2 y generará:

- `model/mobilenet_model.h5`
- `model/mobilenet_model.tflite` (cuantizado para CPU)

---

### 3. Clasificar una imagen (modo consola):

```bash
python classify_tflite.py
```

Este script carga el modelo `.tflite` y realiza inferencia sobre una imagen de prueba.

---

### 4. Dashboard Web (opcional):

```bash
streamlit run dashboard_streamlit.py
```

Te permite subir una imagen y obtener la predicción en una interfaz simple.

---

## 📊 Resultados esperados

- **Precisión:** > 85%
- **RAM usada:** < 4GB en entrenamiento, < 1GB en inferencia
- **Tiempo de procesamiento total:** 6–10 horas para 3.6k imágenes (CPU)

---

## 📈 Innovaciones aplicadas

- MobileNetV2 con `transfer learning`
- Cuantización post-entrenamiento (`int8`)
- Uso de `tf.lite.Interpreter` para CPU
- Sin dependencias de GPU ni frameworks pesados