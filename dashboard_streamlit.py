import streamlit as st
from classify_tflite import predict
import pandas as pd
import io
from datetime import datetime
import zipfile
import tempfile
from pathlib import Path
import os
import time

st.set_page_config(page_title="COVID Classifier", layout="centered")
st.title("🩻 Clasificador COVID por Rayos X")

st.markdown(
    """
    Este modelo predice si una radiografía de tórax corresponde a:
    - 🦠 **COVID-19**
    - 🌫️ **Neumonía Viral**
    - ✅ **Pulmones Normales**

    Puedes subir una imagen **individual** o un archivo `.zip` con varias imágenes para clasificarlas todas.
    """
)

uploaded_file = st.file_uploader("📤 Sube una imagen o un archivo .zip", type=["png", "jpg", "jpeg", "zip"])

results = []

if uploaded_file:
    if uploaded_file.name.endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, "images.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.read())

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmp_dir)

            image_extensions = [".png", ".jpg", ".jpeg"]
            image_paths = []
            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    if Path(file).suffix.lower() in image_extensions:
                        image_paths.append(Path(root) / file)

            if not image_paths:
                st.error("❌ El archivo .zip no contiene imágenes válidas.")
            else:
                st.success(f"✅ Se encontraron {len(image_paths)} imágenes.")

                for img_path in image_paths:
                    start_time = time.time()
                    label, prob = predict(img_path)
                    elapsed = time.time() - start_time

                    results.append({
                        "Archivo": img_path.name,
                        "Clase predicha": label,
                        "Probabilidad": f"{prob:.2%}",
                        "Tiempo (s)": round(elapsed, 2),
                        "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

    else:
        # Imagen individual
        img_path = "temp.png"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.image(img_path, caption="Imagen cargada", width=300)

        start_time = time.time()
        label, prob = predict(img_path)
        elapsed = time.time() - start_time

        results.append({
            "Archivo": uploaded_file.name,
            "Clase predicha": label,
            "Probabilidad": f"{prob:.2%}",
            "Tiempo (s)": round(elapsed, 2),
            "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

# Mostrar resultados
if results:
    df = pd.DataFrame(results)
    st.markdown("### 📋 Resultados de Clasificación")
    st.dataframe(df, use_container_width=True)

    # Descargar Excel
    towrite = io.BytesIO()
    df.to_excel(towrite, index=False, sheet_name="Resultados")
    towrite.seek(0)
    st.download_button(
        label="📥 Descargar resultados en Excel",
        data=towrite,
        file_name="resultados_clasificacion.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
