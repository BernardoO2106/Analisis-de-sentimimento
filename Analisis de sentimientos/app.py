import streamlit as st
import pandas as pd
import openai
import json
import time

st.title("Análisis de Sentimientos con GPT")
st.write("Carga un archivo Excel con una columna llamada **'Comentario'** para obtener un análisis de sentimiento profundo.")

# 1. Pedir la API Key de OpenAI
api_key = st.text_input("Introduce tu OpenAI API Key (no se almacena):", type="password")

# Widget para subir archivo Excel
uploaded_file = st.file_uploader("Elige un archivo Excel (.xlsx)", type=["xlsx"])

# Opciones de modelo
modelo_opciones = ["gpt-3.5-turbo", "gpt-4"]
modelo_seleccionado = st.selectbox("Selecciona el modelo de OpenAI:", modelo_opciones)

# Botón para iniciar el proceso
if st.button("Analizar Comentarios"):
    if not api_key:
        st.error("Por favor, ingresa tu API Key antes de continuar.")
        st.stop()

    if not uploaded_file:
        st.error("Por favor, sube un archivo Excel antes de continuar.")
        st.stop()

    # Configurar la clave de OpenAI
    openai.api_key = api_key

    # Leer el archivo Excel
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        st.stop()

    # Verificar la columna "Comentario"
    if "Comentario" not in df.columns:
        st.error("La columna 'Comentario' no se encontró en el archivo. Verifica el nombre de la columna.")
        st.stop()

    st.success("Archivo cargado correctamente. Procesando análisis de sentimientos...")

    # Barra de progreso
    progress_bar = st.progress(0)
    total_comentarios = len(df["Comentario"])

    resultados = []
    # Procesar cada comentario
    for i, comentario in enumerate(df["Comentario"]):
        if not isinstance(comentario, str):
            comentario = str(comentario)

        # Construir el prompt para GPT
        mensaje_sistema = (
            "Eres un analista experto en sentimientos y emociones humanas en textos. "
            "Analiza el texto proporcionado y responde con un JSON válido que incluya las siguientes claves: "
            "'sentiment': categoría principal (POS=positivo, NEG=negativo, NEU=neutral), "
            "'intensity': intensidad del sentimiento en escala de 1-5 (1=muy bajo, 5=muy alto), "
            "'emotions': array con las emociones específicas detectadas (ej: alegría, tristeza, ira, etc.), "
            "'key_phrases': array con frases o palabras clave que fundamentan el análisis, "
            "'confidence': nivel de confianza del análisis (0.0-1.0), "
            "'explanation': explicación detallada en español de máximo 2 párrafos."
        )
        mensaje_usuario = f"Texto: {comentario}\nRealiza un análisis completo del sentimiento y proporciona el resultado únicamente en formato JSON."

        try:
            response = openai.ChatCompletion.create(
                model=modelo_seleccionado,
                messages=[
                    {"role": "system", "content": mensaje_sistema},
                    {"role": "user", "content": mensaje_usuario}
                ],
                temperature=0.3  # Un equilibrio entre determinismo y creatividad
            )
            respuesta = response.choices[0].message.content.strip()
            # Intentar parsear la respuesta como JSON
            # GPT-3.5/4 a veces puede agregar texto extra, así que hacemos un approach flexible
            # Buscamos la primera '{' y la última '}' para extraer JSON
            inicio = respuesta.find('{')
            fin = respuesta.rfind('}')
            if inicio != -1 and fin != -1:
                json_str = respuesta[inicio:fin+1]
                data = json.loads(json_str)
                sentiment = data.get("sentiment", "N/A")
                intensity = data.get("intensity", 0)
                emotions = ", ".join(data.get("emotions", []))
                key_phrases = ", ".join(data.get("key_phrases", []))
                confidence = data.get("confidence", 0.0)
                explanation = data.get("explanation", "")
            else:
                # Si no encontramos JSON, fallback
                sentiment = "Desconocido"
                intensity = 0
                emotions = ""
                key_phrases = ""
                confidence = 0.0
                explanation = respuesta

            resultados.append({
                "Comentario": comentario,
                "Sentimiento": sentiment,
                "Intensidad": intensity,
                "Emociones": emotions,
                "Frases_Clave": key_phrases,
                "Confianza": confidence,
                "Explicacion": explanation
            })
        except Exception as e:
            st.warning(f"Error procesando comentario #{i+1}: {e}")
            resultados.append({
                "Comentario": comentario,
                "Sentimiento": "Error",
                "Intensidad": 0,
                "Emociones": "",
                "Frases_Clave": "",
                "Confianza": 0.0,
                "Explicacion": str(e)
            })

        # Actualizar barra de progreso
        progress_bar.progress((i + 1) / total_comentarios)
        
        # Pequeña pausa para no saturar la API
        time.sleep(1)

    # Crear DataFrame con resultados
    df_resultados = pd.DataFrame(resultados)

    # Mostrar resultados detallados
    st.write("### Resultados Detallados")
    st.dataframe(df_resultados)

    # Resumen de conteo por sentimiento
    st.write("### Resumen de Sentimientos")
    resumen_sentimiento = df_resultados["Sentimiento"].value_counts().reset_index()
    resumen_sentimiento.columns = ["Sentimiento", "Cantidad"]
    st.dataframe(resumen_sentimiento)
    
    # Resumen de emociones detectadas
    st.write("### Emociones Detectadas")
    # Crear una lista con todas las emociones (separadas por comas)
    todas_emociones = []
    for emociones in df_resultados["Emociones"]:
        if emociones:
            todas_emociones.extend([e.strip() for e in emociones.split(',')])
    
    if todas_emociones:
        resumen_emociones = pd.DataFrame(
            pd.Series(todas_emociones).value_counts().reset_index()
        )
        resumen_emociones.columns = ["Emoción", "Frecuencia"]
        st.dataframe(resumen_emociones)
    else:
        st.info("No se detectaron emociones específicas en los comentarios analizados.")

    # Promedio de intensidad
    promedio_intensidad = df_resultados["Intensidad"].mean()
    st.write(f"### Intensidad Promedio: {promedio_intensidad:.2f}/5")
    
    # Promedio de confianza
    promedio_confianza = df_resultados["Confianza"].mean()
    st.write(f"### Confianza Promedio del Análisis: {promedio_confianza:.2%}")

    # Exportar a CSV
    csv_data = df_resultados.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar resultados en CSV",
        data=csv_data,
        file_name="resultados_sentimiento_gpt.csv",
        mime="text/csv"
    )
    
    # Exportar a Excel
    excel_buffer = pd.ExcelWriter("resultados_sentimiento_gpt.xlsx", engine="xlsxwriter")
    df_resultados.to_excel(excel_buffer, index=False, sheet_name="Análisis Detallado")
    
    # Agregar resúmenes en pestañas adicionales
    resumen_sentimiento.to_excel(excel_buffer, index=False, sheet_name="Resumen Sentimientos")
    
    if todas_emociones:
        resumen_emociones.to_excel(excel_buffer, index=False, sheet_name="Resumen Emociones")
    
    excel_buffer.save()
    with open("resultados_sentimiento_gpt.xlsx", "rb") as f:
        excel_data = f.read()
    
    st.download_button(
        label="Descargar resultados en Excel",
        data=excel_data,
        file_name="resultados_sentimiento_gpt.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.success("¡Análisis completado!")