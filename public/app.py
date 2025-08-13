from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# --- Configuración del Modelo de Embedding ---
EMBEDDING_MODEL_LOCAL = "intfloat/multilingual-e5-large"
embedding_model = None # Se cargará al iniciar la aplicación

def load_model():
    """Carga el modelo de SentenceTransformer al inicio de la aplicación."""
    global embedding_model
    if embedding_model is None:
        print(f"Cargando el modelo de embedding: {EMBEDDING_MODEL_LOCAL}...")
        # Usa 'cpu' si no tienes una GPU o si hay problemas con CUDA.
        # Esto cargará el modelo desde la caché si ya lo descargaste.
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_LOCAL, device='cpu')
        print("Modelo de embedding cargado.")

# Ruta de API para generar embeddings
@app.route('/embed', methods=['POST'])
def embed_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Se requiere un campo 'text' en el cuerpo de la solicitud."}), 400

    text_to_embed = data['text']
    
    try:
        # Genera el embedding
        embedding = embedding_model.encode([text_to_embed]).tolist()[0] # .tolist()[0] para obtener la lista de flotantes
        return jsonify({"embedding": embedding}), 200
    except Exception as e:
        print(f"Error al generar embedding: {e}")
        return jsonify({"error": f"Error interno al generar embedding: {str(e)}"}), 500

# Endpoint para verificar que el servicio está funcionando y el modelo cargado
@app.route('/health', methods=['GET'])
def health_check():
    if embedding_model:
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    else:
        return jsonify({"status": "loading", "model_loaded": False}), 503

if __name__ == '__main__':
    # Carga el modelo cuando la aplicación Flask se inicia
    # Solo en desarrollo, en producción un servidor WSGI (Gunicorn/uWSGI) lo manejaría
    load_model() 
    # Para producción, se suele usar gunicorn: gunicorn -w 4 -b 0.0.0.0:8000 app:app
    app.run(host='0.0.0.0', port=5001)