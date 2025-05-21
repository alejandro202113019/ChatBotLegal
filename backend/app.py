import re
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from nlp.processor import process_legal_query
from dotenv import load_dotenv
import os
import logging
import json
from datetime import datetime
import uuid

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('legal_assistant_app')

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuración desde variables de entorno
app.config['DEBUG'] = os.getenv('DEBUG', 'False') == 'True'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'una-clave-secreta-muy-segura')

# Soporte para sesiones
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hora

# Historial de conversaciones (en memoria para esta demo)
conversation_history = {}

@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint principal para el chat jurídico."""
    data = request.json
    user_message = data.get('message')
    session_id = data.get('session_id')
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        # Crear o recuperar sesión
        if not session_id:
            session_id = str(uuid.uuid4())
            conversation_history[session_id] = []
        elif session_id not in conversation_history:
            conversation_history[session_id] = []
        
        # Recuperar contexto de conversación
        context = {
            "history": conversation_history[session_id],
            "timestamp": datetime.now().isoformat()
        }
        
        # Procesar la consulta con el contexto
        bot_response = process_legal_query(user_message, context)
        
        # Guardar en historial
        conversation_history[session_id].append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        conversation_history[session_id].append({
            "role": "bot",
            "content": bot_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limitar historial a últimos 10 mensajes
        if len(conversation_history[session_id]) > 20:
            conversation_history[session_id] = conversation_history[session_id][-20:]
        
        # Preparar respuesta
        response = {
            "user_message": user_message,
            "bot_response": bot_response,
            "session_id": session_id
        }
        
        # Logging
        logger.info(f"Session {session_id[:8]}: Q: {user_message[:50]}... A: {bot_response[:50]}...")
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error procesando mensaje: {str(e)}", exc_info=True)
        return jsonify({"error": "Error interno del servidor", "details": str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def feedback():
    """Endpoint para recibir feedback del usuario sobre las respuestas."""
    data = request.json
    feedback_type = data.get('type')  # 'positive', 'negative'
    message_id = data.get('message_id')
    session_id = data.get('session_id')
    comments = data.get('comments', '')
    
    if not all([feedback_type, session_id]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        # Registrar feedback
        feedback_log = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "message_id": message_id,
            "feedback_type": feedback_type,
            "comments": comments
        }
        
        # Guardar feedback en archivo
        try:
            with open('feedback_log.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(feedback_log) + '\n')
        except:
            logger.warning("No se pudo guardar el feedback en archivo")
        
        logger.info(f"Feedback recibido - Tipo: {feedback_type}, Sesión: {session_id[:8]}, Comentarios: {comments[:50]}")
        
        return jsonify({"status": "feedback received"})
    except Exception as e:
        logger.error(f"Error procesando feedback: {str(e)}")
        return jsonify({"error": "Error interno del servidor"}), 500

@app.route('/api/context', methods=['POST'])
def set_context():
    """Endpoint para establecer contexto adicional para la conversación."""
    data = request.json
    session_id = data.get('session_id')
    context_data = data.get('context', {})
    
    if not session_id:
        return jsonify({"error": "No session ID provided"}), 400
    
    try:
        # Crear o actualizar contexto para esta sesión
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        
        # Almacenar metadata de contexto
        if session_id not in session.keys():
            session[session_id] = {}
        
        session[session_id]['context'] = context_data
        
        logger.info(f"Contexto actualizado para sesión {session_id[:8]}")
        return jsonify({"status": "context updated"})
    except Exception as e:
        logger.error(f"Error estableciendo contexto: {str(e)}")
        return jsonify({"error": "Error interno del servidor"}), 500

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """Endpoint para limpiar el historial de conversación."""
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({"error": "No session ID provided"}), 400
    
    try:
        # Limpiar historial si existe
        if session_id in conversation_history:
            conversation_history[session_id] = []
        
        logger.info(f"Historial limpiado para sesión {session_id[:8]}")
        return jsonify({"status": "history cleared"})
    except Exception as e:
        logger.error(f"Error limpiando historial: {str(e)}")
        return jsonify({"error": "Error interno del servidor"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint para verificar la salud del sistema."""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/articles', methods=['GET'])
def get_relevant_articles():
    """Endpoint para obtener artículos del CPP relevantes a palabras clave."""
    keywords = request.args.get('keywords', '')
    
    if not keywords:
        return jsonify({"error": "No keywords provided"}), 400
    
    try:
        # Cargar base de conocimiento
        current_dir = os.path.dirname(os.path.abspath(__file__))
        legal_db_path = os.path.join(current_dir, 'data', 'legal_db.json')
        
        with open(legal_db_path, 'r', encoding='utf-8') as f:
            legal_kb = json.load(f)
        
        # Filtrar artículos relevantes
        keywords_list = keywords.lower().split(',')
        
        relevant_articles = []
        for item in legal_kb:
            if not isinstance(item, dict) or 'question' not in item or 'answer' not in item:
                continue
                
            # Buscar coincidencias en la respuesta
            matches = 0
            for kw in keywords_list:
                if kw.strip() and kw.strip() in item['answer'].lower():
                    matches += 1
            
            # Si hay al menos una coincidencia, incluir en resultados
            if matches > 0:
                # Extraer número de artículo si está presente
                article_match = re.search(r'Artículo\s+(\d+[o°\.]*)', item['answer'])
                article_num = article_match.group(1) if article_match else "N/A"
                
                relevant_articles.append({
                    "article_num": article_num,
                    "question": item['question'],
                    "answer": item['answer'],
                    "relevance": matches / len(keywords_list)
                })
        
        # Ordenar por relevancia
        relevant_articles.sort(key=lambda x: x['relevance'], reverse=True)
        
        return jsonify({
            "keywords": keywords_list,
            "results": relevant_articles[:10]  # Limitar a 10 resultados
        })
    except Exception as e:
        logger.error(f"Error buscando artículos relevantes: {str(e)}")
        return jsonify({"error": "Error interno del servidor"}), 500

# Manejadores de errores
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint no encontrado"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Método no permitido"}), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Error interno del servidor: {str(error)}")
    return jsonify({"error": "Error interno del servidor"}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)