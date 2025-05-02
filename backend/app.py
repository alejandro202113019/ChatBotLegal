from flask import Flask, request, jsonify
from flask_cors import CORS
from nlp.processor import process_legal_query
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuración desde variables de entorno
app.config['DEBUG'] = os.getenv('DEBUG', 'False') == 'True'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'una-clave-secreta-muy-segura')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        bot_response = process_legal_query(user_message)
        return jsonify({
            "user_message": user_message,
            "bot_response": bot_response
        })
    except Exception as e:
        app.logger.error(f"Error processing message: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)