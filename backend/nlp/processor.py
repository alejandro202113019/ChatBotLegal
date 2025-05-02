import json
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from pathlib import Path

# Descargar recursos de NLTK (solo primera vez)
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Base de conocimiento por defecto (fallback)
DEFAULT_KB = [
    {
        "question": "¿Qué es un contrato?",
        "answer": "Un contrato es un acuerdo legal entre dos o más partes que crea obligaciones exigibles. Para que sea válido generalmente requiere: 1) Consentimiento, 2) Objeto lícito, 3) Causa lícita y 4) Capacidad legal de las partes. Los contratos pueden ser orales o escritos, aunque algunos requieren forma específica por ley."
    },
    {
        "question": "¿Cómo se realiza un divorcio?",
        "answer": "El divorcio puede ser: 1) Mutuo acuerdo: ambos cónyuges presentan convenio regulador ante notario o juez. 2) Contencioso: cuando no hay acuerdo, requiere demanda y procedimiento judicial. Requisitos básicos: matrimonio válido, transcurrido plazo mínimo (varía por país), y cumplir causas legales."
    },
    {
        "question": "¿Qué derechos tiene un arrendatario?",
        "answer": "Derechos básicos del arrendatario: 1) Uso pacífico de la vivienda, 2) Renovación automática en muchos casos, 3) Limitación al aumento de renta, 4) Derecho a desistimiento en plazos legales, 5) Reparaciones a cargo del propietario (excepto menores). Varían según legislación local."
    },
    {
        "question": "¿Qué es un testamento?",
        "answer": "El testamento es un acto jurídico unilateral por el que una persona dispone sobre su patrimonio para después de su muerte. Tipos comunes: 1) Ológrafo (manuscrito), 2) Abierto (ante notario), 3) Cerrado. Puede modificarse o revocarse mientras el testador viva."
    },
    {
        "question": "¿Cómo denunciar acoso laboral?",
        "answer": "Pasos básicos: 1) Recopilar pruebas (emails, testigos, etc.), 2) Presentar denuncia ante Inspección de Trabajo, 3) Opcionalmente demanda laboral. En muchos países se considera despido nulo si es por acoso. Plazos varían (generalmente 1 año desde los hechos)."
    }
]

# Cargar base de conocimiento legal
try:
    current_dir = Path(__file__).parent
    legal_db_path = current_dir.parent / 'data' / 'legal_db.json'
    
    with open(legal_db_path, 'r', encoding='utf-8') as f:
        LEGAL_KB = json.load(f)
    print(f"Base de conocimiento cargada desde {legal_db_path}")
except FileNotFoundError:
    print(f"Advertencia: No se encontró el archivo {legal_db_path}. Usando base de conocimiento por defecto.")
    LEGAL_KB = DEFAULT_KB
except json.JSONDecodeError as e:
    print(f"Error al parsear el archivo JSON: {e}. Usando base de conocimiento por defecto.")
    LEGAL_KB = DEFAULT_KB
except Exception as e:
    print(f"Error inesperado: {str(e)}. Usando base de conocimiento por defecto.")
    LEGAL_KB = DEFAULT_KB

STOP_WORDS = set(stopwords.words('spanish'))

def preprocess_text(text):
    """Normaliza y tokeniza el texto para análisis."""
    if not isinstance(text, str):
        return []
    
    try:
        # Limpieza básica del texto
        text = text.lower()
        text = re.sub(r'[^\w\sáéíóúüñ]', '', text)  # Conserva caracteres españoles
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in STOP_WORDS and len(token) > 2]
        return tokens
    except Exception as e:
        print(f"Error en preprocesamiento de texto: {str(e)}")
        return []

def calculate_similarity(tokens1, tokens2):
    """Calcula similitud entre conjuntos de tokens usando Jaccard mejorado."""
    if not tokens1 or not tokens2:
        return 0.0
    
    try:
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        # Ponderación por coincidencias exactas
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        # Coeficiente de Jaccard mejorado
        jaccard = intersection / union if union != 0 else 0
        
        # Bonus por coincidencia de palabras completas
        exact_matches = sum(1 for word in tokens1 if word in tokens2)
        bonus = exact_matches * 0.1 / len(tokens1) if tokens1 else 0
        
        return min(jaccard + bonus, 1.0)  # Asegurar no pasar de 1
    except Exception as e:
        print(f"Error calculando similitud: {str(e)}")
        return 0.0

def process_legal_query(query):
    """Procesa una consulta legal y devuelve la respuesta más relevante."""
    if not query or not isinstance(query, str):
        return "Por favor, proporcione una consulta válida."
    
    try:
        query_tokens = preprocess_text(query)
        
        if not query_tokens:
            return "No pude procesar su consulta. ¿Podría reformularla?"
        
        best_match = None
        highest_score = 0.0
        
        # Búsqueda semántica mejorada
        for item in LEGAL_KB:
            if not isinstance(item, dict) or 'question' not in item or 'answer' not in item:
                continue
                
            question_tokens = preprocess_text(item['question'])
            score = calculate_similarity(query_tokens, question_tokens)
            
            # Considerar también coincidencias parciales en la respuesta
            answer_tokens = preprocess_text(item['answer'])
            answer_score = calculate_similarity(query_tokens, answer_tokens) * 0.3
            total_score = score + answer_score
            
            if total_score > highest_score:
                highest_score = total_score
                best_match = item
        
        # Umbral de similitud ajustable
        similarity_threshold = 0.35
        
        if best_match and highest_score > similarity_threshold:
            return best_match['answer']
        else:
            # Sugerencias para preguntas similares
            suggestions = [
                item['question'] for item in LEGAL_KB 
                if calculate_similarity(query_tokens, preprocess_text(item['question'])) > 0.2
            ][:3]
            
            base_msg = "No encontré información específica sobre su consulta."
            if suggestions:
                return (f"{base_msg} ¿Quizás quisiera saber sobre:\n- " + 
                       "\n- ".join(suggestions))
            return f"{base_msg} ¿Podría reformular su pregunta o ser más específico?"
            
    except Exception as e:
        print(f"Error procesando consulta: {str(e)}")
        return "Ocurrió un error al procesar su solicitud. Por favor, intente nuevamente."