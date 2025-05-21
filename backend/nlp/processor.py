import json
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
import logging
from collections import Counter
import spacy
import datetime

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('legal_assistant')

# Cargar modelos de NLP
try:
    # Cargar modelo en español de spaCy para análisis más detallado
    nlp = spacy.load("es_core_news_md")
    logger.info("Modelo de spaCy cargado correctamente")
except Exception as e:
    logger.warning(f"No se pudo cargar el modelo de spaCy: {str(e)}. Instalando...")
    try:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "es_core_news_md"])
        nlp = spacy.load("es_core_news_md")
        logger.info("Modelo de spaCy instalado y cargado correctamente")
    except Exception as e:
        logger.error(f"Error al instalar modelo de spaCy: {str(e)}")
        nlp = None

# Inicializar recursos de NLTK
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    stemmer = SnowballStemmer('spanish')
    spanish_stopwords = set(stopwords.words('spanish'))
    logger.info("Recursos de NLTK cargados correctamente")
except Exception as e:
    logger.error(f"Error al cargar recursos de NLTK: {str(e)}")
    spanish_stopwords = set()
    stemmer = None

# Datos adicionales
CURRENT_DATE = datetime.datetime.now().strftime("%d/%m/%Y")
LEGAL_ENTITIES = {
    "fiscalía": "Organismo encargado de la investigación de los delitos y el ejercicio de la acción penal",
    "tribunal": "Órgano judicial que resuelve los conflictos jurídicos",
    "juez": "Persona que tiene autoridad para juzgar y sentenciar",
    "abogado": "Profesional del derecho que ejerce la defensa jurídica",
    "notario": "Funcionario público autorizado para dar fe de los contratos y actos extrajudiciales",
    "procuraduría": "Institución encargada de velar por el cumplimiento de las leyes",
    "defensoría": "Institución encargada de proteger los derechos de los ciudadanos"
}

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
    logger.info(f"Base de conocimiento cargada desde {legal_db_path}")
except FileNotFoundError:
    logger.warning(f"No se encontró el archivo {legal_db_path}. Usando base de conocimiento por defecto.")
    LEGAL_KB = DEFAULT_KB
except json.JSONDecodeError as e:
    logger.error(f"Error al parsear el archivo JSON: {e}. Usando base de conocimiento por defecto.")
    LEGAL_KB = DEFAULT_KB
except Exception as e:
    logger.error(f"Error inesperado: {str(e)}. Usando base de conocimiento por defecto.")
    LEGAL_KB = DEFAULT_KB

# Variables para vectorización de texto
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: [stemmer.stem(word) for word in word_tokenize(x.lower()) 
                        if word not in spanish_stopwords and len(word) > 2],
    stop_words=list(spanish_stopwords) if spanish_stopwords else None
)

# Preprocesar y vectorizar la base de conocimiento
try:
    # Preparar corpus para vectorización
    corpus = [item['question'] + " " + item['answer'] for item in LEGAL_KB]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    logger.info(f"Base de conocimiento vectorizada correctamente: {len(corpus)} documentos")
except Exception as e:
    logger.error(f"Error al vectorizar base de conocimiento: {str(e)}")
    tfidf_matrix = None

# Función para preprocesar texto
def preprocess_text(text):
    """Preprocesa el texto para análisis con técnicas avanzadas de NLP."""
    if not isinstance(text, str):
        return "", []
    
    try:
        # Limpieza básica del texto
        text = text.lower()
        text = re.sub(r'[^\w\sáéíóúüñ¿?¡!.,;]', '', text)  # Conserva caracteres españoles y puntuación básica
        
        # Tokenización y eliminación de stopwords
        tokens = word_tokenize(text)
        clean_tokens = [token for token in tokens if token not in spanish_stopwords and len(token) > 2]
        
        # Análisis con spaCy si está disponible
        if nlp:
            doc = nlp(text)
            # Extraer entidades nombradas
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            # Extraer frases nominales principales
            noun_chunks = [chunk.text for chunk in doc.noun_chunks]
            # Análisis de dependencias para identificar relaciones
            dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
            
            return text, {
                "tokens": clean_tokens,
                "entities": entities,
                "noun_chunks": noun_chunks,
                "dependencies": dependencies,
                "stemmed_tokens": [stemmer.stem(token) for token in clean_tokens] if stemmer else []
            }
        else:
            # Versión simplificada si spaCy no está disponible
            return text, {
                "tokens": clean_tokens,
                "stemmed_tokens": [stemmer.stem(token) for token in clean_tokens] if stemmer else [],
                "entities": [],
                "noun_chunks": [],
                "dependencies": []
            }
    except Exception as e:
        logger.error(f"Error en preprocesamiento de texto: {str(e)}")
        return text, {"tokens": [], "stemmed_tokens": [], "entities": [], "noun_chunks": [], "dependencies": []}

def extract_entities(nlp_features):
    """Extrae entidades legales reconocidas en el texto."""
    entities = {}
    
    # Entidades de spaCy
    for ent, label in nlp_features.get("entities", []):
        if label in ["ORG", "PER", "LOC", "LAW"]:
            entities[ent.lower()] = label
    
    # Buscar entidades legales comunes
    for token in nlp_features.get("tokens", []):
        if token.lower() in LEGAL_ENTITIES:
            entities[token.lower()] = "LEGAL_ENTITY"
    
    # Buscar referencias a artículos
    text = " ".join(nlp_features.get("tokens", []))
    article_refs = re.findall(r'art[íi]culo\s+(\d+[o°\.]?)', text, re.IGNORECASE)
    if article_refs:
        entities["referencias_articulos"] = article_refs
    
    return entities

def identify_query_type(text, nlp_features):
    """Identifica el tipo de consulta para mejorar la búsqueda."""
    text = text.lower()
    
    # Patrones de consultas
    patterns = {
        "definicion": [r"qu[ée]\s+es", r"defin[ei]", r"concepto", r"significa"],
        "procedimiento": [r"c[óo]mo\s+(?:se|puedo)", r"procedimiento", r"pasos", r"tr[áa]mite"],
        "normativa": [r"ley", r"norma", r"regulaci[óo]n", r"art[íi]culo", r"decreto"],
        "derechos": [r"derecho", r"garant[íi]a", r"protecci[óo]n", r"amparo"],
        "plazos": [r"plazo", r"t[ée]rmino", r"fecha", r"vencimiento", r"prescripci[óo]n"],
        "requisitos": [r"requisito", r"necesito", r"condici[óo]n", r"exigencia"]
    }
    
    query_types = []
    for qtype, pattern_list in patterns.items():
        for pattern in pattern_list:
            if re.search(pattern, text):
                query_types.append(qtype)
                break
    
    # Verificar si es una pregunta directa
    is_question = bool(re.search(r'\?|c[óo]mo|qu[ée]|cu[áa]ndo|d[óo]nde|por\s+qu[ée]|cu[áa]l', text))
    
    # Detectar si busca referencias a artículos específicos
    article_reference = bool(re.search(r'art[íi]culo\s+\d+', text, re.IGNORECASE))
    
    return {
        "types": list(set(query_types)),  # Eliminar duplicados
        "is_question": is_question,
        "article_reference": article_reference
    }

def query_enhancement(query, nlp_features):
    """Mejora la consulta para búsqueda semántica."""
    # Extraer los términos más relevantes
    terms = nlp_features.get("tokens", [])
    
    # Identificar y expandir términos legales
    expanded_terms = []
    for term in terms:
        expanded_terms.append(term)
        
        # Agregar sinónimos o términos relacionados para términos legales comunes
        legal_synonyms = {
            "divorcio": ["separación", "disolución matrimonial"],
            "contrato": ["acuerdo", "convenio", "pacto"],
            "herencia": ["sucesión", "legado", "testamento"],
            "delito": ["crimen", "infracción", "ilícito"],
            "demanda": ["denuncia", "querella", "reclamación"],
            "juicio": ["proceso", "procedimiento", "litigio"],
            "sentencia": ["fallo", "resolución", "decisión judicial"]
        }
        
        if term.lower() in legal_synonyms:
            expanded_terms.extend(legal_synonyms[term.lower()])
    
    # Identificar el tipo de consulta
    query_type = identify_query_type(query, nlp_features)
    
    # Si es una referencia a un artículo, priorizarla
    if query_type["article_reference"]:
        article_nums = re.findall(r'art[íi]culo\s+(\d+[o°\.]?)', query, re.IGNORECASE)
        if article_nums:
            expanded_terms.extend([f"articulo {num}" for num in article_nums])
    
    enhanced_query = " ".join(list(set(expanded_terms)))
    return enhanced_query, query_type

def semantic_search(query, query_type):
    """Realiza búsqueda semántica en la base de conocimiento."""
    if tfidf_matrix is None:
        return []
    
    try:
        # Vectorizar la consulta
        query_vector = vectorizer.transform([query])
        
        # Calcular similitud con todos los documentos
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Ordenar resultados por similitud
        results = []
        for idx, score in enumerate(cosine_similarities):
            if score > 0.1:  # Umbral mínimo de similitud
                results.append({
                    "index": idx,
                    "score": float(score),
                    "item": LEGAL_KB[idx]
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Aplicar criterios de clasificación adicionales basados en el tipo de consulta
        if query_type["types"]:
            for result in results:
                # Bonificación para resultados que coinciden con el tipo de consulta
                for qtype in query_type["types"]:
                    if qtype == "definicion" and any(p in result["item"]["question"].lower() for p in ["qué es", "definición"]):
                        result["score"] *= 1.2
                    elif qtype == "procedimiento" and any(p in result["item"]["question"].lower() for p in ["cómo", "procedimiento"]):
                        result["score"] *= 1.2
                    elif qtype == "normativa" and any(p in result["item"]["question"].lower() for p in ["ley", "artículo"]):
                        result["score"] *= 1.2
        
        # Reordenar después de aplicar bonificaciones
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results
    except Exception as e:
        logger.error(f"Error en búsqueda semántica: {str(e)}")
        return []

def extract_answer_parts(query, answer, nlp_features):
    """Extrae partes relevantes de una respuesta según la consulta."""
    if not answer or not query:
        return answer
    
    try:
        # Dividir la respuesta en oraciones
        sentences = sent_tokenize(answer)
        
        # Identificar oraciones más relevantes
        query_tokens = set(nlp_features.get("tokens", []))
        sentence_scores = []
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = set(word_tokenize(sentence.lower()))
            common_tokens = query_tokens.intersection(sentence_tokens)
            
            # Calcular puntuación para esta oración
            score = len(common_tokens) / max(1, len(sentence_tokens))
            
            # Bonificación para primeras oraciones (suelen contener definiciones)
            if i == 0:
                score += 0.3
            
            sentence_scores.append((i, score, sentence))
        
        # Ordenar oraciones por relevancia
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Tomar las oraciones más relevantes (máximo 70% del total)
        max_sentences = max(1, int(len(sentences) * 0.7))
        selected_indices = [x[0] for x in sentence_scores[:max_sentences]]
        selected_indices.sort()  # Reordenar según orden original
        
        relevant_sentences = [sentences[i] for i in selected_indices]
        
        # Si son muy pocas oraciones, devolver respuesta completa
        if len(relevant_sentences) < 2 and len(sentences) > 3:
            return answer
            
        return " ".join(relevant_sentences)
    except Exception as e:
        logger.error(f"Error al extraer partes relevantes: {str(e)}")
        return answer

def generate_contextual_response(query, results, nlp_features, user_context=None):
    """Genera una respuesta contextual basada en los resultados de búsqueda."""
    if not results:
        return generate_fallback_response(query, nlp_features)
    
    try:
        # Obtener mejor resultado
        best_match = results[0]
        answer = best_match["item"]["answer"]
        
        # Extraer partes más relevantes de la respuesta
        focused_answer = extract_answer_parts(query, answer, nlp_features)
        
        # Personalizar la respuesta según el tipo de consulta
        query_type = identify_query_type(query, nlp_features)
        
        response_parts = []
        
        # Introducción personalizada según tipo de consulta
        if query_type["types"]:
            if "definicion" in query_type["types"]:
                response_parts.append("De acuerdo con la legislación colombiana,")
            elif "procedimiento" in query_type["types"]:
                response_parts.append("El procedimiento establecido indica que")
            elif "derechos" in query_type["types"]:
                response_parts.append("En cuanto a los derechos que mencionas,")
        
        # Agregar la respuesta principal
        response_parts.append(focused_answer)
        
        # Agregar fuentes si hay alta confianza
        if best_match["score"] > 0.7:
            article_match = re.search(r'Art[íi]culo\s+(\d+[o°\.]*)', answer)
            if article_match:
                response_parts.append(f"\n\nEsta información está basada en el Artículo {article_match.group(1)} del Código de Procedimiento Penal colombiano.")
        
        # Verificar si hay respuestas adicionales relevantes
        if len(results) > 1 and results[1]["score"] > 0.5:
            additional_info = extract_answer_parts(query, results[1]["item"]["answer"], nlp_features)
            if additional_info and additional_info != focused_answer:
                response_parts.append("\n\nAdicionalmente, debes tener en cuenta que: " + additional_info)
        
        # Si la confianza es baja, agregar disclaimers
        if best_match["score"] < 0.4:
            response_parts.append("\n\nTe recomiendo consultar con un abogado para obtener asesoría específica sobre tu caso particular.")
        
        return " ".join(response_parts)
    except Exception as e:
        logger.error(f"Error generando respuesta contextual: {str(e)}")
        return results[0]["item"]["answer"]

def generate_fallback_response(query, nlp_features):
    """Genera una respuesta cuando no se encuentran coincidencias."""
    # Identificar entidades y tipo de consulta
    entities = extract_entities(nlp_features)
    query_type = identify_query_type(query, nlp_features)
    
    # Construir respuesta personalizada según los datos disponibles
    if query_type["types"]:
        if "definicion" in query_type["types"]:
            return "No tengo información específica sobre la definición que buscas. Te recomiendo consultar el Código de Procedimiento Penal o un abogado especializado para obtener esta información. ¿Hay algún otro tema legal sobre el que pueda ayudarte?"
        
        if "procedimiento" in query_type["types"]:
            return "No tengo detalles específicos sobre el procedimiento que mencionas. Los procedimientos legales suelen estar regulados en códigos específicos y pueden variar. Te sugiero consultar con un profesional del derecho para obtener información precisa sobre este trámite."
    
    # Verificar si hay referencias a artículos
    if query_type["article_reference"] and "referencias_articulos" in entities:
        return f"No tengo información detallada sobre el artículo {entities['referencias_articulos'][0]} que mencionas. ¿Te gustaría que busque información sobre algún otro tema legal relacionado?"
    
    # Respuesta general
    suggestions = [
        "derechos fundamentales en el proceso penal",
        "etapas del proceso penal",
        "recursos en el proceso penal",
        "garantías procesales",
        "funciones del juez de control de garantías"
    ]
    
    # Seleccionar sugerencias más relevantes
    relevant_suggestions = []
    for suggestion in suggestions:
        for token in nlp_features.get("tokens", []):
            if token in suggestion:
                relevant_suggestions.append(suggestion)
                break
    
    if not relevant_suggestions:
        relevant_suggestions = suggestions[:3]  # Tomar primeras 3 sugerencias generales
    
    suggestions_text = ", ".join(relevant_suggestions)
    
    return f"No tengo información específica sobre tu consulta. Puedo ayudarte con temas como: {suggestions_text}. ¿Te gustaría obtener información sobre alguno de estos temas?"

def process_legal_query(query, user_context=None):
    """Procesa una consulta legal y devuelve la respuesta más relevante."""
    if not query or not isinstance(query, str):
        return "Por favor, proporcione una consulta válida."
    
    try:
        # 1. Preprocesar la consulta
        clean_query, nlp_features = preprocess_text(query)
        
        if not nlp_features["tokens"]:
            return "No pude procesar su consulta. ¿Podría reformularla?"
        
        # 2. Mejorar la consulta y determinar su tipo
        enhanced_query, query_type = query_enhancement(clean_query, nlp_features)
        
        # 3. Realizar búsqueda semántica
        search_results = semantic_search(enhanced_query, query_type)
        
        # 4. Generar respuesta contextual
        if search_results:
            response = generate_contextual_response(clean_query, search_results, nlp_features, user_context)
        else:
            response = generate_fallback_response(clean_query, nlp_features)
        
        # 5. Registrar la consulta para análisis y mejoras futuras
        logger.info(f"Consulta: '{query}' | Tipo: {query_type['types']} | Resultados: {len(search_results)}")
        
        return response
    except Exception as e:
        logger.error(f"Error general procesando consulta: {str(e)}")
        return "Ocurrió un error al procesar su solicitud. Por favor, intente nuevamente con una consulta diferente."

# Ejemplo de uso:
if __name__ == "__main__":
    test_queries = [
        "¿Qué es el principio de presunción de inocencia?",
        "¿Cómo se realiza una audiencia de imputación?",
        "¿Cuáles son los derechos de las víctimas en el proceso penal?",
        "¿Qué establece el Artículo 8 sobre el derecho a la defensa?",
        "¿Cuándo se considera cosa juzgada?"
    ]
    
    print("=== TESTING LEGAL ASSISTANT ===")
    for query in test_queries:
        print(f"\nQ: {query}")
        response = process_legal_query(query)
        print(f"A: {response}")
    print("\n=== TEST COMPLETE ===")