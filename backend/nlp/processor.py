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
import datetime

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('legal_assistant')

# Intentar cargar spaCy (opcional)
try:
    import spacy
    nlp = spacy.load("es_core_news_md")
    logger.info("Modelo de spaCy cargado correctamente")
except Exception as e:
    logger.warning(f"No se pudo cargar spaCy: {str(e)}. Continuando sin análisis avanzado.")
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

# Definir dominios legales y sus palabras clave
LEGAL_DOMAINS = {
    'penal': {
        'keywords': {
            'delito', 'delitos', 'penal', 'criminal', 'imputado', 'acusado', 'fiscal', 'fiscalia',
            'juez', 'tribunal', 'audiencia', 'captura', 'detencion', 'proceso', 'procedimiento',
            'investigacion', 'juzgamiento', 'sentencia', 'condena', 'absolucion', 'presuncion',
            'inocencia', 'defensa', 'victima', 'victimas', 'testimonio', 'prueba', 'pruebas',
            'recurso', 'apelacion', 'casacion', 'garantias', 'derechos', 'libertad', 'cautelar',
            'medida', 'cpp', 'control', 'legalidad'
        },
        'patterns': [
            r'art[íi]culo\s*\d+.*cpp',
            r'código.*procedimiento.*penal',
            r'proceso.*penal',
            r'derecho.*procesal'
        ],
        'file': 'legal_db.json',
        'description': 'derecho procesal penal'
    },
    'civil': {
        'keywords': {
            'contrato', 'contratos', 'civil', 'matrimonio', 'divorcio', 'herencia', 'testamento',
            'sucesion', 'propiedad', 'arrendamiento', 'compraventa', 'hipoteca', 'patrimonio',
            'persona', 'familia', 'menor', 'tutela', 'curatela', 'adopcion', 'patria', 'potestad',
            'sociedad', 'conyugal', 'gananciales', 'legitima', 'legado', 'albaceas', 'notario',
            'registro', 'escritura', 'tradicion', 'posesion', 'dominio', 'usufructo', 'servidumbre'
        },
        'patterns': [
            r'código.*civil',
            r'derecho.*civil',
            r'registro.*civil',
            r'notaria'
        ],
        'file': 'codigo_civil.json',
        'description': 'derecho civil'
    },
    'laboral': {
        'keywords': {
            'trabajo', 'laboral', 'empleado', 'empleador', 'empresa', 'contrato.*trabajo',
            'salario', 'sueldo', 'prestaciones', 'vacaciones', 'cesantias', 'prima', 'liquidacion',
            'despido', 'renuncia', 'incapacidad', 'pension', 'jubilacion', 'eps', 'arl',
            'sindicato', 'convencion', 'colectiva', 'huelga', 'fuero', 'sindical', 'ministerio.*trabajo',
            'codigo.*trabajo', 'jornada', 'horas.*extras', 'dominical', 'festivo', 'licencia',
            'maternidad', 'paternidad', 'acoso.*laboral', 'riesgos.*profesionales'
        },
        'patterns': [
            r'código.*trabajo',
            r'derecho.*laboral',
            r'ministerio.*trabajo',
            r'contrato.*trabajo'
        ],
        'file': 'codigo_trabajo.json',
        'description': 'derecho laboral'
    },
    'penal_sustantivo': {
        'keywords': {
            'codigo.*penal', 'delito', 'delitos', 'pena', 'penas', 'sancion', 'multa', 'prision',
            'homicidio', 'asesinato', 'lesiones', 'hurto', 'robo', 'estafa', 'fraude', 'extorsion',
            'secuestro', 'violacion', 'abuso', 'sexual', 'corrupcion', 'soborno', 'prevaricato',
            'peculado', 'cohecho', 'concusion', 'trafico', 'drogas', 'narcotrafio', 'terrorismo',
            'rebelion', 'sedicion', 'calumnia', 'injuria', 'difamacion', 'falsedad', 'falsificacion'
        },
        'patterns': [
            r'código.*penal(?!.*procedimiento)',
            r'derecho.*penal(?!.*procesal)',
            r'pena.*privativa',
            r'tipo.*penal'
        ],
        'file': 'codigo_penal.json',
        'description': 'derecho penal sustantivo'
    }
}

# Cargar bases de conocimiento de todos los dominios
KNOWLEDGE_BASES = {}
VECTORIZERS = {}
TFIDF_MATRICES = {}

def load_knowledge_base(domain, filename):
    """Carga una base de conocimiento específica."""
    try:
        current_dir = Path(__file__).parent
        file_path = current_dir.parent / 'data' / filename
        
        if not file_path.exists():
            logger.warning(f"Archivo no encontrado: {file_path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Base de conocimiento '{domain}' cargada: {len(data)} documentos")
        return data
    except Exception as e:
        logger.error(f"Error cargando {filename}: {str(e)}")
        return []

def initialize_knowledge_bases():
    """Inicializa todas las bases de conocimiento y sus vectorizadores."""
    for domain, config in LEGAL_DOMAINS.items():
        # Cargar base de conocimiento
        kb = load_knowledge_base(domain, config['file'])
        if kb:
            KNOWLEDGE_BASES[domain] = kb
            
            # Crear vectorizador específico para este dominio
            try:
                vectorizer = TfidfVectorizer(
                    tokenizer=lambda x: [stemmer.stem(word) for word in word_tokenize(x.lower()) 
                                        if word not in spanish_stopwords and len(word) > 2] if stemmer else word_tokenize(x.lower()),
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=1,  # Reducido de 2 a 1
                    max_df=0.90,  # Aumentado de 0.85 a 0.90
                    token_pattern=None
                )
                
                # Preparar corpus
                corpus = [item['question'] + " " + item['answer'] for item in kb]
                tfidf_matrix = vectorizer.fit_transform(corpus)
                
                VECTORIZERS[domain] = vectorizer
                TFIDF_MATRICES[domain] = tfidf_matrix
                
                logger.info(f"Vectorizador para '{domain}' creado: {len(corpus)} documentos")
            except Exception as e:
                logger.error(f"Error creando vectorizador para {domain}: {str(e)}")

# Inicializar todas las bases de conocimiento
initialize_knowledge_bases()

def detect_legal_domain(text):
    """Detecta el dominio legal más probable para una consulta."""
    text_lower = text.lower()
    domain_scores = {}
    
    for domain, config in LEGAL_DOMAINS.items():
        score = 0
        
        # Puntuar por palabras clave
        keyword_matches = sum(1 for keyword in config['keywords'] if keyword in text_lower)
        score += keyword_matches * 2
        
        # Puntuar por patrones regex
        pattern_matches = sum(1 for pattern in config['patterns'] if re.search(pattern, text_lower))
        score += pattern_matches * 3
        
        domain_scores[domain] = score
    
    # Encontrar el dominio con mayor puntaje
    if domain_scores:
        best_domain = max(domain_scores, key=domain_scores.get)
        max_score = domain_scores[best_domain]
        
        # Solo devolver un dominio si tiene un puntaje mínimo
        if max_score >= 1:
            return best_domain, max_score
    
    return None, 0

def detect_query_intent(text, domain=None):
    """Detecta la intención de la consulta considerando el dominio."""
    text_lower = text.lower()
    
    # Intenciones específicas por dominio
    if domain == 'civil':
        if any(word in text_lower for word in ['testamento', 'herencia', 'sucesion']):
            return 'inheritance_query'
        elif any(word in text_lower for word in ['contrato', 'arrendamiento', 'compraventa']):
            return 'contract_query'
        elif any(word in text_lower for word in ['matrimonio', 'divorcio', 'familia']):
            return 'family_query'
        return 'civil_general'
    
    elif domain == 'laboral':
        if any(word in text_lower for word in ['despido', 'liquidacion', 'terminacion']):
            return 'termination_query'
        elif any(word in text_lower for word in ['salario', 'prestaciones', 'pago']):
            return 'salary_query'
        elif any(word in text_lower for word in ['contrato', 'vinculacion', 'trabajo']):
            return 'employment_query'
        return 'labor_general'
    
    elif domain == 'penal':
        if any(word in text_lower for word in ['artículo', 'establece', 'dice', 'contenido']):
            return 'article_query'
        elif any(word in text_lower for word in ['cómo', 'procedimiento', 'proceso', 'pasos', 'audiencia', 'control']):
            return 'procedure_query'
        elif any(word in text_lower for word in ['derechos', 'garantías', 'protección', 'victima', 'victimas']):
            return 'rights_query'
        return 'penal_general'
    
    elif domain == 'penal_sustantivo':
        if any(word in text_lower for word in ['pena', 'sancion', 'castigo']):
            return 'penalty_query'
        elif any(word in text_lower for word in ['delito', 'crimen', 'tipificacion']):
            return 'crime_query'
        return 'penal_substantive_general'
    
    # Intenciones generales
    if any(phrase in text_lower for phrase in ['de que', 'que temas', 'que puedes', 'ayuda', 'sobre que']):
        return 'general_help'
    
    if any(word in text_lower for word in ['qué es', 'definición', 'concepto', 'significa']):
        return 'definition_query'
    
    return 'general_query'

def semantic_search_domain(query, domain, min_score=0.15):  # Reducido de 0.25 a 0.15
    """Realiza búsqueda semántica en un dominio específico."""
    if domain not in VECTORIZERS or domain not in TFIDF_MATRICES:
        return []
    
    try:
        vectorizer = VECTORIZERS[domain]
        tfidf_matrix = TFIDF_MATRICES[domain]
        kb = KNOWLEDGE_BASES[domain]
        
        # Vectorizar la consulta
        query_vector = vectorizer.transform([query])
        
        # Calcular similitud
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Filtrar y ordenar resultados
        results = []
        for idx, score in enumerate(cosine_similarities):
            if score > min_score:
                results.append({
                    "index": idx,
                    "score": float(score),
                    "item": kb[idx],
                    "domain": domain
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:10]  # Aumentado de 5 a 10
        
    except Exception as e:
        logger.error(f"Error en búsqueda semántica para {domain}: {str(e)}")
        return []

def preprocess_text(text):
    """Preprocesa el texto para análisis."""
    if not isinstance(text, str):
        return "", []
    
    try:
        text = text.lower()
        text = re.sub(r'[^\w\sáéíóúüñ¿?¡!.,;]', '', text)
        tokens = word_tokenize(text)
        clean_tokens = [token for token in tokens if token not in spanish_stopwords and len(token) > 2]
        
        if nlp:
            doc = nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            noun_chunks = [chunk.text for chunk in doc.noun_chunks]
            dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
            
            return text, {
                "tokens": clean_tokens,
                "entities": entities,
                "noun_chunks": noun_chunks,
                "dependencies": dependencies,
                "stemmed_tokens": [stemmer.stem(token) for token in clean_tokens] if stemmer else []
            }
        else:
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

def extract_relevant_answer_parts(answer, query_tokens, max_sentences=3):
    """Extrae las partes más relevantes de una respuesta larga."""
    try:
        sentences = sent_tokenize(answer)
        if len(sentences) <= max_sentences:
            return answer
        
        # Calcular relevancia de cada oración
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            sentence_tokens = set(word_tokenize(sentence.lower()))
            query_token_set = set(query_tokens)
            
            # Calcular similitud por tokens comunes
            common_tokens = query_token_set.intersection(sentence_tokens)
            score = len(common_tokens) / max(1, len(sentence_tokens))
            
            # Bonificación para oraciones al inicio (suelen contener definiciones)
            if i == 0:
                score += 0.2
            
            sentence_scores.append((i, score, sentence))
        
        # Ordenar por relevancia y tomar las mejores
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        best_sentences = sentence_scores[:max_sentences]
        
        # Reordenar según orden original
        best_sentences.sort(key=lambda x: x[0])
        
        return " ".join([sent[2] for sent in best_sentences])
    except:
        return answer

def generate_domain_response(query, results, domain, intent, query_tokens):
    """Genera respuesta específica según el dominio detectado."""
    # Usar umbral más bajo para mejor cobertura
    threshold = 0.2  # Reducido de 0.4 a 0.2
    
    if not results or results[0]["score"] < threshold:
        # Respuestas por defecto según dominio
        if domain == 'civil':
            return f"No encontré información específica sobre tu consulta de derecho civil. Te puedo ayudar con temas como: contratos, matrimonio, divorcio, herencias, testamentos, propiedad, arrendamientos, y otros aspectos del Código Civil. ¿Podrías ser más específico?"
        elif domain == 'laboral':
            return f"No encontré información específica sobre tu consulta laboral. Te puedo ayudar con: contratos de trabajo, despidos, liquidaciones, prestaciones sociales, jornada laboral, y otros temas del Código de Trabajo. ¿Qué aspecto laboral específico te interesa?"
        elif domain == 'penal_sustantivo':
            return f"No encontré información específica sobre tu consulta de derecho penal. Te puedo ayudar con: tipos penales, delitos, penas, y otros aspectos del Código Penal. ¿Sobre qué delito o pena específica quieres saber?"
        elif domain == 'penal':
            return f"No encontré información específica sobre tu consulta de procedimiento penal. Te puedo ayudar con: etapas del proceso, derechos del imputado, garantías procesales, derechos de víctimas, y otros aspectos del Código de Procedimiento Penal."
    
    # Si hay buenos resultados
    if results and results[0]["score"] > threshold:
        best_match = results[0]
        answer = best_match["item"]["answer"]
        
        # Extraer partes relevantes de respuestas largas
        processed_answer = extract_relevant_answer_parts(answer, query_tokens, max_sentences=4)
        
        domain_desc = LEGAL_DOMAINS[domain]['description']
        
        # Personalizar introducción según dominio e intención
        if intent == 'inheritance_query':
            return f"Según el Código Civil colombiano en materia de herencias: {processed_answer}"
        elif intent == 'contract_query':
            return f"En cuanto a contratos, el Código Civil establece: {processed_answer}"
        elif intent == 'family_query':
            return f"Sobre derecho de familia: {processed_answer}"
        elif intent == 'termination_query':
            return f"En materia laboral sobre terminación de contratos: {processed_answer}"
        elif intent == 'salary_query':
            return f"Según la legislación laboral sobre salarios: {processed_answer}"
        elif intent == 'crime_query':
            return f"El Código Penal establece sobre este delito: {processed_answer}"
        elif intent == 'penalty_query':
            return f"En cuanto a las penas, el Código Penal dispone: {processed_answer}"
        elif intent == 'procedure_query':
            return f"El procedimiento establecido indica que: {processed_answer}"
        elif intent == 'rights_query':
            return f"En cuanto a los derechos que mencionas: {processed_answer}"
        elif intent == 'article_query':
            return f"Según la legislación ({domain_desc}): {processed_answer}"
        else:
            return f"De acuerdo con la legislación colombiana ({domain_desc}): {processed_answer}"
    
    return "No encontré información específica sobre tu consulta. ¿Podrías reformular tu pregunta o ser más específico?"

def process_legal_query(query, user_context=None):
    """Procesa una consulta legal considerando múltiples dominios."""
    if not query or not isinstance(query, str):
        return "Por favor, proporcione una consulta válida."
    
    try:
        # 1. Detectar dominio legal
        domain, domain_score = detect_legal_domain(query)
        
        # 2. Si es consulta de ayuda general
        if not domain and any(phrase in query.lower() for phrase in ['de que', 'que temas', 'que puedes', 'ayuda']):
            available_domains = list(KNOWLEDGE_BASES.keys())
            domain_descriptions = [LEGAL_DOMAINS[d]['description'] for d in available_domains]
            return f"Soy un asistente legal que puede ayudarte con: {', '.join(domain_descriptions)}. También puedo responder sobre artículos específicos, procedimientos, derechos y garantías. ¿Sobre qué tema específico te gustaría saber más?"
        
        # 3. Si no se detecta dominio específico, buscar en todos
        if not domain:
            # Buscar en todos los dominios disponibles
            all_results = []
            clean_query, nlp_features = preprocess_text(query)
            
            for search_domain in KNOWLEDGE_BASES.keys():
                domain_results = semantic_search_domain(clean_query, search_domain, min_score=0.15)
                for result in domain_results:
                    all_results.append(result)
            
            # Ordenar todos los resultados por score
            all_results.sort(key=lambda x: x["score"], reverse=True)
            
            if all_results and all_results[0]["score"] > 0.2:
                best_result = all_results[0]
                found_domain = best_result["domain"]
                domain_desc = LEGAL_DOMAINS[found_domain]['description']
                answer = extract_relevant_answer_parts(best_result["item"]["answer"], nlp_features["tokens"])
                
                logger.info(f"Consulta: '{query}' | Dominio encontrado: {found_domain} | Score: {best_result['score']:.3f}")
                return f"Encontré información relevante en {domain_desc}: {answer}"
            else:
                return "No pude identificar el área legal de tu consulta. Te puedo ayudar con derecho civil, laboral, penal, y procesal penal. ¿Podrías ser más específico sobre el tema legal que te interesa?"
        
        # 4. Preprocesar consulta
        clean_query, nlp_features = preprocess_text(query)
        
        if not nlp_features["tokens"]:
            return "No pude procesar su consulta. ¿Podría reformularla?"
        
        # 5. Detectar intención específica del dominio
        intent = detect_query_intent(query, domain)
        
        # 6. Realizar búsqueda semántica en el dominio detectado
        search_results = semantic_search_domain(clean_query, domain, min_score=0.15)
        
        # 7. Generar respuesta específica del dominio
        response = generate_domain_response(query, search_results, domain, intent, nlp_features["tokens"])
        
        # 8. Registrar la consulta
        score_info = f" | Score: {search_results[0]['score']:.3f}" if search_results else ""
        logger.info(f"Consulta: '{query}' | Dominio: {domain} (score: {domain_score}) | Intención: {intent} | Resultados: {len(search_results)}{score_info}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error general procesando consulta: {str(e)}")
        return "Ocurrió un error al procesar su solicitud. Por favor, intente nuevamente."

# Ejemplo de uso
if __name__ == "__main__":
    test_queries = [
        "¿Cómo hago un testamento?",
        "¿Qué es el principio de presunción de inocencia?",
        "¿Cuáles son mis derechos laborales?",
        "¿Qué pena tiene el homicidio?",
        "¿Cuáles son los derechos de las víctimas?",
        "¿Cómo se realiza una audiencia de control de garantías?",
        "de qué temas me puedes ayudar"
    ]
    
    print("=== TESTING OPTIMIZED MULTI-DOMAIN LEGAL ASSISTANT ===")
    for query in test_queries:
        print(f"\nQ: {query}")
        response = process_legal_query(query)
        print(f"A: {response}")
    print("\n=== TEST COMPLETE ===")