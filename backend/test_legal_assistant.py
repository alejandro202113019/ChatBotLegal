import unittest
import json
import os
import sys
from pathlib import Path

# Agregar directorio padre al path para importar módulos
sys.path.append(str(Path(__file__).parent.parent))

# Importar módulos a probar
from nlp.processor import (
    process_legal_query,
    preprocess_text,
    extract_entities,
    identify_query_type,
    query_enhancement,
    semantic_search,
    extract_answer_parts
)

class TestLegalAssistant(unittest.TestCase):
    """Test suite para el asistente legal inteligente."""
    
    def setUp(self):
        """Configurar datos de prueba."""
        # Cargar dataset de prueba
        try:
            test_data_path = Path(__file__).parent / 'test_data' / 'test_queries.json'
            with open(test_data_path, 'r', encoding='utf-8') as f:
                self.test_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error cargando datos de prueba: {e}")
            # Crear datos mínimos de prueba si no se puede cargar el archivo
            self.test_data = {
                "test_queries": [
                    {
                        "query": "¿Qué es la presunción de inocencia?",
                        "expected_keywords": ["inocencia", "presunción"],
                        "expected_type": "definicion"
                    },
                    {
                        "query": "¿Cómo se realiza una audiencia de imputación?",
                        "expected_keywords": ["audiencia", "imputación"],
                        "expected_type": "procedimiento"
                    },
                    {
                        "query": "¿Qué establece el Artículo 8 del CPP?",
                        "expected_keywords": ["artículo", "8"],
                        "expected_type": "normativa"
                    }
                ]
            }
    
    def test_preprocessing(self):
        """Probar la funcionalidad de preprocesamiento de texto."""
        for test_case in self.test_data.get("test_queries", []):
            query = test_case.get("query", "")
            if not query:
                continue
                
            # Probar preprocesamiento
            clean_text, features = preprocess_text(query)
            
            # Verificar que el resultado no esté vacío
            self.assertTrue(clean_text)
            self.assertTrue(features.get("tokens"))
            
            # Verificar que se extraen palabras clave esperadas
            expected_keywords = test_case.get("expected_keywords", [])
            for keyword in expected_keywords:
                found = False
                for token in features.get("tokens", []):
                    if keyword.lower() in token.lower():
                        found = True
                        break
                self.assertTrue(found, f"Palabra clave '{keyword}' no encontrada en '{query}'")
    
    def test_query_type_identification(self):
        """Probar la identificación del tipo de consulta."""
        for test_case in self.test_data.get("test_queries", []):
            query = test_case.get("query", "")
            expected_type = test_case.get("expected_type", "")
            
            if not query or not expected_type:
                continue
                
            # Preprocesar para obtener características
            _, features = preprocess_text(query)
            
            # Identificar tipo de consulta
            query_type = identify_query_type(query, features)
            
            # Verificar que el tipo esperado está en los tipos identificados
            self.assertTrue(
                expected_type in query_type.get("types", []),
                f"Tipo '{expected_type}' no identificado en '{query}', se encontró: {query_type.get('types')}"
            )
    
    def test_entity_extraction(self):
        """Probar la extracción de entidades legales."""
        # Casos de prueba para extracción de entidades
        entity_test_cases = [
            {
                "query": "¿Qué dice la fiscalía sobre el juez en el artículo 10?",
                "expected_entities": ["fiscalía", "juez", "artículo"]
            },
            {
                "query": "El abogado defensor puede solicitar una audiencia?",
                "expected_entities": ["abogado"]
            }
        ]
        
        for test_case in entity_test_cases:
            query = test_case.get("query", "")
            expected_entities = test_case.get("expected_entities", [])
            
            if not query:
                continue
                
            # Preprocesar para obtener características
            _, features = preprocess_text(query)
            
            # Extraer entidades
            entities = extract_entities(features)
            
            # Verificar que las entidades esperadas están presentes
            for entity in expected_entities:
                found = False
                for extracted_entity in entities.keys():
                    if entity.lower() in extracted_entity.lower():
                        found = True
                        break
                
                self.assertTrue(
                    found,
                    f"Entidad '{entity}' no encontrada en '{query}', se encontraron: {list(entities.keys())}"
                )
    
    def test_query_enhancement(self):
        """Probar la mejora de consultas."""
        for test_case in self.test_data.get("test_queries", []):
            query = test_case.get("query", "")
            
            if not query:
                continue
                
            # Preprocesar para obtener características
            _, features = preprocess_text(query)
            
            # Mejorar consulta
            enhanced_query, _ = query_enhancement(query, features)
            
            # Verificar que la consulta mejorada no está vacía y es más larga
            # (indicando expansión de términos)
            self.assertTrue(enhanced_query)
            self.assertTrue(
                len(enhanced_query.split()) >= len(query.split()),
                f"La consulta mejorada no es más completa: '{enhanced_query}' vs '{query}'"
            )
    
    def test_extract_answer_parts(self):
        """Probar la extracción de partes relevantes de respuestas."""
        test_cases = [
            {
                "query": "¿Qué es la presunción de inocencia?",
                "answer": "La presunción de inocencia es un principio fundamental del derecho procesal penal. Establece que toda persona se presume inocente hasta que se demuestre su culpabilidad. En el sistema colombiano, está consagrado en el Artículo 7 del Código de Procedimiento Penal. Allí se establece que el acusado no está obligado a presentar prueba de su inocencia.",
                "expected_substrings": ["presunción de inocencia", "Artículo 7"]
            },
            {
                "query": "¿Cuándo prescribe un delito?",
                "answer": "La prescripción de la acción penal depende del tipo de delito. Para delitos con pena privativa de libertad, el término de prescripción es igual al máximo de la pena fijada en la ley, pero no puede ser inferior a 5 años ni superior a 20. Para delitos con penas no privativas de libertad, la acción prescribirá en 5 años. La prescripción se interrumpe con la formulación de la imputación. El Código Penal establece condiciones específicas para ciertos delitos.",
                "expected_substrings": ["prescripción", "delito"]
            }
        ]
        
        for test_case in test_cases:
            query = test_case.get("query", "")
            answer = test_case.get("answer", "")
            expected_substrings = test_case.get("expected_substrings", [])
            
            if not query or not answer:
                continue
                
            # Preprocesar consulta
            _, features = preprocess_text(query)
            
            # Extraer partes relevantes
            extracted_answer = extract_answer_parts(query, answer, features)
            
            # Verificar que los fragmentos esperados están presentes
            for substring in expected_substrings:
                self.assertTrue(
                    substring.lower() in extracted_answer.lower(),
                    f"Substring '{substring}' no encontrado en la respuesta extraída: '{extracted_answer}'"
                )
    
    def test_end_to_end_processing(self):
        """Probar el procesamiento completo de consultas (end-to-end)."""
        test_queries = [
            "¿Qué es la presunción de inocencia?",
            "¿Cómo se realiza una audiencia de imputación?",
            "¿Qué establece el Artículo 8 del CPP?",
            "¿Cuáles son los derechos de las víctimas?",
            "¿Qué es el principio de oralidad?"
        ]
        
        for query in test_queries:
            # Procesar consulta
            response = process_legal_query(query)
            
            # Verificar que la respuesta no está vacía
            self.assertTrue(response)
            self.assertTrue(len(response) > 50, f"Respuesta demasiado corta para '{query}': '{response}'")
            
            # Verificar que la respuesta contiene palabras clave de la consulta
            for word in query.lower().split():
                if len(word) > 3 and word not in ["como", "cual", "cuales", "donde", "cuando"]:
                    self.assertTrue(
                        word in response.lower() or any(w.startswith(word[:-1]) for w in response.lower().split()),
                        f"Palabra clave '{word}' no encontrada en la respuesta a '{query}'"
                    )

def create_test_data():
    """Crear un archivo de datos de prueba si no existe."""
    test_data_dir = Path(__file__).parent / 'test_data'
    test_data_file = test_data_dir / 'test_queries.json'
    
    if not test_data_dir.exists():
        test_data_dir.mkdir(parents=True)
    
    if not test_data_file.exists():
        test_data = {
            "test_queries": [
                {
                    "query": "¿Qué es la presunción de inocencia?",
                    "expected_keywords": ["inocencia", "presunción"],
                    "expected_type": "definicion"
                },
                {
                    "query": "¿Cómo se realiza una audiencia de imputación?",
                    "expected_keywords": ["audiencia", "imputación"],
                    "expected_type": "procedimiento"
                },
                {
                    "query": "¿Qué establece el Artículo 8 del CPP?",
                    "expected_keywords": ["artículo", "8"],
                    "expected_type": "normativa"
                },
                {
                    "query": "¿Cuáles son los derechos de las víctimas?",
                    "expected_keywords": ["derechos", "víctimas"],
                    "expected_type": "derechos"
                },
                {
                    "query": "¿Cuándo prescribe un delito menor?",
                    "expected_keywords": ["prescribe", "delito", "menor"],
                    "expected_type": "plazos"
                },
                {
                    "query": "¿Qué requisitos hay para solicitar libertad condicional?",
                    "expected_keywords": ["requisitos", "libertad", "condicional"],
                    "expected_type": "requisitos"
                }
            ]
        }
        
        with open(test_data_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        print(f"Archivo de datos de prueba creado en {test_data_file}")

if __name__ == "__main__":
    # Crear datos de prueba si no existen
    create_test_data()
    
    # Ejecutar pruebas
    unittest.main()