import pandas as pd
import numpy as np
import json
import os
import logging
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import torch
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pickle
import time

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('legal_model_trainer')

# Descargar recursos de NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    spanish_stopwords = set(stopwords.words('spanish'))
except Exception as e:
    logger.warning(f"Error al descargar recursos NLTK: {str(e)}")
    spanish_stopwords = set()

class LegalModelTrainer:
    """Entrenador para modelos de procesamiento de texto en el dominio legal."""
    
    def __init__(self, data_path, output_dir="models"):
        """
        Inicializa el entrenador.
        
        Args:
            data_path: Ruta al archivo JSON con la base de conocimiento
            output_dir: Directorio donde se guardarán los modelos entrenados
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.ensure_output_dir()
        
        # Cargar datos
        self.legal_kb = self.load_data()
        
        # Preparar datos de entrenamiento
        self.prepare_training_data()
        
        # Inicializar modelos
        self.tfidf_vectorizer = None
        self.domain_classifier = None
        self.sentence_transformer = None
    
    def ensure_output_dir(self):
        """Crear directorio de salida si no existe."""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            logger.info(f"Directorio de salida creado: {self.output_dir}")
    
    def load_data(self):
        """Cargar datos desde archivo JSON."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                legal_kb = json.load(f)
            logger.info(f"Datos cargados: {len(legal_kb)} registros")
            return legal_kb
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            # Devolver datos de ejemplo para evitar fallos
            return [
                {"question": "¿Qué es un contrato?", "answer": "Un contrato es un acuerdo legal..."},
                {"question": "¿Qué es un delito?", "answer": "Un delito es una acción u omisión típica..."}
            ]
    
    def prepare_training_data(self):
        """Preparar datos para entrenamiento."""
        self.query_data = []
        self.query_types = []
        self.query_pairs = []
        
        # Categorías legales para clasificación
        categories = {
            "definicion": ["qué es", "definición", "concepto", "significa"],
            "procedimiento": ["cómo", "procedimiento", "pasos", "trámite"],
            "normativa": ["artículo", "ley", "norma", "establece"],
            "derechos": ["derecho", "garantía", "protección"],
            "plazos": ["plazo", "término", "cuando", "tiempo"],
            "requisitos": ["requisito", "necesario", "condición"]
        }
        
        try:
            # Crear pares de consultas y categorías
            for item in self.legal_kb:
                if not isinstance(item, dict) or 'question' not in item or 'answer' not in item:
                    continue
                
                question = item['question']
                answer = item['answer']
                
                # Determinar categoría
                category = "otros"
                for cat, keywords in categories.items():
                    if any(kw in question.lower() for kw in keywords):
                        category = cat
                        break
                
                # Para entrenamiento de clasificador
                self.query_data.append(question)
                self.query_types.append(category)
                
                # Para entrenamiento de embeddings
                self.query_pairs.append({
                    "query": question,
                    "answer": answer,
                    "category": category
                })
            
            logger.info(f"Datos preparados: {len(self.query_data)} consultas")
        except Exception as e:
            logger.error(f"Error preparando datos: {str(e)}")
    
    def preprocess_text(self, text):
        """Preprocesar texto para entrenamiento."""
        try:
            # Convertir a minúsculas
            text = text.lower()
            
            # Eliminar caracteres especiales
            text = re.sub(r'[^\w\sáéíóúüñ¿?¡!.,;]', '', text)
            
            # Tokenizar
            tokens = word_tokenize(text)
            
            # Eliminar stopwords
            filtered_tokens = [t for t in tokens if t not in spanish_stopwords and len(t) > 2]
            
            return ' '.join(filtered_tokens)
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {str(e)}")
            return text
    
    def train_tfidf_vectorizer(self):
        """Entrenar vectorizador TF-IDF."""
        try:
            logger.info("Entrenando vectorizador TF-IDF...")
            
            # Preprocesar texto
            processed_data = [self.preprocess_text(text) for text in self.query_data]
            
            # Crear y entrenar vectorizador
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.85
            )
            
            # Ajustar vectorizador
            self.tfidf_vectorizer.fit(processed_data)
            
            # Guardar modelo
            tfidf_path = self.output_dir / 'tfidf_vectorizer.pkl'
            with open(tfidf_path, 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            
            vocab_size = len(self.tfidf_vectorizer.vocabulary_)
            logger.info(f"Vectorizador TF-IDF entrenado: {vocab_size} términos en vocabulario")
            logger.info(f"Modelo guardado en {tfidf_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error entrenando vectorizador TF-IDF: {str(e)}")
            return False
    
    def train_query_classifier(self):
        """Entrenar clasificador de tipos de consulta."""
        try:
            logger.info("Entrenando clasificador de consultas...")
            
            if self.tfidf_vectorizer is None:
                success = self.train_tfidf_vectorizer()
                if not success:
                    logger.error("No se pudo entrenar el vectorizador TF-IDF, abortando entrenamiento del clasificador")
                    return False
            
            # Preprocesar y vectorizar datos
            processed_data = [self.preprocess_text(text) for text in self.query_data]
            X = self.tfidf_vectorizer.transform(processed_data)
            y = self.query_types
            
            # Dividir en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entrenar clasificador
            self.domain_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1
            )
            
            self.domain_classifier.fit(X_train, y_train)
            
            # Evaluar
            y_pred = self.domain_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Clasificador entrenado con precisión: {accuracy:.4f}")
            
            # Informe detallado
            report = classification_report(y_test, y_pred)
            logger.info(f"Informe de clasificación:\n{report}")
            
            # Guardar modelo
            classifier_path = self.output_dir / 'query_classifier.pkl'
            with open(classifier_path, 'wb') as f:
                pickle.dump(self.domain_classifier, f)
            
            logger.info(f"Clasificador guardado en {classifier_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error entrenando clasificador: {str(e)}")
            return False
    
    def create_sentence_pairs(self):
        """Crear pares de oraciones para entrenamiento de transformer."""
        try:
            pairs = []
            
            # Dividir respuestas en oraciones
            for item in self.query_pairs:
                query = item['query']
                answer = item['answer']
                
                # Dividir respuesta en oraciones
                sentences = nltk.sent_tokenize(answer)
                
                if not sentences:
                    continue
                
                # Crear pares positivos (consulta-oración relevante)
                for sentence in sentences:
                    if len(sentence.split()) > 5:  # Ignorar oraciones muy cortas
                        pairs.append((query, sentence, 0.9))  # Alta similitud
                
                # Crear pares negativos (consulta-oración no relacionada)
                # Seleccionar oraciones de otras respuestas
                negative_sentences = []
                for other_item in self.query_pairs[:10]:  # Limitar búsqueda para eficiencia
                    if other_item['query'] != query:
                        other_sentences = nltk.sent_tokenize(other_item['answer'])
                        if other_sentences:
                            negative_sentences.append(other_sentences[0])
                
                for neg_sentence in negative_sentences[:3]:  # Tomar hasta 3 ejemplos negativos
                    pairs.append((query, neg_sentence, 0.1))  # Baja similitud
            
            logger.info(f"Pares de oraciones creados: {len(pairs)}")
            return pairs
        except Exception as e:
            logger.error(f"Error creando pares de oraciones: {str(e)}")
            return []
    
    def train_sentence_transformer(self, base_model="all-MiniLM-L6-v2", epochs=3):
        """Entrenar un modelo de transformers para embeddings de oraciones."""
        try:
            logger.info(f"Entrenando modelo de embeddings usando {base_model} como base...")
            
            # Crear pares de entrenamiento
            pairs = self.create_sentence_pairs()
            
            if not pairs:
                logger.error("No se pudieron crear pares de entrenamiento")
                return False
            
            # Crear ejemplos de entrenamiento
            train_examples = []
            for query, answer, score in pairs:
                train_examples.append(InputExample(texts=[query, answer], label=score))
            
            # Dividir en entrenamiento y evaluación
            train_size = int(0.8 * len(train_examples))
            train_data = train_examples[:train_size]
            eval_data = train_examples[train_size:]
            
            # Cargar modelo base
            self.sentence_transformer = SentenceTransformer(base_model)
            
            # Configurar entrenamiento
            train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
            train_loss = losses.CosineSimilarityLoss(model=self.sentence_transformer)
            
            # Evaluador
            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(eval_data)
            
            # Entrenar
            warmup_steps = int(len(train_dataloader) * 0.1)
            logger.info(f"Iniciando entrenamiento por {epochs} épocas...")
            
            self.sentence_transformer.fit(
                train_objectives=[(train_dataloader, train_loss)],
                evaluator=evaluator,
                epochs=epochs,
                warmup_steps=warmup_steps,
                evaluation_steps=len(train_dataloader)
            )
            
            # Guardar modelo
            model_path = self.output_dir / 'legal_sentence_transformer'
            self.sentence_transformer.save(str(model_path))
            
            logger.info(f"Modelo de embeddings guardado en {model_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error entrenando modelo de embeddings: {str(e)}")
            return False
    
    def train_all_models(self):
        """Entrenar todos los modelos."""
        logger.info("Iniciando entrenamiento de todos los modelos...")
        
        success = True
        
        # 1. Entrenar vectorizador TF-IDF
        tfidf_success = self.train_tfidf_vectorizer()
        if not tfidf_success:
            logger.warning("No se pudo entrenar el vectorizador TF-IDF")
            success = False
        
        # 2. Entrenar clasificador de consultas
        classifier_success = self.train_query_classifier()
        if not classifier_success:
            logger.warning("No se pudo entrenar el clasificador de consultas")
            success = False
        
        # 3. Entrenar modelo de embeddings
        transformer_success = self.train_sentence_transformer()
        if not transformer_success:
            logger.warning("No se pudo entrenar el modelo de embeddings")
            success = False
        
        if success:
            logger.info("Todos los modelos entrenados correctamente")
        else:
            logger.warning("Algunos modelos no se pudieron entrenar")
        
        return success
    
    def analyze_data_distribution(self):
        """Analizar la distribución de los datos y crear visualizaciones."""
        try:
            logger.info("Analizando distribución de datos...")
            
            # Distribución de categorías
            category_counts = Counter(self.query_types)
            
            # Longitudes de preguntas y respuestas
            question_lengths = [len(item['question'].split()) for item in self.query_pairs]
            answer_lengths = [len(item['answer'].split()) for item in self.query_pairs]
            
            # Crear visualizaciones
            plt.figure(figsize=(15, 10))
            
            # Distribución de categorías
            plt.subplot(2, 2, 1)
            categories = list(category_counts.keys())
            counts = list(category_counts.values())
            plt.bar(categories, counts)
            plt.title('Distribución de Categorías')
            plt.xticks(rotation=45)
            
            # Distribución de longitudes de preguntas
            plt.subplot(2, 2, 2)
            plt.hist(question_lengths, bins=20)
            plt.title('Longitud de Preguntas (palabras)')
            
            # Distribución de longitudes de respuestas
            plt.subplot(2, 2, 3)
            plt.hist(answer_lengths, bins=20)
            plt.title('Longitud de Respuestas (palabras)')
            
            # Guardar figura
            plt.tight_layout()
            analysis_path = self.output_dir / 'data_distribution.png'
            plt.savefig(analysis_path)
            plt.close()
            
            logger.info(f"Análisis de distribución guardado en {analysis_path}")
            
            # Estadísticas descriptivas
            stats = {
                "total_samples": len(self.query_pairs),
                "category_distribution": category_counts,
                "question_length": {
                    "min": min(question_lengths),
                    "max": max(question_lengths),
                    "avg": sum(question_lengths) / len(question_lengths)
                },
                "answer_length": {
                    "min": min(answer_lengths),
                    "max": max(answer_lengths),
                    "avg": sum(answer_lengths) / len(answer_lengths)
                }
            }
            
            # Guardar estadísticas
            stats_path = self.output_dir / 'data_stats.json'
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Estadísticas guardadas en {stats_path}")
            
            return stats
        except Exception as e:
            logger.error(f"Error analizando distribución de datos: {str(e)}")
            return None
    
    def export_processed_data(self):
        """Exportar datos procesados para uso en otros sistemas."""
        try:
            # Preprocesar todas las preguntas y respuestas
            processed_data = []
            
            for item in self.query_pairs:
                processed_item = {
                    "original_question": item['query'],
                    "processed_question": self.preprocess_text(item['query']),
                    "original_answer": item['answer'],
                    "processed_answer": self.preprocess_text(item['answer']),
                    "category": item['category']
                }
                processed_data.append(processed_item)
            
            # Guardar datos procesados
            output_path = self.output_dir / 'processed_legal_data.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Datos procesados exportados a {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exportando datos procesados: {str(e)}")
            return False

def main():
    """Función principal para ejecutar el entrenamiento de modelos."""
    print("\n=== ENTRENADOR DE MODELOS PARA DOMINIO LEGAL ===\n")
    
    # Parsear argumentos
    import argparse
    parser = argparse.ArgumentParser(description='Entrena modelos para procesamiento de texto legal')
    parser.add_argument('--data', type=str, default='data/legal_db.json', help='Ruta al archivo JSON de base de conocimiento')
    parser.add_argument('--output', type=str, default='models', help='Directorio para guardar modelos')
    parser.add_argument('--analyze', action='store_true', help='Analizar distribución de datos')
    parser.add_argument('--export', action='store_true', help='Exportar datos procesados')
    parser.add_argument('--tfidf', action='store_true', help='Entrenar solo vectorizador TF-IDF')
    parser.add_argument('--classifier', action='store_true', help='Entrenar solo clasificador')
    parser.add_argument('--transformer', action='store_true', help='Entrenar solo modelo de embeddings')
    parser.add_argument('--all', action='store_true', help='Entrenar todos los modelos')
    parser.add_argument('--epochs', type=int, default=3, help='Número de épocas para entrenar transformer')
    
    args = parser.parse_args()
    
    # Si no hay datos, usar ruta relativa
    if not os.path.exists(args.data):
        # Buscar en directorio actual, /data y ../data
        search_paths = [
            args.data,
            os.path.join(os.getcwd(), args.data),
            os.path.join(os.getcwd(), 'data', os.path.basename(args.data)),
            os.path.join(os.getcwd(), '..', 'data', os.path.basename(args.data))
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                args.data = path
                break
        else:
            logger.error(f"No se encontró el archivo de datos en las rutas buscadas")
            print("ERROR: No se pudo encontrar el archivo de datos. Especifica la ruta correcta con --data")
            return
    
    # Inicializar entrenador
    trainer = LegalModelTrainer(args.data, args.output)
    
    # Ejecutar acciones solicitadas
    if args.analyze:
        print("Analizando distribución de datos...")
        stats = trainer.analyze_data_distribution()
        if stats:
            print(f"Análisis completado. {stats['total_samples']} muestras analizadas.")
            print(f"Distribución de categorías: {stats['category_distribution']}")
            print(f"Longitud promedio de preguntas: {stats['question_length']['avg']:.1f} palabras")
            print(f"Longitud promedio de respuestas: {stats['answer_length']['avg']:.1f} palabras")
    
    if args.export:
        print("Exportando datos procesados...")
        success = trainer.export_processed_data()
        if success:
            print("Datos exportados correctamente")
    
    # Entrenar modelos
    specific_training = args.tfidf or args.classifier or args.transformer
    
    if args.all or not specific_training:
        print("Entrenando todos los modelos...")
        trainer.train_all_models()
    else:
        if args.tfidf:
            print("Entrenando vectorizador TF-IDF...")
            trainer.train_tfidf_vectorizer()
        
        if args.classifier:
            print("Entrenando clasificador de consultas...")
            trainer.train_query_classifier()
        
        if args.transformer:
            print("Entrenando modelo de embeddings...")
            trainer.train_sentence_transformer(epochs=args.epochs)
    
    print("\n=== ENTRENAMIENTO FINALIZADO ===\n")

if __name__ == "__main__":
    try:
        start_time = time.time()
        main()
        elapsed_time = time.time() - start_time
        print(f"Tiempo total de ejecución: {elapsed_time:.2f} segundos")
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        logger.exception("Error en ejecución principal")