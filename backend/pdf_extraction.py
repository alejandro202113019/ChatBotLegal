import PyPDF2
import re
import json
import os
import nltk
from nltk.tokenize import sent_tokenize
from pathlib import Path
import argparse

# Descargar recursos necesarios de NLTK
try:
    nltk.download('punkt', quiet=True)
except:
    print("No se pudo descargar recursos de NLTK. Continuando...")


def extract_text_from_pdf(pdf_path):
    """Extrae todo el texto de un archivo PDF."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error al extraer texto del PDF: {str(e)}")
        return ""


def preprocess_text(text):
    """Limpia y preprocesa el texto extraído."""
    # Reemplazar saltos de línea múltiples
    text = re.sub(r'\n+', '\n', text)
    # Reemplazar espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    # Unir líneas que terminan con guión
    text = re.sub(r'-\s*\n', '', text)
    return text.strip()


def extract_articles(text):
    """Extrae artículos del código procesal penal."""
    articles = []

    # Patrón mejorado para encontrar artículos (considera formato real del PDF)
    article_pattern = r'(?:✿\s*)?ARTÍCULO\s+(\d+[o°\.]*)[\.\s]*(.*?)(?=(?:✿\s*)?ARTÍCULO\s+\d+[o°\.]|TÍTULO|CAPÍTULO|\Z)'

    matches = re.finditer(article_pattern, text, re.DOTALL | re.IGNORECASE)

    for match in matches:
        if len(match.groups()) >= 2:
            article_num = match.group(1).strip()
            content = match.group(2).strip()

            # Limpiar el contenido
            content = re.sub(r'\s+', ' ', content)
            content = re.sub(r'Jurisprudencia Vigencia', '', content)
            content = re.sub(r'Notas de Vigencia', '', content)
            content = re.sub(r'Legislación Anterior', '', content)

            if article_num and content:
                articles.append({
                    "article_num": article_num,
                    "content": content
                })

    return articles


def extract_topics(text):
    """Extrae títulos, capítulos y secciones del código procesal penal."""
    topics = []

    # Patrones mejorados para encontrar títulos, capítulos y secciones
    patterns = [
        (r'TITULO\s+(PRE|I[IVXLCDM]+|[0-9]+)[\.\s]*(.*?)(?=TITULO|CAPITULO|ARTÍCULO|\Z)', 'TÍTULO'),
        (r'CAPITULO\s+(I[IVXLCDM]+|[0-9]+)[\.\s]*(.*?)(?=TITULO|CAPITULO|ARTÍCULO|\Z)', 'CAPÍTULO'),
        (r'SECCIÓN\s+(I[IVXLCDM]+|[0-9]+)[\.\s]*(.*?)(?=TITULO|CAPITULO|SECCIÓN|ARTÍCULO|\Z)', 'SECCIÓN'),
        (r'LIBRO\s+(I[IVXLCDM]+|[0-9]+)[\.\s]*(.*?)(?=LIBRO|TITULO|CAPITULO|ARTÍCULO|\Z)', 'LIBRO')
    ]

    for pattern, topic_type in patterns:
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)

        for match in matches:
            if len(match.groups()) >= 2:
                number = match.group(1).strip()
                title = match.group(2).strip()

                # Limpiar el título
                title = re.sub(r'\s+', ' ', title)
                title = re.sub(r'\.$', '', title)

                if number and title:
                    topics.append({
                        "type": topic_type,
                        "number": number,
                        "title": title
                    })

    return topics


def generate_qa_from_articles(articles):
    """Genera pares de preguntas y respuestas a partir de los artículos."""
    qa_pairs = []

    for article in articles:
        article_num = article["article_num"]
        content = article["content"]

        # Generar preguntas estándar para cada artículo
        questions = [
            f"¿Qué establece el Artículo {article_num} del Código de Procedimiento Penal?",
            f"¿Cuál es el contenido del Artículo {article_num} del CPP?",
            f"¿Qué derechos o procedimientos regula el Artículo {article_num}?"
        ]

        # Detectar si el artículo trata sobre derechos fundamentales
        if any(keyword in content.lower() for keyword in ["derecho", "garantía", "principio"]):
            questions.append(f"¿Qué garantías procesales establece el Artículo {article_num}?")
            questions.append(f"¿Cómo protege el Artículo {article_num} los derechos fundamentales?")

        # Detectar si menciona procedimientos específicos
        proc_match = re.search(r'(procedimiento|actuación|proceso)\s+(.*?)\s+(?:será|deberá)', content)
        if proc_match:
            proc = proc_match.group(2).strip()
            questions.append(f"¿Cómo se regula el procedimiento de {proc} en el Artículo {article_num}?")

        # Añadir los pares de Q&A para este artículo
        for question in questions:
            qa_pairs.append({
                "question": question,
                "answer": f"El Artículo {article_num} del Código de Procedimiento Penal establece: {content}"
            })

    return qa_pairs


def generate_qa_from_topics(topics):
    """Genera pares de preguntas y respuestas a partir de los temas."""
    qa_pairs = []

    for topic in topics:
        topic_type = topic["type"]
        number = topic["number"]
        title = topic["title"]

        # Generar preguntas basadas en el tipo y título
        questions = [
            f"¿Qué regula el {topic_type} {number} del Código de Procedimiento Penal?",
            f"¿Cuál es el contenido del {topic_type} {number} titulado '{title}'?",
            f"¿Qué aspectos procesales cubre el {topic_type} {number}?"
        ]

        # Generar preguntas específicas basadas en el título
        if "principios" in title.lower():
            questions.append(f"¿Qué principios rectores establece el {topic_type} {number}?")

        if "derechos" in title.lower() or "garantías" in title.lower():
            questions.append(f"¿Qué garantías procesales contiene el {topic_type} {number}?")

        if "competencia" in title.lower():
            questions.append(f"¿Cómo se regula la competencia judicial en el {topic_type} {number}?")

        # Añadir los pares de Q&A para este tema
        for question in questions:
            qa_pairs.append({
                "question": question,
                "answer": f"El {topic_type} {number} del Código de Procedimiento Penal, titulado '{title}', regula los aspectos relacionados con este tema en el proceso penal."
            })

    return qa_pairs


def generate_procedural_qa(text):
    """Genera pares de Q&A sobre procedimientos penales específicos."""
    qa_pairs = []

    # Temas procedimentales clave en el CPP
    procedural_topics = [
        "investigación", "imputación", "acusación", "pruebas",
        "medidas cautelares", "audiencias", "recursos",
        "juzgamiento", "sentencia", "ejecución penal",
        "víctimas", "defensa técnica", "fiscalía",
        "juez de control de garantías", "principio de oralidad",
        "presunción de inocencia", "dignidad humana",
        "contradicción", "inmediación", "concentración"
    ]

    for topic in procedural_topics:
        # Buscar menciones del tema con más contexto
        pattern = rf'(?:✿\s*)?ARTÍCULO\s+\d+[o°\.]*[^.]*{topic}[^.]*\.'
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            context = match.group(0).strip()

            # Generar preguntas sobre este tema procedimental
            questions = [
                f"¿Cómo regula el CPP el {topic} en el proceso penal?",
                f"¿Qué establece el CPP sobre {topic}?",
                f"¿Cuáles son las normas sobre {topic} en el proceso penal?"
            ]

            for question in questions:
                if not any(qa["question"].lower() == question.lower() for qa in qa_pairs):
                    qa_pairs.append({
                        "question": question,
                        "answer": f"El Código de Procedimiento Penal establece lo siguiente sobre {topic}: {context}"
                    })

    return qa_pairs


def generate_rights_qa(text):
    """Genera pares de Q&A sobre derechos y garantías procesales."""
    qa_pairs = []

    # Derechos fundamentales en el proceso penal
    rights = [
        "dignidad humana", "libertad", "igualdad", "imparcialidad",
        "legalidad", "presunción de inocencia", "defensa",
        "intimidad", "contradicción", "doble instancia",
        "cosa juzgada", "lealtad", "gratuidad"
    ]

    for right in rights:
        # Buscar artículos que mencionen estos derechos
        pattern = rf'(?:✿\s*)?ARTÍCULO\s+(\d+[o°\.]*)[^.]*{right}[^.]*\.'
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            if match.groups():
                article_num = match.group(1).strip()
                context = match.group(0).strip()

                questions = [
                    f"¿Cómo protege el CPP el derecho a {right}?",
                    f"¿Qué garantías establece el Artículo {article_num} sobre {right}?",
                    f"¿En qué artículo del CPP se regula {right}?"
                ]

                for question in questions:
                    if not any(qa["question"].lower() == question.lower() for qa in qa_pairs):
                        qa_pairs.append({
                            "question": question,
                            "answer": f"El derecho a {right} en el proceso penal está regulado en el Artículo {article_num} del CPP, que establece: {context}"
                        })

    return qa_pairs


def clean_qa_pairs(qa_pairs):
    """Limpia y mejora los pares de Q&A."""
    cleaned_pairs = []
    seen_questions = set()

    for pair in qa_pairs:
        question = pair["question"].strip()
        answer = pair["answer"].strip()

        # Normalizar la pregunta
        question = re.sub(r'\s+', ' ', question)
        if not question.startswith("¿"):
            question = "¿" + question
        if not question.endswith("?"):
            question = question + "?"

        # Normalizar la respuesta
        answer = re.sub(r'\s+', ' ', answer)
        answer = re.sub(r'(\.\s*){2,}', '. ', answer)

        # Verificar que no sea duplicado y tenga contenido suficiente
        normalized_q = question.lower()
        if (normalized_q not in seen_questions and
                len(question.split()) >= 4 and
                len(answer.split()) >= 8):
            seen_questions.add(normalized_q)
            cleaned_pairs.append({
                "question": question,
                "answer": answer
            })

    return cleaned_pairs


def save_to_json(qa_pairs, output_file):
    """Guarda los pares de Q&A en un archivo JSON."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        print(f"Archivo guardado exitosamente: {output_file}")
        return True
    except Exception as e:
        print(f"Error al guardar archivo JSON: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Extrae información del Código de Procedimiento Penal para crear una base de conocimiento')
    parser.add_argument('pdf_path', help='Ruta al archivo PDF del CPP')
    parser.add_argument('--output', '-o', default='data/legal_db.json', help='Archivo de salida JSON')

    args = parser.parse_args()

    # Verificar si el archivo existe
    if not os.path.exists(args.pdf_path):
        print(f"Error: El archivo {args.pdf_path} no existe.")
        return

    # Crear directorio de salida si no existe
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extraer texto del PDF
    print(f"Extrayendo texto de {args.pdf_path}...")
    text = extract_text_from_pdf(args.pdf_path)

    if not text:
        print("No se pudo extraer texto del PDF.")
        return

    # Preprocesar texto
    print("Preprocesando texto...")
    processed_text = preprocess_text(text)

    # Extraer artículos y temas
    print("Extrayendo artículos y estructura del CPP...")
    articles = extract_articles(processed_text)
    topics = extract_topics(processed_text)

    print(f"- Encontrados {len(articles)} artículos")
    print(f"- Encontrados {len(topics)} títulos/capítulos/secciones")

    # Generar pares de Q&A
    print("Generando pares de preguntas y respuestas...")
    qa_pairs = []

    # 1. Q&A basados en artículos
    article_qa = generate_qa_from_articles(articles)
    print(f"- Generados {len(article_qa)} pares desde artículos")
    qa_pairs.extend(article_qa)

    # 2. Q&A basados en temas
    topic_qa = generate_qa_from_topics(topics)
    print(f"- Generados {len(topic_qa)} pares desde títulos/capítulos")
    qa_pairs.extend(topic_qa)

    # 3. Q&A sobre procedimientos específicos
    procedural_qa = generate_procedural_qa(processed_text)
    print(f"- Generados {len(procedural_qa)} pares sobre procedimientos")
    qa_pairs.extend(procedural_qa)

    # 4. Q&A sobre derechos y garantías
    rights_qa = generate_rights_qa(processed_text)
    print(f"- Generados {len(rights_qa)} pares sobre derechos procesales")
    qa_pairs.extend(rights_qa)

    # Limpiar y mejorar pares de Q&A
    print("Limpiando y mejorando pares de Q&A...")
    qa_pairs = clean_qa_pairs(qa_pairs)

    print(f"Total de pares Q&A generados: {len(qa_pairs)}")

    # Guardar resultados
    print(f"Guardando {len(qa_pairs)} pares en {args.output}...")
    if save_to_json(qa_pairs, args.output):
        print("¡Proceso completado exitosamente!")
    else:
        print("Error al guardar los resultados.")


if __name__ == "__main__":
    main()