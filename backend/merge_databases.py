import json
import os
from pathlib import Path

def merge_legal_databases():
    """
    Combina múltiples bases de datos legales en un solo archivo.
    """
    # Definir rutas
    data_dir = Path("data")
    
    # Archivos a combinar
    files_to_merge = [
        data_dir / "legal_db.json",
        data_dir / "legal_db_extension.json", 
        data_dir / "legal_db_extension_civil.json"
    ]
    
    # Lista para almacenar todos los datos
    combined_data = []
    total_items = 0
    
    print("=== MERGE DE BASES DE DATOS LEGALES ===\n")
    
    # Procesar cada archivo
    for file_path in files_to_merge:
        if file_path.exists():
            try:
                print(f"Procesando: {file_path.name}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Verificar que sea una lista
                if isinstance(data, list):
                    items_count = len(data)
                    combined_data.extend(data)
                    total_items += items_count
                    print(f"  ✓ Agregados {items_count} elementos")
                else:
                    print(f"  ⚠ Formato inesperado en {file_path.name} (no es una lista)")
                    
            except json.JSONDecodeError as e:
                print(f"  ✗ Error al leer {file_path.name}: {e}")
            except Exception as e:
                print(f"  ✗ Error inesperado con {file_path.name}: {e}")
        else:
            print(f"Archivo no encontrado: {file_path.name}")
    
    print(f"\nTotal de elementos combinados: {total_items}")
    
    # Eliminar duplicados (opcional)
    print("Eliminando duplicados...")
    unique_data = []
    seen_questions = set()
    
    for item in combined_data:
        if isinstance(item, dict) and 'question' in item:
            question = item['question'].strip().lower()
            if question not in seen_questions:
                seen_questions.add(question)
                unique_data.append(item)
            else:
                print(f"  Duplicado encontrado: {item['question'][:50]}...")
    
    final_count = len(unique_data)
    duplicates_removed = total_items - final_count
    
    print(f"Duplicados eliminados: {duplicates_removed}")
    print(f"Elementos finales: {final_count}")
    
    # Guardar archivo combinado
    output_file = data_dir / "legal_db.json"
    
    # Hacer backup del archivo original si existe
    if output_file.exists():
        backup_file = data_dir / "legal_db_backup.json"
        print(f"\nCreando backup: {backup_file.name}")
        os.rename(output_file, backup_file)
    
    # Guardar archivo combinado
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Base de datos combinada guardada en: {output_file.name}")
    print(f"✓ Total de documentos: {final_count}")
    
    # Mostrar estadísticas por tipo de pregunta
    print("\n=== ESTADÍSTICAS ===")
    question_types = {}
    for item in unique_data:
        if isinstance(item, dict) and 'question' in item:
            question = item['question'].lower()
            if 'artículo' in question:
                question_types['Artículos'] = question_types.get('Artículos', 0) + 1
            elif any(word in question for word in ['qué es', 'definición', 'concepto']):
                question_types['Definiciones'] = question_types.get('Definiciones', 0) + 1
            elif any(word in question for word in ['cómo', 'procedimiento', 'proceso']):
                question_types['Procedimientos'] = question_types.get('Procedimientos', 0) + 1
            elif any(word in question for word in ['derecho', 'garantía']):
                question_types['Derechos'] = question_types.get('Derechos', 0) + 1
            else:
                question_types['Otros'] = question_types.get('Otros', 0) + 1
    
    for tipo, cantidad in question_types.items():
        print(f"{tipo}: {cantidad}")
    
    return final_count

if __name__ == "__main__":
    try:
        count = merge_legal_databases()
        print(f"\n🎉 Proceso completado exitosamente!")
        print(f"🎉 Tu base de conocimiento ahora tiene {count} documentos")
        print("\n📝 Próximos pasos:")
        print("1. Reinicia tu servidor Flask: python app.py")
        print("2. Prueba algunas consultas para verificar que funciona")
        print("3. Los archivos originales están como backup por seguridad")
        
    except Exception as e:
        print(f"\n❌ Error durante el proceso: {e}")
        print("Verifica que todos los archivos JSON tengan el formato correcto")