import requests
import json
import re
import datetime
import threading  # Para hilos en segundo plano
import time  # Para sleeps y timestamps
from io import BytesIO  # NUEVO: Para manejar bytes de imágenes descargadas
from PIL import Image  # NUEVO: Para procesar imágenes
import torch  # NUEVO: Para BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration  # NUEVO: Para BLIP

# Configuración
LM_API = "http://localhost:1234/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b"
DUCK_API = "https://api.duckduckgo.com/"

# NUEVO: Variables globales para BLIP (cargan lazy)
processor = None
model = None

# Lista de fuentes confiables de distintos enfoques
safe_sites = [
    "reuters.com",
    "theguardian.com",
    "democracynow.org",
    "eldiario.es",
    "wsj.com",
    "foxnews.com",
    "dailymail.co.uk",
    "infobae.com",
    "abc.es",
    "elmundo.es",
    "elpais.com",
    "ambito.com",
    "lanacion.com.ar",
    "perfil.com",
    "pagina12.com.ar",
    "bbc.com",
    "nytimes.com",
    "clarin.com",
    "*.gov",
    "*.edu",
    "*.org"
]

# Clasificación política aproximada de medios
source_bias = {
    "reuters.com": "centro",
    "theguardian.com": "izquierda",
    "democracynow.org": "izquierda",
    "eldiario.es": "izquierda",
    "wsj.com": "derecha",
    "foxnews.com": "derecha",
    "dailymail.co.uk": "derecha",
    "infobae.com": "centro",
    "abc.es": "derecha",
    "elmundo.es": "derecha",
    "elpais.com": "izquierda",
    "ambito.com": "centro",
    "lanacion.com.ar": "derecha",
    "bbc.com": "centro",
    "nytimes.com": "izquierda",
    "clarin.com": "derecha",
    "*.gov": "oficial",
    "*.edu": "académico",
    "*.org": "organización",
    "pagina12.com.ar": "izquierda",
    "perfil.com": "centro-derecha"
}

# ---- Memoria y Config ----
def load_memory():
    try:
        with open("memory.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_memory(memory):
    with open("memory.json", "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

def load_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            return [{"role": "system", "content": config["persona"]}]
    except:
        return []

def clear_memory():
    with open("memory.json", "w", encoding="utf-8") as f:
        json.dump([], f)
    print("🗑 Memoria borrada.")

# Funciones para Knowledge Base
def load_knowledge_base():
    try:
        with open("knowledge_base.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_knowledge_base(kb):
    with open("knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)
        

def clean_knowledge_base():
    kb = load_knowledge_base()
    now = datetime.datetime.now()
    kb = [entry for entry in kb if (now - datetime.datetime.fromisoformat(entry["timestamp"])).days < 30]
    save_knowledge_base(kb)

def add_to_knowledge_base(entry):
    kb = load_knowledge_base()
    # Verificar duplicado simple (por contenido)
    if not any(entry["content"] in existing["content"] for existing in kb):
        entry["timestamp"] = datetime.datetime.now().isoformat()
        kb.append(entry)
        save_knowledge_base(kb)
        print(f"Agregado a KB: {entry['query'][:50]}...")

# ---- Búsqueda balanceada ----
def contains_recent_keywords(text):
    current_year = datetime.datetime.now().year
    keywords = [
        "2023", str(current_year), "actual", "nuevo", "reciente",
        "último", "ahora", "este año", "hoy día"
    ]
    return any(re.search(rf'\b{k}\b', text.lower()) for k in keywords)

def should_search(prompt, memory):
    if contains_recent_keywords(prompt):
        return True
    for msg in reversed(memory):
        if msg["role"] == "system" and "Datos de búsqueda" in msg["content"]:
            stored_data = msg["content"]
            if prompt.lower() in stored_data.lower():
                return False
    keywords = [
        "quién", "qué es", "cuándo", "dónde", "último",
        "noticias", "precio", "definición", "significado", "actual"
    ]
    return any(k in prompt.lower() for k in keywords)

def duck_search(query):
    site_filter = " OR ".join([f"site:{site}" for site in safe_sites])
    params = {
        "q": f"{query} ({site_filter}) -2020..2022",
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1
    }
    try:
        resp = requests.get(DUCK_API, params=params, timeout=10)
        data = resp.json()
        results = []

        if data.get("AbstractText"):
            source = data.get("AbstractURL", "")
            bias = detect_bias(source)
            results.append(f"[{bias}] {data['AbstractText']} (Fuente: {source})")

        if data.get("RelatedTopics"):
            for topic in data["RelatedTopics"][:5]:
                if "Text" in topic and any(site in topic.get("FirstURL", "") for site in safe_sites):
                    bias = detect_bias(topic.get("FirstURL", ""))
                    results.append(f"[{bias}] {topic['Text']} (Fuente: {topic.get('FirstURL', '')})")

        return results if results else ["No encontré información actualizada en las fuentes confiables definidas."]
    except Exception as e:
        return [f"Error en búsqueda: {str(e)}"]

def detect_bias(url):
    for domain, bias in source_bias.items():
        if domain in url:
            return bias
    return "desconocido"

# NUEVO: Funciones para BLIP (reconocimiento de imágenes)
def load_blip_model():
    global processor, model
    if processor is None:
        print("Cargando modelo BLIP por primera vez (paciencia)...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("BLIP listo.")

def describe_image(image_url):
    load_blip_model()
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_length=50, num_beams=5)  # Beam search para mejor caption
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f" Error procesando imagen {image_url}: {str(e)}")
        return "No pude describir la imagen (URL inválida o error de descarga)."

def has_image_url(text):
    # Regex para detectar URLs de imágenes comunes
    pattern = r'https?://\S+\.(jpg|jpeg|png|gif|bmp|webp)\b'
    return bool(re.search(pattern, text, re.IGNORECASE))

# Función para actualización periódica
def periodic_update(config):
    interests = config.get("interests", [])
    update_interval = config.get("update_interval", 60)  # Minutos
    while True:  # Loop infinito en background
        print(f"Iniciando actualización periódica (cada {update_interval} min)...")
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)  # Recargar config por si cambia
        except:
            config = {"interests": [], "update_interval": 60}
        
        for interest in interests[:3]:  # Limitar a 3 por ciclo para no sobrecargar (puedes ajustar)
            results = duck_search(interest)
            if results and "No encontré" not in results[0]:
                entry = {
                    "query": interest,
                    "content": "\n".join(results),
                    "sources": [r for r in results if "(Fuente:" in r]
                }
                add_to_knowledge_base(entry)
        
        print(f"Actualización completada. Esperando {update_interval} minutos...")
        time.sleep(update_interval * 60)  # Sleep en segundos

# ---- Generación con IA (actualizada con KB e imágenes) ----
def generate(prompt, memory, config):  # Agregué config como param
    search_results = None
    if should_search(prompt, memory):
        search_results = duck_search(prompt)
        if search_results and "No encontré" not in search_results[0]:
            prompt = (
                f"CONTEXTO ACTUALIZADO (post-2023):\n"
                + "\n".join(search_results)
                + "\n\nCONOCIMIENTO BASE (pre-2023):\n[Modelo: {MODEL_NAME}]\n\n"
                + "Resume equilibrando las perspectivas según la ideología entre corchetes. "
                  "Aclara si hay diferencias significativas y cita las fuentes.\n\n"
                  f"Pregunta: {prompt}"
            )

    # Inyectar conocimiento relevante de KB
    kb = load_knowledge_base()
    relevant_kb = []
    for entry in kb[-5:]:  # Últimas 5 entradas para no sobrecargar
        if any(keyword in prompt.lower() for keyword in entry["query"].split()):
            relevant_kb.append(entry["content"])
    if relevant_kb:
        prompt += f"\n\nCONOCIMIENTO RELEVANTE DE KB (actualizado): {chr(10).join(relevant_kb[:2])}"  # Limitar a 2

    # NUEVO: Procesar imágenes si hay URLs en el prompt
    if has_image_url(prompt):
        image_urls = re.findall(r'https?://\S+\.(jpg|jpeg|png|gif|bmp|webp)\b', prompt, re.IGNORECASE)
        for url in image_urls[:2]:  # Limitar a 2 imágenes por prompt para no sobrecargar
            desc = describe_image(url)
            prompt += f"\n\n[Descripción de imagen ({url}): {desc}]"
            print(f"🖼️ Imagen procesada: {desc}")

    full_context = memory + [{"role": "user", "content": prompt}]
    data = {
        "model": MODEL_NAME,
        "messages": full_context,
        "temperature": 0.7,
        "max_tokens": 800,
        "stream": False
    }

    response = requests.post(LM_API, headers=HEADERS, json=data)
    output = response.json()
    answer = output["choices"][0]["message"]["content"]

    memory.append({"role": "user", "content": prompt})
    memory.append({"role": "assistant", "content": answer})
    if search_results:
        memory.append({"role": "system", "content": f"Datos de búsqueda (post-2023): {search_results}"})
    save_memory(memory)

    return answer

 #---- Consola (actualizada con /rate) ----
def loji_console():
    print("Loji 1.3 ('/exit' para salir, '/clear' para borrar memoria, '/rate' para calificar respuesta)")
    
    # Cargar config para interests y interval
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
    except:
        config = {"interests": [], "update_interval": 60}
    
    memory = load_config() + load_memory()
    
    # Iniciar hilo de actualización en background
    update_thread = threading.Thread(target=periodic_update, args=(config,), daemon=True)
    update_thread.start()
    print("Hilo de actualización iniciado (cada 60 min).")

    while True:
        user_input = input("Tú: ")
        if user_input.lower() == "/exit":
            break
        if user_input.lower() == "/clear":
            clear_memory()
            memory = load_config()
            continue
        if user_input.lower().startswith("/rate"):
            try:
                rating = user_input.split()[1].lower()
                if rating not in ["buena", "mala"]:
                    print("Loji: Por favor, usa '/rate buena' o '/rate mala'.")
                    continue
                last_response = memory[-1]["content"] if memory and memory[-1]["role"] == "assistant" else "No hay respuesta previa."
                last_query = memory[-2]["content"] if len(memory) >= 2 and memory[-2]["role"] == "user" else "No hay pregunta previa."
                add_to_knowledge_base({
                    "query": "feedback",
                    "content": f"Calificación: {rating}, Pregunta: {last_query}, Respuesta: {last_response}"
                })
                print(f"Loji: Gracias por calificar la respuesta como '{rating}'. ¡Guardado en la base de conocimientos!")
            except IndexError:
                print("Loji: Por favor, usa '/rate buena' o '/rate mala'.")
            continue
        reply = generate(user_input, memory, config)  # Pasar config
        print(f"Loji: {reply}\n")

if __name__ == "__main__":
    loji_console()