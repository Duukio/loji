import requests
import json
import re
import datetime
import threading  # Para hilos en segundo plano
import time  # Para sleeps y timestamps
import os
import base64
from urllib.parse import urlparse, unquote
from io import BytesIO  # NUEVO: Para manejar bytes de imágenes descargadas
from PIL import Image  # NUEVO: Para procesar imágenes
import torch  # NUEVO: Para BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration  # NUEVO: Para BLIP

# Configuración
LM_API = "http://localhost:1234/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b"
DUCK_API = "https://api.duckduckgo.com/"

DEFAULT_CONFIG = {
    "persona": "",
    "interests": [],
    "update_interval": 60,
    "lm_api": LM_API,
    "model_name": MODEL_NAME,
    "max_tokens": 800,
    "temperature": 0.7,
    "stream": False,
    "search_enabled": True,
}

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

def load_settings():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
    except:
        config = {}

    normalized = DEFAULT_CONFIG.copy()
    normalized.update(config)
    if "intereses" in config and not config.get("interests"):
        normalized["interests"] = config["intereses"]
    return normalized

def load_config():
    config = load_settings()
    persona = config.get("persona", "").strip()
    if not persona:
        return []
    return [{"role": "system", "content": persona}]

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

def should_search(prompt, memory, config):
    if not config.get("search_enabled", True):
        return False
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

def load_image_from_source(source):
    if source.startswith("data:image/"):
        header, encoded = source.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(encoded))).convert('RGB')
    if source.startswith("file://"):
        parsed = urlparse(source)
        file_path = unquote(parsed.path)
        if os.name == "nt" and file_path.startswith("/"):
            file_path = file_path[1:]
        return Image.open(file_path).convert('RGB')
    if IMAGE_URL_PATTERN.match(source):
        response = requests.get(source, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    file_path = os.path.expanduser(source)
    if os.path.exists(file_path):
        return Image.open(file_path).convert('RGB')
    raise ValueError("Fuente de imagen no válida o inexistente")

def describe_image(source):
    load_blip_model()
    try:
        image = load_image_from_source(source)
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_length=50, num_beams=5)  # Beam search para mejor caption
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f" Error procesando imagen {source}: {str(e)}")
        return "No pude describir la imagen (fuente inválida o error de lectura)."

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")
IMAGE_URL_PATTERN = re.compile(r'https?://\S+\.(?:jpg|jpeg|png|gif|bmp|webp)\b', re.IGNORECASE)
IMAGE_DATA_PATTERN = re.compile(r'data:image/(?:jpg|jpeg|png|gif|bmp|webp);base64,[A-Za-z0-9+/=]+', re.IGNORECASE)

def extract_image_sources(text):
    sources = []
    sources.extend(match.group(0) for match in IMAGE_DATA_PATTERN.finditer(text))
    sources.extend(match.group(0) for match in IMAGE_URL_PATTERN.finditer(text))
    tokens = re.split(r"\s+", text)
    for token in tokens:
        cleaned = token.strip("\"'()[]{}<>,")
        if not cleaned or cleaned.startswith("http") or cleaned.startswith("data:image/"):
            continue
        if cleaned.lower().endswith(IMAGE_EXTENSIONS):
            path = os.path.expanduser(cleaned)
            if os.path.exists(path):
                sources.append(cleaned)
    return list(dict.fromkeys(sources))

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
            config = DEFAULT_CONFIG.copy()

        normalized = DEFAULT_CONFIG.copy()
        normalized.update(config)
        if "intereses" in config and not config.get("interests"):
            normalized["interests"] = config["intereses"]
        interests = normalized.get("interests", [])
        update_interval = normalized.get("update_interval", 60)
        
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
def safe_post(url, payload):
    try:
        response = requests.post(url, headers=HEADERS, json=payload, timeout=60)
        response.raise_for_status()
        return response.json(), None
    except Exception as e:
        return None, str(e)

def extract_answer(payload):
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None

def generate(prompt, memory, config):  # Agregué config como param
    search_results = None
    if should_search(prompt, memory, config):
        search_results = duck_search(prompt)
        if search_results and "No encontré" not in search_results[0]:
            prompt = (
                f"CONTEXTO ACTUALIZADO (post-2023):\n"
                + "\n".join(search_results)
                + f"\n\nCONOCIMIENTO BASE (pre-2023):\n[Modelo: {config.get('model_name', MODEL_NAME)}]\n\n"
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

    # NUEVO: Procesar imágenes si hay fuentes válidas en el prompt (URL, data URI o archivo local)
    image_sources = extract_image_sources(prompt)
    if image_sources:
        for source in image_sources[:2]:  # Limitar a 2 imágenes por prompt para no sobrecargar
            desc = describe_image(source)
            prompt += f"\n\n[Descripción de imagen ({source}): {desc}]"
            print(f"🖼️ Imagen procesada: {desc}")

    full_context = memory + [{"role": "user", "content": prompt}]
    data = {
        "model": config.get("model_name", MODEL_NAME),
        "messages": full_context,
        "temperature": config.get("temperature", 0.7),
        "max_tokens": config.get("max_tokens", 800),
        "stream": config.get("stream", False)
    }

    output, error = safe_post(config.get("lm_api", LM_API), data)
    answer = extract_answer(output) if output else None
    if not answer:
        return f"Lo siento, no pude obtener respuesta del modelo. Error: {error or 'respuesta inválida'}"

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
    config = load_settings()
    
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
