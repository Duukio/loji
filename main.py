import requests
import json
import re
import datetime
import threading  # Para hilos en segundo plano
import time  # Para sleeps y timestamps
import os
import base64
import random
from urllib.parse import urlparse, unquote
from io import BytesIO  # NUEVO: Para manejar bytes de imágenes descargadas
from PIL import Image  # NUEVO: Para procesar imágenes
import torch  # NUEVO: Para BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration  # NUEVO: Para BLIP

# Configuración
LM_API = "http://localhost:1234/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b"
SEARXNG_API = "http://localhost:8080/search"
last_llm_call = 0
LLM_MIN_INTERVAL = 1.5


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
    "searxng_api": SEARXNG_API,
    "search_cache_ttl": 30,  # minutos
    "knowledge_base_max_entries": 200,
    "vision_enabled": True,
    "search_backoff_base": 1.5,
    "search_max_retries": 3,
}

# NUEVO: Variables globales para BLIP (cargan lazy)
processor = None
model = None
search_cache = {}

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
]

source_bias = {
    "reuters.com": "centro",
    "theguardian.com": "izquierda",
    "democracynow.org": "izquierda",
    "eldiario.es": "izquierda",
    "wsj.com": "derecha",
    "foxnews.com": "derecha",
    "dailymail.co.uk": "derecha",
    "infobae.com": "derecha",
    "abc.es": "centro-derecha",
    "elmundo.es": "centro-derecha",
    "elpais.com": "centro-izquierda",
    "ambito.com": "centro-derecha",
    "Lanacion.com.ar": "centro-derecha",
    "perfil.com": "centro-izquierda",
    "pagina12.com.ar": "izquierda",
    "bbc.com": "centro",
    "nytimes.com": "centro-izquierda",
    "clarin.com": "centro-derecha",
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

def save_settings(config):
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

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

def add_to_knowledge_base(entry, config=None):
    kb = load_knowledge_base()
    # Verificar duplicado simple (por contenido)
    if not any(entry["content"] in existing["content"] for existing in kb):
        entry["timestamp"] = datetime.datetime.now().isoformat()
        kb.append(entry)
        max_entries = (config or load_settings()).get("knowledge_base_max_entries", 200)
        if max_entries and len(kb) > max_entries:
            kb = kb[-max_entries:]
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

def read_search_cache(query, config):
    ttl_minutes = config.get("search_cache_ttl", 30)
    cached = search_cache.get(query.lower())
    if not cached:
        return None
    timestamp, results = cached
    if (time.time() - timestamp) / 60 <= ttl_minutes:
        return results
    search_cache.pop(query.lower(), None)
    return None

def write_search_cache(query, results):
    search_cache[query.lower()] = (time.time(), results)

def should_cache_result(results):
    return results and "Error en búsqueda" not in results[0]

def searxng_search(query, config, force_refresh=False):
    if not force_refresh:
        cached = read_search_cache(query, config)
        if cached:
            return cached

    base_backoff = config.get("search_backoff_base", 1.5)
    max_retries = config.get("search_max_retries", 3)
    api_url = config.get("searxng_api", SEARXNG_API)

    params = {
        "q": f"{query} actualidad análisis",
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1,
        "language": "es"
    }

    last_error = None

    for attempt in range(max_retries):
        try:
            resp = requests.get(api_url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("results", [])[:7]:
                source = item.get("url", "")
                if not source or not any(site in source for site in safe_sites):
                    continue

                bias = detect_bias(source)
                title = item.get("title") or ""
                snippet = item.get("content") or item.get("snippet") or ""

                if title or snippet:
                    results.append(
                        f"[{bias}] {title} {snippet} (Fuente: {source})".strip()
                    )

            if not results:
                results = ["No encontré información actualizada en fuentes confiables."]

            if should_cache_result(results):
                write_search_cache(query, results)

            return results

        except Exception as e:
            last_error = e
            time.sleep(base_backoff * (2 ** attempt) + random.uniform(0, 0.5))

    return [f"Error en búsqueda: {last_error}"]


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
        out = model.generate(**inputs, max_length=50, num_beams=5)
        return processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        print(f"🖼️ Error procesando imagen {source}: {e}")
        return "No pude describir la imagen."


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")
IMAGE_URL_PATTERN = re.compile(r'https?://\S+\.(?:jpg|jpeg|png|gif|bmp|webp)\b', re.IGNORECASE)
IMAGE_DATA_PATTERN = re.compile(r'data:image/[a-zA-Z0-9+/;=,]+', re.IGNORECASE)



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
        print(f"⏱️  Iniciando actualización periódica (cada {update_interval} min)...")
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
            results = searxng_search(interest, normalized)
            if results and "No encontré" not in results[0]:
                entry = {
                    "query": interest,
                    "content": "\n".join(results),
                    "sources": [r for r in results if "(Fuente:" in r]
                }
                add_to_knowledge_base(entry, normalized)
        
        print(f"Actualización completada. Esperando {update_interval} minutos...")
        print(f"✅ Actualización completada. Esperando {update_interval} minutos...")
        time.sleep(update_interval * 60)  # Sleep en segundos

# ---- Generación con IA (actualizada con KB e imágenes) ----
def safe_post(url, payload):
    global last_llm_call
    wait = LLM_MIN_INTERVAL - (time.time() - last_llm_call)
    if wait > 0:
        time.sleep(wait)

    last_llm_call = time.time()

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
        search_results = searxng_search(prompt, config)
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
    if config.get ("vision_enabled", False):
        image_source = extract_image_sources (prompt)
        for source in image_source [:1]:  # Limitar a 1 imágen por prompt para no sobrecargar memoria
            desc = describe_image (source)
            prompt += f"\n\n[Descripción de imagen ({source}): {desc}]"
            print (f"🖼️ Imagen procesada: {desc}")
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
 #---- Consola (actualizada con comandos útiles) ----
def loji_console():
    print("Loji 1.4")
    print("Escribe '/help' para ver comandos disponibles.")
    
    # Cargar config para interests y interval
    config = load_settings()
    
    memory = load_config() + load_memory()
    
    # Iniciar hilo de actualización en background
    update_thread = threading.Thread(target=periodic_update, args=(config,), daemon=True)
    update_thread.start()
    print("Hilo de actualización iniciado (cada 60 min).")
    print("✅ Hilo de actualización iniciado.")

    def print_help():
        print(
            "\nComandos disponibles:\n"
            "/help                → mostrar lista de comandos\n"
            "/config              → ver config actual\n"
            "/interests           → ver intereses\n"
            "/interests set a,b   → reemplazar intereses\n"
            "/interests add <t>   → agregar interés\n"
            "/interests remove <t>→ quitar interés\n"
            "/refresh <tema>      → forzar búsqueda ahora de ese interés\n"
            "/vision on/off       → activar/desactivar BLIP\n"
            "/rate buena|mala     → calificar respuesta\n"
            "/clear               → borrar memoria\n"
            "/exit                → salir\n"
        )

    while True:
        user_input = input("Tú: ")
        if user_input.lower() == "/help":
            print_help()
            continue
        if user_input.lower() == "/exit":
            break
        if user_input.lower() == "/clear":
            clear_memory()
            memory = load_config()
            continue
        if user_input.lower() == "/config":
            print(json.dumps(config, indent=2, ensure_ascii=False))
            continue
        if user_input.lower().startswith("/vision"):
            parts = user_input.split()
            if len(parts) != 2 or parts[1].lower() not in ["on", "off"]:
                print("Loji: Usa '/vision on' o '/vision off'.")
                continue
            config["vision_enabled"] = parts[1].lower() == "on"
            save_settings(config)
            status = "activado" if config["vision_enabled"] else "desactivado"
            print(f"Loji: BLIP {status}.")
            continue
        if user_input.lower().startswith("/interests"):
            parts = user_input.split(maxsplit=2)
            if len(parts) == 1:
                print(f"Intereses actuales: {', '.join(config.get('interests', [])) or 'ninguno'}")
                continue
            action = parts[1].lower()
            value = parts[2].strip() if len(parts) > 2 else ""
            if action == "set" and value:
                config["interests"] = [v.strip() for v in value.split(",") if v.strip()]
                save_settings(config)
                print("Loji: Intereses actualizados.")
                continue
            if action == "add" and value:
                interests = config.get("interests", [])
                if value not in interests:
                    interests.append(value)
                config["interests"] = interests
                save_settings(config)
                print(f"Loji: Interés agregado: {value}")
                continue
            if action == "remove" and value:
                config["interests"] = [i for i in config.get("interests", []) if i != value]
                save_settings(config)
                print(f"Loji: Interés eliminado: {value}")
                continue
            print("Loji: Usa '/interests', '/interests set a,b', '/interests add <t>' o '/interests remove <t>'.")
            continue
        if user_input.lower().startswith("/refresh"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Loji: Usa '/refresh <tema>'.")
                continue
            topic = parts[1].strip()
            results = searxng_search(topic, config, force_refresh=True)
            if results and "No encontré" not in results[0]:
                entry = {
                    "query": topic,
                    "content": "\n".join(results),
                    "sources": [r for r in results if "(Fuente:" in r]
                }
                add_to_knowledge_base(entry, config)
                print("Loji: Búsqueda actualizada en la base de conocimientos.")
            else:
                print("Loji: No encontré resultados nuevos.")
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
                }, config)

                print(f"Loji: Gracias por calificar la respuesta como '{rating}'. ¡Guardado en la base de conocimientos!")
            except IndexError:
                print("Loji: Por favor, usa '/rate buena' o '/rate mala'.")
            continue
        reply = generate(user_input, memory, config)  # Pasar config
        print(f"Loji: {reply}\n")

if __name__ == "__main__":
    loji_console()