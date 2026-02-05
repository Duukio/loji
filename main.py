from csv import reader
import requests
import json
import re
import datetime
import threading
import time
import os
import base64
import random
import easyocr
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse, unquote, parse_qs

#Inicialización de OCR
ocr_reader = easyocr.Reader(['es', 'en'], gpu=False)

# Configuración
LM_API = "http://127.0.0.1:1234/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
MODEL_NAME = "local-model" 
DUCKDUCKGO_API = "https://api.duckduckgo.com"
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
    "DUCKDUCKGO_API": DUCKDUCKGO_API,
    "search_cache_ttl": 30,
    "knowledge_base_max_entries": 200,
    "vision_enabled": True,
    "search_backoff_base": 1.5,
    "search_max_retries": 3,
}

search_cache = {}

safe_sites = [
    "reuters.com", "theguardian.com", "democracynow.org", "eldiario.es",
    "wsj.com", "foxnews.com", "dailymail.co.uk", "infobae.com",
    "abc.es", "elmundo.es", "elpais.com",
]

source_bias = {
    "reuters.com": "neutral",
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
}



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

def strip_html(text):
    return re.sub(r"<[^>]+>", "", text or "").strip()

def normalize_duckduckgo_url(url):
    if not url:
        return url
    parsed = urlparse(url)
    if parsed.netloc == "duckduckgo.com" and parsed.path == "/l/":
        params = parse_qs(parsed.query)
        redirected = params.get("uddg", [None])[0]
        if redirected:
            return unquote(redirected)
    return url

def duckduckgo_search(query, config, force_refresh=False):
    if not force_refresh:
        cached = read_search_cache(query, config)
        if cached:
            return cached

    params = {
        "q": query,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
        "pretty": "1",
    }

    base_backoff = config.get("search_backoff_base", 1.5)
    max_retries = config.get("search_max_retries", 3)
    last_error = None

    for attempt in range(max_retries):
        try:
            resp = requests.get(DUCKDUCKGO_API, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            results = []

            if data.get("AbstractText"):
                source = data.get("AbstractURL", "Sin fuente")
                bias = detect_bias(source)
                results.append(f"[{bias}] {data['AbstractText']} (Fuente: {source})")

            if data.get("RelatedTopics"):
                for topic in data["RelatedTopics"][:6]:
                    if "Text" in topic:
                        url = topic.get("FirstURL", "")
                        if url:
                            bias = detect_bias(url)
                            text = topic["Text"]
                            results.append(f"[{bias}] {text} (Fuente: {url})")

            if not results:
                results = ["No encontré información relevante en DuckDuckGo."]

            if should_cache_result(results):
                write_search_cache(query, results)

            return results

        except Exception as e:
            last_error = e
            sleep_time = base_backoff * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(sleep_time)

    return [f"Error en búsqueda tras {max_retries} intentos: {str(last_error)}"]

def detect_bias(url):
    for domain, bias in source_bias.items():
        if domain in url:
            return bias
    return "desconocido"

def contains_recent_keywords(prompt):
    keywords = ["hoy", "último", "actual", "noticia", "2024", "2025"]
    return any(k in prompt.lower() for k in keywords)

def load_settings():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return DEFAULT_CONFIG.copy()

def save_settings(config):
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def load_config():
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    config = load_settings()
    persona = config.get("persona", "")
    if persona:
        return [{"role": "system", "content": f"Eres {persona}. Fecha actual: {current_date}."}]
    return []

def load_memory():
    try:
        with open("memory.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def clear_memory():
    with open("memory.json", "w", encoding="utf-8") as f:
        json.dump([], f)

def save_memory(memory):
    with open("memory.json", "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

def load_knowledge_base():
    try:
        with open("knowledge_base.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def add_to_knowledge_base(entry, config=None):
    #¿Por que tuve que torturme haciendo una IA? 
    kb = load_knowledge_base()
    
    # Añadir timestamp si no existe
    if "timestamp" not in entry:
        entry["timestamp"] = time.time()
    
    # 1. Evitar duplicados EXACTOS (mismo query + mismo contenido)
    new_query = entry.get("query", "").lower()
    new_content_hash = hash(entry.get("content", ""))
    
    to_remove = []
    for i, existing in enumerate(kb):
        existing_query = existing.get("query", "").lower()
        existing_content_hash = hash(existing.get("content", ""))
        
        #Elimina Duplicado exacto
        if new_query == existing_query and new_content_hash == existing_content_hash:
            print(f"[KB] Duplicado exacto ignorado: {new_query}")
            return
        
        # Reemplazar si query es IDÉNTICO (no similar)
        if new_query == existing_query:
            to_remove.append(i)
            print(f"[KB] Reemplazando entrada antigua: {new_query}")
    
    # Eliminar entradas a reemplazar (de atrás hacia adelante)
    for i in sorted(to_remove, reverse=True):
        del kb[i]
    
    # 2. Agregar nueva entrada
    kb.append(entry)
    
    # 3. Limpiar por tiempo (> 7 días)
    max_age_days = 7
    current_time = time.time()
    kb = [e for e in kb if current_time - e.get("timestamp", 0) < max_age_days * 86400]
    
    # 4. Limitar por cantidad
    max_entries = DEFAULT_CONFIG.get("knowledge_base_max_entries", 200)
    if len(kb) > max_entries:
        # Eliminar las más viejas primero
        kb.sort(key=lambda x: x.get("timestamp", 0))
        kb = kb[-max_entries:]
    
    # 5. Guardar
    with open("knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

def periodic_update(config):
    # Actualización periódica de datos según intereses del usuario.
    update_interval = config.get("update_interval", 60)
    while True:
        print(f"⏱️ Iniciando actualización periódica (cada {update_interval} min)...")
        
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
        except:
            config = DEFAULT_CONFIG.copy()
        
        interests = config.get("interests", [])
        kb = load_knowledge_base()
        
        # Obtener intereses que NO han sido buscados recientemente
        recent_queries = [entry.get("query", "").lower() for entry in kb[-5:]]  # Últimas 5
        interests_to_update = []
        
        for interest in interests[:3]:  # Limitar a 3
            if interest.lower() not in recent_queries:
                interests_to_update.append(interest)
            else:
                print(f"[Update] Saltando '{interest}' - ya en KB reciente")
        
        # Solo buscar intereses NO recientes
        for interest in interests_to_update:
            print(f"[Update] Buscando: {interest}")
            results = duckduckgo_search(interest, config)
            if results and "No encontré" not in results[0]:
                entry = {
                    "query": interest,
                    "content": "\n".join(results),
                    "sources": [r for r in results if "(Fuente:" in r],
                    "timestamp": time.time()
                }
                add_to_knowledge_base(entry, config)
                print(f"[Update] Agregado: {interest}")
            else:
                print(f"[Update] Sin resultados nuevos para: {interest}")
        
        print(f"✅ Actualización completada. Esperando {update_interval} minutos...")
        time.sleep(update_interval * 60)

def extract_image_urls(text):
    """Detecta más tipos de URLs de imágenes"""
    # Patrón base
    pattern1 = r'https?://[^\s<>"]+?\.(?:png|jpg|jpeg|gif|webp|bmp)(?:\?[^\s<>"]*)?'
    
    # Para Imgur (sin extensión)
    pattern2 = r'https?://i\.imgur\.com/[a-zA-Z0-9]+(?:\.\w+)?'
    
    # Para otros CDNs comunes
    pattern3 = r'https?://(?:cdn\.|media\.)[^\s]+\.(?:png|jpg|jpeg|gif)'
    
    all_patterns = f'({pattern1})|({pattern2})|({pattern3})'
    
    matches = re.findall(all_patterns, text, re.IGNORECASE)
    
    # Flatten y limpiar
    urls = []
    for match in matches:
        for group in match:
            if group:
                urls.append(group)
                break
    
    return list(set(urls))  # Eliminar duplicados

def extract_text_from_image_url(image_url):
    """Versión simplificada para Windows"""
    try:
        print(f"[OCR] Procesando imagen...")
        
        # Descarga SIMPLE (sin headers complicados)
        response = requests.get(image_url, timeout=10)
        
        if response.status_code != 200:
            return f"[Error HTTP {response.status_code}]"
        
        # Procesar imagen
        img = Image.open(BytesIO(response.content))
        
        # OCR
        results = ocr_reader.readtext(img, paragraph=True)
        
        if not results:
            return "[Sin texto legible]"
        
        # Extraer textos con buena confianza
        texts = []
        for bbox, text, confidence in results:
            if confidence > 0.3 and text.strip():
                texts.append(text.strip())
        
        if not texts:
            return "[Texto no claro]"
        
        full_text = " ".join(texts)
        if len(full_text) > 500:
            full_text = full_text[:500] + "..."
        
        return full_text
        
    except Exception as e:
        return f"[Error: {type(e).__name__}]"

def safe_post(url, payload):
    global last_llm_call
    wait = LLM_MIN_INTERVAL - (time.time() - last_llm_call)
    if wait > 0:
        time.sleep(wait)

    last_llm_call = time.time()

    try:
        #Aumentar timeout para Qwen3-VL
        response = requests.post(url, headers=HEADERS, json=payload, timeout=120)
        response.raise_for_status()
        return response.json(), None
    except Exception as e:
        return None, str(e)

def extract_answer(payload):
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None

def generate(user_prompt, memory, config):
    
    #Fecha y hora actual
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    date_info = f"hoy es {current_date}."
    
    if not any(msg["role"] == "system" and "Fecha actual" in msg["content"] for msg in memory):
        memory.insert(0, {"role": "system", "content": f"Fecha actual: {date_info}"})
    
    # --- KB ---
    kb = load_knowledge_base()
    relevant_kb = []
    user_words = set(user_prompt.lower().split())
    
    for entry in kb[-10:]:
        entry_query = entry.get("query", "").lower()
        entry_words = set(entry_query.split())
        if user_words & entry_words:
            relevant_kb.append(entry["content"])

    final_prompt = user_prompt
    if relevant_kb:
        final_prompt += (
            "\n\n[CONOCIMIENTO RELEVANTE DE KB]:\n"
            + "\n---\n".join(relevant_kb[:2])
        )

    # --- IMÁGENES CON OCR REAL ---
    image_urls = extract_image_urls(user_prompt)
    
    if image_urls and config.get("vision_enabled", True):
        print(f"\n🔍 [Loji] Detecté imagen: {image_urls[0][:50]}...")
        
        # EXTRAER TEXTO REAL CON OCR
        ocr_result = extract_text_from_image_url(image_urls[0])
        
        if not ocr_result.startswith("[Error") and not ocr_result.startswith("[La imagen"):
            # ✅ TEXTO EXTRAÍDO CON ÉXITO
            final_prompt += f"\n\n[TEXTO EXTRAÍDO DE LA IMAGEN]:\n{ocr_result}"
            print(f"✅ Texto extraído ({len(ocr_result)} caracteres)")
            
            # Reemplazar URL por marcador
            final_prompt = final_prompt.replace(image_urls[0], "[IMAGEN]")
        else:
            # ❌ OCR FALLÓ
            final_prompt += f"\n\n[Imagen detectada pero no contiene texto legible]"
            print(f"⚠️  {ocr_result}")
    
    # --- LLM CALL ---
    messages = memory.copy()
    messages.append({"role": "user", "content": final_prompt})

    payload = {
        "model": config.get("model_name", "local-model"),
        "messages": messages,
        "max_tokens": config.get("max_tokens", 800),
        "temperature": config.get("temperature", 0.7),
        "stream": False
    }

    response, error = safe_post(config.get("lm_api", LM_API), payload)
    if error:
        return f"Error al contactar LLM: {error}"

    answer = extract_answer(response)
    if not answer:
        return "No pude generar una respuesta."

    memory.append({"role": "user", "content": user_prompt})
    memory.append({"role": "assistant", "content": answer})
    save_memory(memory)

    return answer

def loji_console():
    print("Loji 1.5 - Asistente de IA con búsqueda web y visión por computadora")
    print("Escribe '/help' para ver comandos disponibles.")
    
    config = load_settings()
    memory = load_config() + load_memory()
    
    # ✅ Solo un hilo de actualización
    update_thread = threading.Thread(target=periodic_update, args=(config,), daemon=True)
    update_thread.start()
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
            "/vision on/off       → activar/desactivar Vision\n"
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
            print(f"Loji: Visión {status}.")
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
            results = duckduckgo_search(topic, config, force_refresh=True)
            if results and "No encontré" not in results[0]:
                entry = {
                    "query": topic,
                    "content": "\n".join(results),
                    "sources": [r for r in results if "(Fuente:" in r],
                    "timestamp": time.time()
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
                    "content": f"Calificación: {rating}, Pregunta: {last_query}, Respuesta: {last_response}",
                    "timestamp": time.time()
                }, config)
                print(f"Loji: Gracias por calificar la respuesta como '{rating}'.")
            except IndexError:
                print("Loji: Por favor, usa '/rate buena' o '/rate mala'.")
            continue
        reply = generate(user_input, memory, config)
        print(f"Loji: {reply}\n")

if __name__ == "__main__":
    loji_console()
from csv import reader
import requests
import json
import re
import datetime
import threading
import time
import os
import base64
import random
import easyocr
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse, unquote, parse_qs

#Inicialización de OCR
ocr_reader = easyocr.Reader(['es', 'en'], gpu=False)

# Configuración
LM_API = "http://127.0.0.1:1234/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
MODEL_NAME = "local-model" 
DUCKDUCKGO_API = "https://api.duckduckgo.com"
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
    "DUCKDUCKGO_API": DUCKDUCKGO_API,
    "search_cache_ttl": 30,
    "knowledge_base_max_entries": 200,
    "vision_enabled": True,
    "search_backoff_base": 1.5,
    "search_max_retries": 3,
}

search_cache = {}

safe_sites = [
    "reuters.com", "theguardian.com", "democracynow.org", "eldiario.es",
    "wsj.com", "foxnews.com", "dailymail.co.uk", "infobae.com",
    "abc.es", "elmundo.es", "elpais.com",
]

source_bias = {
    "reuters.com": "neutral",
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
}



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

def strip_html(text):
    return re.sub(r"<[^>]+>", "", text or "").strip()

def normalize_duckduckgo_url(url):
    if not url:
        return url
    parsed = urlparse(url)
    if parsed.netloc == "duckduckgo.com" and parsed.path == "/l/":
        params = parse_qs(parsed.query)
        redirected = params.get("uddg", [None])[0]
        if redirected:
            return unquote(redirected)
    return url

def duckduckgo_search(query, config, force_refresh=False):
    if not force_refresh:
        cached = read_search_cache(query, config)
        if cached:
            return cached

    params = {
        "q": query,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
        "pretty": "1",
    }

    base_backoff = config.get("search_backoff_base", 1.5)
    max_retries = config.get("search_max_retries", 3)
    last_error = None

    for attempt in range(max_retries):
        try:
            resp = requests.get(DUCKDUCKGO_API, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            results = []

            if data.get("AbstractText"):
                source = data.get("AbstractURL", "Sin fuente")
                bias = detect_bias(source)
                results.append(f"[{bias}] {data['AbstractText']} (Fuente: {source})")

            if data.get("RelatedTopics"):
                for topic in data["RelatedTopics"][:6]:
                    if "Text" in topic:
                        url = topic.get("FirstURL", "")
                        if url:
                            bias = detect_bias(url)
                            text = topic["Text"]
                            results.append(f"[{bias}] {text} (Fuente: {url})")

            if not results:
                results = ["No encontré información relevante en DuckDuckGo."]

            if should_cache_result(results):
                write_search_cache(query, results)

            return results

        except Exception as e:
            last_error = e
            sleep_time = base_backoff * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(sleep_time)

    return [f"Error en búsqueda tras {max_retries} intentos: {str(last_error)}"]

def detect_bias(url):
    for domain, bias in source_bias.items():
        if domain in url:
            return bias
    return "desconocido"

def contains_recent_keywords(prompt):
    keywords = ["hoy", "último", "actual", "noticia", "2024", "2025"]
    return any(k in prompt.lower() for k in keywords)

def load_settings():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return DEFAULT_CONFIG.copy()

def save_settings(config):
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def load_config():
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    config = load_settings()
    persona = config.get("persona", "")
    if persona:
        return [{"role": "system", "content": f"Eres {persona}. Fecha actual: {current_date}."}]
    return []

def load_memory():
    try:
        with open("memory.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def clear_memory():
    with open("memory.json", "w", encoding="utf-8") as f:
        json.dump([], f)

def save_memory(memory):
    with open("memory.json", "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

def load_knowledge_base():
    try:
        with open("knowledge_base.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def add_to_knowledge_base(entry, config=None):
    #¿Por que tuve que torturme haciendo una IA? 
    kb = load_knowledge_base()
    
    # Añadir timestamp si no existe
    if "timestamp" not in entry:
        entry["timestamp"] = time.time()
    
    # 1. Evitar duplicados EXACTOS (mismo query + mismo contenido)
    new_query = entry.get("query", "").lower()
    new_content_hash = hash(entry.get("content", ""))
    
    to_remove = []
    for i, existing in enumerate(kb):
        existing_query = existing.get("query", "").lower()
        existing_content_hash = hash(existing.get("content", ""))
        
        #Elimina Duplicado exacto
        if new_query == existing_query and new_content_hash == existing_content_hash:
            print(f"[KB] Duplicado exacto ignorado: {new_query}")
            return
        
        # Reemplazar si query es IDÉNTICO (no similar)
        if new_query == existing_query:
            to_remove.append(i)
            print(f"[KB] Reemplazando entrada antigua: {new_query}")
    
    # Eliminar entradas a reemplazar (de atrás hacia adelante)
    for i in sorted(to_remove, reverse=True):
        del kb[i]
    
    # 2. Agregar nueva entrada
    kb.append(entry)
    
    # 3. Limpiar por tiempo (> 7 días)
    max_age_days = 7
    current_time = time.time()
    kb = [e for e in kb if current_time - e.get("timestamp", 0) < max_age_days * 86400]
    
    # 4. Limitar por cantidad
    max_entries = DEFAULT_CONFIG.get("knowledge_base_max_entries", 200)
    if len(kb) > max_entries:
        # Eliminar las más viejas primero
        kb.sort(key=lambda x: x.get("timestamp", 0))
        kb = kb[-max_entries:]
    
    # 5. Guardar
    with open("knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

def periodic_update(config):
    # Actualización periódica de datos según intereses del usuario.
    update_interval = config.get("update_interval", 60)
    while True:
        print(f"⏱️ Iniciando actualización periódica (cada {update_interval} min)...")
        
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
        except:
            config = DEFAULT_CONFIG.copy()
        
        interests = config.get("interests", [])
        kb = load_knowledge_base()
        
        # Obtener intereses que NO han sido buscados recientemente
        recent_queries = [entry.get("query", "").lower() for entry in kb[-5:]]  # Últimas 5
        interests_to_update = []
        
        for interest in interests[:3]:  # Limitar a 3
            if interest.lower() not in recent_queries:
                interests_to_update.append(interest)
            else:
                print(f"[Update] Saltando '{interest}' - ya en KB reciente")
        
        # Solo buscar intereses NO recientes
        for interest in interests_to_update:
            print(f"[Update] Buscando: {interest}")
            results = duckduckgo_search(interest, config)
            if results and "No encontré" not in results[0]:
                entry = {
                    "query": interest,
                    "content": "\n".join(results),
                    "sources": [r for r in results if "(Fuente:" in r],
                    "timestamp": time.time()
                }
                add_to_knowledge_base(entry, config)
                print(f"[Update] Agregado: {interest}")
            else:
                print(f"[Update] Sin resultados nuevos para: {interest}")
        
        print(f"✅ Actualización completada. Esperando {update_interval} minutos...")
        time.sleep(update_interval * 60)

def extract_image_urls(text):
    """Detecta más tipos de URLs de imágenes"""
    # Patrón base
    pattern1 = r'https?://[^\s<>"]+?\.(?:png|jpg|jpeg|gif|webp|bmp)(?:\?[^\s<>"]*)?'
    
    # Para Imgur (sin extensión)
    pattern2 = r'https?://i\.imgur\.com/[a-zA-Z0-9]+(?:\.\w+)?'
    
    # Para otros CDNs comunes
    pattern3 = r'https?://(?:cdn\.|media\.)[^\s]+\.(?:png|jpg|jpeg|gif)'
    
    all_patterns = f'({pattern1})|({pattern2})|({pattern3})'
    
    matches = re.findall(all_patterns, text, re.IGNORECASE)
    
    # Flatten y limpiar
    urls = []
    for match in matches:
        for group in match:
            if group:
                urls.append(group)
                break
    
    return list(set(urls))  # Eliminar duplicados

def extract_text_from_image_url(image_url):
    """Versión simplificada para Windows"""
    try:
        print(f"[OCR] Procesando imagen...")
        
        # Descarga SIMPLE (sin headers complicados)
        response = requests.get(image_url, timeout=10)
        
        if response.status_code != 200:
            return f"[Error HTTP {response.status_code}]"
        
        # Procesar imagen
        img = Image.open(BytesIO(response.content))
        
        # OCR
        results = ocr_reader.readtext(img, paragraph=True)
        
        if not results:
            return "[Sin texto legible]"
        
        # Extraer textos con buena confianza
        texts = []
        for bbox, text, confidence in results:
            if confidence > 0.3 and text.strip():
                texts.append(text.strip())
        
        if not texts:
            return "[Texto no claro]"
        
        full_text = " ".join(texts)
        if len(full_text) > 500:
            full_text = full_text[:500] + "..."
        
        return full_text
        
    except Exception as e:
        return f"[Error: {type(e).__name__}]"

def safe_post(url, payload):
    global last_llm_call
    wait = LLM_MIN_INTERVAL - (time.time() - last_llm_call)
    if wait > 0:
        time.sleep(wait)

    last_llm_call = time.time()

    try:
        #Aumentar timeout para Qwen3-VL
        response = requests.post(url, headers=HEADERS, json=payload, timeout=120)
        response.raise_for_status()
        return response.json(), None
    except Exception as e:
        return None, str(e)

def extract_answer(payload):
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None

def generate(user_prompt, memory, config):
    
    #Fecha y hora actual
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    date_info = f"hoy es {current_date}."
    
    if not any(msg["role"] == "system" and "Fecha actual" in msg["content"] for msg in memory):
        memory.insert(0, {"role": "system", "content": f"Fecha actual: {date_info}"})
    
    # --- KB ---
    kb = load_knowledge_base()
    relevant_kb = []
    user_words = set(user_prompt.lower().split())
    
    for entry in kb[-10:]:
        entry_query = entry.get("query", "").lower()
        entry_words = set(entry_query.split())
        if user_words & entry_words:
            relevant_kb.append(entry["content"])

    final_prompt = user_prompt
    if relevant_kb:
        final_prompt += (
            "\n\n[CONOCIMIENTO RELEVANTE DE KB]:\n"
            + "\n---\n".join(relevant_kb[:2])
        )

    # --- IMÁGENES CON OCR REAL ---
    image_urls = extract_image_urls(user_prompt)
    
    if image_urls and config.get("vision_enabled", True):
        print(f"\n🔍 [Loji] Detecté imagen: {image_urls[0][:50]}...")
        
        # EXTRAER TEXTO REAL CON OCR
        ocr_result = extract_text_from_image_url(image_urls[0])
        
        if not ocr_result.startswith("[Error") and not ocr_result.startswith("[La imagen"):
            # ✅ TEXTO EXTRAÍDO CON ÉXITO
            final_prompt += f"\n\n[TEXTO EXTRAÍDO DE LA IMAGEN]:\n{ocr_result}"
            print(f"✅ Texto extraído ({len(ocr_result)} caracteres)")
            
            # Reemplazar URL por marcador
            final_prompt = final_prompt.replace(image_urls[0], "[IMAGEN]")
        else:
            # ❌ OCR FALLÓ
            final_prompt += f"\n\n[Imagen detectada pero no contiene texto legible]"
            print(f"⚠️  {ocr_result}")
    
    # --- LLM CALL ---
    messages = memory.copy()
    messages.append({"role": "user", "content": final_prompt})

    payload = {
        "model": config.get("model_name", "local-model"),
        "messages": messages,
        "max_tokens": config.get("max_tokens", 800),
        "temperature": config.get("temperature", 0.7),
        "stream": False
    }

    response, error = safe_post(config.get("lm_api", LM_API), payload)
    if error:
        return f"Error al contactar LLM: {error}"

    answer = extract_answer(response)
    if not answer:
        return "No pude generar una respuesta."

    memory.append({"role": "user", "content": user_prompt})
    memory.append({"role": "assistant", "content": answer})
    save_memory(memory)

    return answer

def loji_console():
    print("Loji 1.5 - Asistente de IA con búsqueda web y visión por computadora")
    print("Escribe '/help' para ver comandos disponibles.")
    
    config = load_settings()
    memory = load_config() + load_memory()
    
    # ✅ Solo un hilo de actualización
    update_thread = threading.Thread(target=periodic_update, args=(config,), daemon=True)
    update_thread.start()
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
            "/vision on/off       → activar/desactivar Vision\n"
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
            print(f"Loji: Visión {status}.")
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
            results = duckduckgo_search(topic, config, force_refresh=True)
            if results and "No encontré" not in results[0]:
                entry = {
                    "query": topic,
                    "content": "\n".join(results),
                    "sources": [r for r in results if "(Fuente:" in r],
                    "timestamp": time.time()
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
                    "content": f"Calificación: {rating}, Pregunta: {last_query}, Respuesta: {last_response}",
                    "timestamp": time.time()
                }, config)
                print(f"Loji: Gracias por calificar la respuesta como '{rating}'.")
            except IndexError:
                print("Loji: Por favor, usa '/rate buena' o '/rate mala'.")
            continue
        reply = generate(user_input, memory, config)
        print(f"Loji: {reply}\n")

if __name__ == "__main__":
    loji_console()
