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
import html as html_lib
from urllib.parse import urlparse, unquote, parse_qs
from io import BytesIO  # NUEVO: Para manejar bytes de imágenes descargadas
from PIL import Image  # NUEVO: Para procesar imágenes
import torch  # NUEVO: Para BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration  # NUEVO: Para BLIP

# Configuración
LM_API = "http://localhost:1234/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b"
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

def search_web(query, config, force_refresh=False):
    return duckduckgo_search(query, config, force_refresh)

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

            # Abstract (respuesta instantánea si existe)
            if data.get("AbstractText"):
                source = data.get("AbstractURL", "Sin fuente")
                bias = detect_bias(source)
                results.append(f"[{bias}] {data['AbstractText']} (Fuente: {source})")

            # Related Topics (resultados relacionados)
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


def load_knowledge_base():
    try:
        with open("knowledge_base.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []


def add_to_knowledge_base(entry, config=None):
    kb = load_knowledge_base()
    kb.append(entry)
    kb = kb[-DEFAULT_CONFIG["knowledge_base_max_entries"]:]
    with open("knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)


def extract_image_sources(text):
    return []


def describe_image(source):
    return "Visión no implementada"



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
            results = duckduckgo_search(interest, normalized)
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

def generate(prompt, memory, config):
    # Inyectar conocimiento relevante de KB
    kb = load_knowledge_base()
    relevant_kb = []
    for entry in kb[-5:]:  # Últimas 5 entradas para no sobrecargar
        if any(keyword in prompt.lower() for keyword in entry["query"].split()):
            relevant_kb.append(entry["content"])
    if relevant_kb:
        prompt += f"\n\nCONOCIMIENTO RELEVANTE DE KB (actualizado): {chr(10).join(relevant_kb[:2])}"  # Limitar a 2

    # NUEVO: Procesar imágenes si hay fuentes válidas en el prompt (URL, data URI o archivo local)
    if config.get("vision_enabled", False):
        image_source = extract_image_sources(prompt)
        for source in image_source[:1]:  # Limitar a 1 imágen por prompt para no sobrecargar memoria
            desc = describe_image(source)
            prompt += f"\n\n[Descripción de imagen ({source}): {desc}]"

    payload = {
        "model": config.get("model_name", MODEL_NAME),
        "messages": memory + [{"role": "user", "content": prompt}],
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

    memory.append({"role": "user", "content": prompt})
    memory.append({"role": "assistant", "content": answer})

    return answer
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
            results = duckduckgo_search(topic, config, force_refresh=True)
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
