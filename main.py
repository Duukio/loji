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
    "max_tokens": 350,  # Era 800, ahora 350 (mejor para CPU)
    "temperature": 0.6,  # Era 0.7, ahora 0.6 (más consistente)
    "stream": False,
    "search_enabled": True,
    "DUCKDUCKGO_API": DUCKDUCKGO_API,
    "search_cache_ttl": 60,  #Eran 30, ahora 60 (menos búsquedas)
    "knowledge_base_max_entries": 200,
    "vision_enabled": True,
    "search_backoff_base": 1.5,
    "search_max_retries": 3,
    "enable_smart_continuation": True,  #NUEVO: Habilita continuación inteligente
    "dynamic_tokens": True,  #Ajusta tokens según tipo de pregunta
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
        "quién", "qué es", "cuándo", "dónde", "último", "última",
        "noticias", "precio", "definición", "significado", "actual",
        "qué pasó", "que paso", "qué fue de", "que fue de",
        "noticia sobre", "últimas noticias", "reciente",
        "renunció", "renuncio", "murió", "murio", "falleció"
    ]
    
    # Detectar preguntas sobre personas específicas (nombres propios)
    # Patron: dos palabras capitalizadas juntas (ej: "Juan Pérez")
    if re.search(r'\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\b', prompt):
        # Lista AMPLIADA de action words
        action_words = [
            # Preguntas básicas
            "es", "fue", "será", "sea", "son", "eran", "están", "estaba", "estuvo",
            # Acciones pasadas
            "hizo", "dijo", "anunció", "anuncio", "declaró", "declaro", "presentó", "presento",
            "renunció", "renuncio", "dimitió", "dimiti", "asumió", "asumo", 
            "ganó", "gano", "perdió", "perdio", "logró", "logro",
            # Estados y cambios
            "paso", "pasó", "pasa", "ocurrió", "ocurrio", "sucedió", "sucedio",
            "murió", "murio", "falleció", "fallecio", "nació", "nacio",
            # Acciones presentes/futuras
            "hace", "trabaja", "dirige", "lidera", "encabeza", "maneja",
            "sigue", "continúa", "continua", "mantiene",
            # Eventos/Noticias
            "pasó con", "paso con", "fue de", "fue del", "fue de la",
            "noticia", "noticias", "información", "informacion", "dato", "datos",
            # Cargos y posiciones
            "director", "directora", "ministro", "ministra", "presidente", "presidenta",
            "secretario", "secretaria", "jefe", "jefa", "titular", "cargo",
            # Verbos de investigación
            "investigó", "investigo", "descubrió", "descubrio", "reveló", "revelo",
            "confirmó", "confirmo", "desmintió", "desmintio",
            # Cambios de estado
            "cambió", "cambio", "modificó", "modifico", "actualizó", "actualizo",
            "reemplazó", "reemplazo", "sustituyó", "sustituyo", "dejó", "dejo",
            # Acciones legales/políticas
            "acusó", "acuso", "denunció", "denuncio", "demandó", "demando",
            "condenó", "condeno", "absolvió", "absolvio", "procesó", "proceso",
            # Verbos de comunicación
            "comentó", "comento", "opinó", "opino", "criticó", "critico",
            "defendió", "defendio", "apoyó", "apoyo", "rechazó", "rechazo",
            # Otros verbos comunes
            "llegó", "llego", "salió", "salio", "entró", "entro",
            "viajó", "viajo", "visitó", "visito", "asistió", "asistio",
            "participó", "participo", "organizó", "organizo",
            # Verbos en infinitivo que pueden aparecer
            "hacer", "decir", "tener", "estar", "ir", "venir", "dar", "poder",
            # Frases completas comunes
            "qué hace", "que hace", "a qué se dedica", "a que se dedica",
            "dónde está", "donde esta", "dónde trabaja", "donde trabaja",
            "cuál es su", "cual es su", "quién es", "quien es"
        ]
        if any(word in prompt.lower() for word in action_words):
            return True
    
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
    keywords = ["hoy", "último", "actual", "noticia", "2024", "2025", "2026"]
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
    kb.append(entry)
    
    # Limitar tamaño de KB
    max_entries = config.get("knowledge_base_max_entries", 200) if config else 200
    if len(kb) > max_entries:
        kb = kb[-max_entries:]
    
    with open("knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

def periodic_update(config):
    """Actualiza knowledge base periódicamente según intereses"""
    interval_minutes = config.get("update_interval", 60)
    interests = config.get("interests", [])
    
    while True:
        time.sleep(interval_minutes * 60)
        
        if not interests:
            continue
            
        for interest in interests:
            try:
                results = duckduckgo_search(interest, config, force_refresh=True)
                if results and "No encontré" not in results[0]:
                    entry = {
                        "query": interest,
                        "content": "\n".join(results),
                        "sources": [r for r in results if "(Fuente:" in r],
                        "timestamp": time.time()
                    }
                    add_to_knowledge_base(entry, config)
                    print(f"[Auto-update] Actualizado: {interest}")
            except Exception as e:
                print(f"[Auto-update] Error con {interest}: {e}")

def extract_image_urls(text):
    """Extrae URLs de imágenes del texto"""
    # Patrones comunes de URLs de imágenes
    patterns = [
        r'https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp|bmp)',
        r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+',
    ]
    
    urls = []
    for pattern in patterns:
        urls.extend(re.findall(pattern, text, re.IGNORECASE))
    
    return urls

def extract_text_from_image_url(image_url):
    """Extrae texto de imagen usando EasyOCR"""
    try:
        if image_url.startswith('data:image'):
            # Base64 image
            header, encoded = image_url.split(',', 1)
            image_data = base64.b64decode(encoded)
            image = Image.open(BytesIO(image_data))
        else:
            # URL normal
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Usar EasyOCR
        results = ocr_reader.readtext(image)
        
        if not results:
            return "[La imagen no contiene texto legible]"
        
        # Extraer texto con confianza > 0.3
        texts = [text for (bbox, text, conf) in results if conf > 0.3]
        
        if not texts:
            return "[La imagen no contiene texto con suficiente confianza]"
        
        full_text = " ".join(texts)
        if len(full_text) > 500:
            full_text = full_text[:500] + "..."
        
        return full_text
        
    except Exception as e:
        return f"[Error: {type(e).__name__}]"

# Detecta si una respuesta fue truncada
def detect_truncation(text):
    """Detecta si una respuesta fue truncada a mitad de idea"""
    if not text or len(text.strip()) < 20:
        return False
    
    # Terminaciones que indican truncamiento
    truncation_indicators = [
        r'\*\*\s*$',           # Termina en **
        r'-\s*$',              # Termina en guión
        r':\s*$',              # Termina en dos puntos
        r',\s*$',              # Termina en coma
        r'\(\s*$',             # Paréntesis abierto sin cerrar
        r'porque\s+(en|el|la|los|las|un|una)\s*$',  # "porque en/el/la..." al final
        r'(ya\s+que|dado\s+que|debido\s+a)\s*$',    # Conectores incompletos
    ]
    
    text_stripped = text.strip()
    
    for pattern in truncation_indicators:
        if re.search(pattern, text_stripped, re.IGNORECASE):
            return True
    
    # Detectar última oración muy larga sin punto final
    sentences = re.split(r'[.!?]\s+', text_stripped)
    if sentences:
        last_part = sentences[-1]
        # Si la última parte tiene más de 100 caracteres y no termina en puntuación
        if len(last_part) > 100 and not re.search(r'[.!?]\s*$', last_part):
            return True
    
    return False

# Ajusta el prompt para respuestas largas
def split_response_if_needed(prompt, config):
    """Modifica el prompt si se espera una respuesta larga"""
    if not config.get("enable_smart_continuation", True):
        return prompt
    
    # Detectar preguntas que probablemente necesiten respuestas largas
    long_answer_keywords = [
        "historia", "explica", "detalla", "cuéntame", "describe",
        "cómo funciona", "por qué", "analiza", "comparación"
    ]
    
    needs_long_answer = any(kw in prompt.lower() for kw in long_answer_keywords)
    
    if needs_long_answer:
        # Modificar prompt para versión resumida
        return f"""{prompt}

[IMPORTANTE: Da una respuesta CONCISA y bien estructurada. Si el tema requiere más extensión, termina con "...¿Querés que siga con más detalles?" o similar.]"""
    
    return prompt

# Calcula tokens dinámicos según tipo de pregunta
def calculate_dynamic_tokens(prompt, base_tokens):
    """Calcula tokens óptimos según la complejidad de la pregunta"""
    # Preguntas simples: tokens base
    simple_keywords = ["hola", "qué tal", "cómo estás", "gracias", "adiós"]
    if any(kw in prompt.lower() for kw in simple_keywords):
        return min(base_tokens, 200)
    
    # Preguntas de explicación: más tokens
    complex_keywords = [
        "historia", "explica", "cómo", "por qué", "diferencia",
        "comparación", "analiza", "detalla", "describe"
    ]
    if any(kw in prompt.lower() for kw in complex_keywords):
        return min(base_tokens * 2, 700)  # Hasta 700 para explicaciones
    
    # Preguntas normales: tokens base
    return base_tokens

def safe_post(url, payload, max_tokens):
    global last_llm_call
    wait = LLM_MIN_INTERVAL - (time.time() - last_llm_call)
    if wait > 0:
        time.sleep(wait)

    last_llm_call = time.time()

    try:
        # Timeout conservador para CPU
        timeout = 120 
        
        response = requests.post(url, headers=HEADERS, json=payload, timeout=timeout)
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
    
    # --- FECHA ACTUAL ---
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

    final_prompt = split_response_if_needed(final_prompt, config)

    if should_search(user_prompt, memory, config):
        print(f"\n🔍 [Loji] Buscando información actualizada...")
        search_results = duckduckgo_search(user_prompt, config)
        
        if search_results and "No encontré" not in search_results[0] and "Error" not in search_results[0]:
            # Agregar resultados al prompt
            search_text = "\n".join(search_results[:3])  # Top 3 resultados
            final_prompt += f"\n\n[DATOS DE BÚSQUEDA ACTUAL]:\n{search_text}"
            print(f"✅ Encontrados {len(search_results)} resultados")
            
            # Guardar en KB para futuras consultas
            add_to_knowledge_base({
                "query": user_prompt,
                "content": search_text,
                "sources": search_results,
                "timestamp": time.time()
            }, config)
        else:
            print(f"⚠️  No se encontró información actualizada")

    # --- IMÁGENES CON OCR REAL ---
    image_urls = extract_image_urls(user_prompt)
    
    if image_urls and config.get("vision_enabled", True):
        print(f"\n🔍 [Loji] Detecté imagen: {image_urls[0][:50]}...")
        
        # EXTRAER TEXTO REAL CON OCR
        ocr_result = extract_text_from_image_url(image_urls[0])
        
        if not ocr_result.startswith("[Error") and not ocr_result.startswith("[La imagen"):
            #TEXTO EXTRAÍDO CON ÉXITO
            final_prompt += f"\n\n[TEXTO EXTRAÍDO DE LA IMAGEN]:\n{ocr_result}"
            print(f"✅ Texto extraído ({len(ocr_result)} caracteres)")
            
            # Reemplazar URL por marcador
            final_prompt = final_prompt.replace(image_urls[0], "[IMAGEN]")
        else:
            #OCR FALLÓ
            final_prompt += f"\n\n[Imagen detectada pero no contiene texto legible]"
            print(f"⚠️  {ocr_result}")
    
    # Calcular tokens dinámicos
    base_tokens = config.get("max_tokens", 350)
    if config.get("dynamic_tokens", True):
        max_tokens_adjusted = calculate_dynamic_tokens(user_prompt, base_tokens)
    else:
        max_tokens_adjusted = base_tokens
    
    # --- LLM CALL ---
    messages = memory.copy()
    messages.append({"role": "user", "content": final_prompt})

    payload = {
        "model": config.get("model_name", "local-model"),
        "messages": messages,
        "max_tokens": max_tokens_adjusted,  # ✅ Ahora es dinámico
        "temperature": config.get("temperature", 0.7),
        "stream": False
    }

    response, error = safe_post(config.get("lm_api", LM_API), payload, max_tokens_adjusted)
    if error:
        return f"Error al contactar LLM: {error}"

    answer = extract_answer(response)
    if not answer:
        return "No pude generar una respuesta."

     # Detectar y marcar truncamiento
    if config.get("enable_smart_continuation", True) and detect_truncation(answer):
        answer += "\n\n💬 *[Respuesta extensa. Escribí 'seguí' o 'continúa' para ver más detalles]*"

    memory.append({"role": "user", "content": user_prompt})
    memory.append({"role": "assistant", "content": answer})
    save_memory(memory)

    return answer

def loji_console():
    print("=" * 60)
    print("Loji 1.5.2 - Asistente IA Optimizado")
    print("Created by Emiliano Cabella - Universidad del Comahue")
    print("=" * 60)
    print("Escribe '/help' para ver comandos disponibles.\n")
    
    config = load_settings()
    memory = load_config() + load_memory()
    
    #Se actualiza en un solo hilo 
    update_thread = threading.Thread(target=periodic_update, args=(config,), daemon=True)
    update_thread.start()
    print("Hilo de actualización automática iniciado.\n")

    def print_help():
        print(
            "\n" + "=" * 60 +
            "\nComandos disponibles:\n" +
            "=" * 60 +
            "\n/help                → Mostrar esta ayuda\n"
            "/config              → Ver configuración actual\n"
            "/interests           → Ver intereses\n"
            "/interests set a,b   → Reemplazar intereses\n"
            "/interests add <t>   → Agregar interés\n"
            "/interests remove <t>→ Quitar interés\n"
            "/refresh <tema>      → Forzar búsqueda de ese tema\n"
            "/vision on/off       → Activar/desactivar OCR\n"
            "/dynamic on/off      → Activar/desactivar tokens dinámicos\n"
            "/continuation on/off → Activar/desactivar continuación inteligente\n"
            "/rate buena|mala     → Calificar última respuesta\n"
            "/clear               → Borrar memoria de conversación\n"
            "/stats               → Ver estadísticas del sistema\n"
            "/exit                → Salir\n"
            "=" * 60 + "\n"
        )

    def print_stats(config):
        """Muestra estadísticas del sistema"""
        kb = load_knowledge_base()
        cache_size = len(search_cache)
        print("\n" + "=" * 60)
        print("ESTADÍSTICAS DEL SISTEMA")
        print("=" * 60)
        print(f"Entradas en Knowledge Base: {len(kb)}")
        print(f"Búsquedas en caché: {cache_size}")
        print(f"Max tokens configurado: {config.get('max_tokens', 350)}")
        print(f"Temperature: {config.get('temperature', 0.6)}")
        print(f"Visión (OCR): {'Activado' if config.get('vision_enabled') else '❌ Desactivado'}")
        print(f"Tokens dinámicos: {'Activado' if config.get('dynamic_tokens') else '❌ Desactivado'}")
        print(f"Continuación inteligente: {'Activado' if config.get('enable_smart_continuation') else '❌ Desactivado'}")
        print("=" * 60 + "\n")

    while True:
        user_input = input("Tú: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() == "/help":
            print_help()
            continue
            
        if user_input.lower() == "/exit":
            print("Loji: ¡Hasta luego!")
            break
            
        if user_input.lower() == "/clear":
            clear_memory()
            memory = load_config()
            print("Loji: Memoria borrada.\n")
            continue
            
        if user_input.lower() == "/config":
            print(json.dumps(config, indent=2, ensure_ascii=False))
            continue
            
        if user_input.lower() == "/stats":
            print_stats(config)
            continue
            
        # Comando para activar/desactivar tokens dinámicos
        if user_input.lower().startswith("/dynamic"):
            parts = user_input.split()
            if len(parts) != 2 or parts[1].lower() not in ["on", "off"]:
                print("Loji: Usa '/dynamic on' o '/dynamic off'.\n")
                continue
            config["dynamic_tokens"] = parts[1].lower() == "on"
            save_settings(config)
            status = "activado" if config["dynamic_tokens"] else "desactivado"
            print(f"Loji: Tokens dinámicos {status}.\n")
            continue
            
        # Comando para activar/desactivar continuación inteligente
        if user_input.lower().startswith("/continuation"):
            parts = user_input.split()
            if len(parts) != 2 or parts[1].lower() not in ["on", "off"]:
                print("Loji: Usa '/continuation on' o '/continuation off'.\n")
                continue
            config["enable_smart_continuation"] = parts[1].lower() == "on"
            save_settings(config)
            status = "activado" if config["enable_smart_continuation"] else "desactivado"
            print(f"Loji: Continuación inteligente {status}.\n")
            continue
            
        if user_input.lower().startswith("/vision"):
            parts = user_input.split()
            if len(parts) != 2 or parts[1].lower() not in ["on", "off"]:
                print("Loji: Usa '/vision on' o '/vision off'.\n")
                continue
            config["vision_enabled"] = parts[1].lower() == "on"
            save_settings(config)
            status = "activado" if config["vision_enabled"] else "desactivado"
            print(f"Loji: Visión (OCR) {status}.\n")
            continue
            
        if user_input.lower().startswith("/interests"):
            parts = user_input.split(maxsplit=2)
            if len(parts) == 1:
                interests = config.get('interests', [])
                if interests:
                    print(f"Intereses actuales: {', '.join(interests)}\n")
                else:
                    print("No hay intereses configurados.\n")
                continue
            action = parts[1].lower()
            value = parts[2].strip() if len(parts) > 2 else ""
            if action == "set" and value:
                config["interests"] = [v.strip() for v in value.split(",") if v.strip()]
                save_settings(config)
                print("Loji: Intereses actualizados.\n")
                continue
            if action == "add" and value:
                interests = config.get("interests", [])
                if value not in interests:
                    interests.append(value)
                config["interests"] = interests
                save_settings(config)
                print(f"Loji: Interés agregado: {value}\n")
                continue
            if action == "remove" and value:
                config["interests"] = [i for i in config.get("interests", []) if i != value]
                save_settings(config)
                print(f"Loji: Interés eliminado: {value}\n")
                continue
            print("Loji: Usa '/interests', '/interests set a,b', '/interests add <t>' o '/interests remove <t>'.\n")
            continue
            
        if user_input.lower().startswith("/refresh"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Loji: Usa '/refresh <tema>'.\n")
                continue
            topic = parts[1].strip()
            print(f"Buscando: {topic}...")
            results = duckduckgo_search(topic, config, force_refresh=True)
            if results and "No encontré" not in results[0]:
                entry = {
                    "query": topic,
                    "content": "\n".join(results),
                    "sources": [r for r in results if "(Fuente:" in r],
                    "timestamp": time.time()
                }
                add_to_knowledge_base(entry, config)
                print("Loji: Búsqueda actualizada en la base de conocimientos.\n")
            else:
                print("Loji: No encontré resultados nuevos.\n")
            continue
            
        if user_input.lower().startswith("/rate"):
            try:
                rating = user_input.split()[1].lower()
                if rating not in ["buena", "mala"]:
                    print("Loji: Por favor, usa '/rate buena' o '/rate mala'.\n")
                    continue
                last_response = memory[-1]["content"] if memory and memory[-1]["role"] == "assistant" else "No hay respuesta previa."
                last_query = memory[-2]["content"] if len(memory) >= 2 and memory[-2]["role"] == "user" else "No hay pregunta previa."
                add_to_knowledge_base({
                    "query": "feedback",
                    "content": f"Calificación: {rating}, Pregunta: {last_query}, Respuesta: {last_response}",
                    "timestamp": time.time()
                }, config)
                print(f"Loji: Gracias por calificar como '{rating}'.\n")
            except IndexError:
                print("Loji: Por favor, usa '/rate buena' o '/rate mala'.\n")
            continue
        
        #Comando para continuar respuestas anteriores
        if user_input.lower() in ["seguí", "continua", "continúa", "sigue", "continuar", "más"]:
            if memory and len(memory) >= 2 and memory[-1]["role"] == "assistant":
                print("Continuando respuesta anterior...\n")
                continuation_prompt = "Continuá tu respuesta anterior donde la dejaste, desarrollando más el tema."
                reply = generate(continuation_prompt, memory, config)
                print(f"Loji: {reply}\n")
            else:
                print("Loji: No hay una respuesta previa que continuar.\n")
            continue
        
        # Generar respuesta normal
        reply = generate(user_input, memory, config)
        print(f"Loji: {reply}\n")

if __name__ == "__main__":
    loji_console()