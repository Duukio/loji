# Loji

Loji es un experimento de asistente local con memoria, búsqueda balanceada (DuckDuckGo), y soporte opcional para descripción de imágenes con BLIP. Usa un endpoint compatible con OpenAI para generar respuestas con un modelo local.

## Requisitos

- Python 3.9+
- Un servidor local compatible con OpenAI (por ejemplo LM Studio, llama.cpp, ollama + openai proxy).

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuración

Edita `config.json` para personalizar:

- `persona`: prompt del sistema.
- `interests` o `intereses`: temas para la actualización periódica de la base de conocimiento.
- `update_interval`: minutos entre actualizaciones automáticas.
- `lm_api`: URL del endpoint `v1/chat/completions`.
- `model_name`: nombre del modelo expuesto por tu servidor.
- `temperature`, `max_tokens`, `stream`: parámetros del modelo.
- `search_enabled`: habilita/deshabilita la búsqueda web.

## Uso

```bash
python main.py
```

Comandos:
- `/exit`: salir.
- `/clear`: borrar memoria.
- `/rate buena|mala`: calificar la última respuesta (se guarda en la base de conocimiento).

## Notas

- La búsqueda balanceada se apoya en una lista de fuentes predefinidas.
- La descripción de imágenes se activa si detecta URLs de imágenes en el prompt.

