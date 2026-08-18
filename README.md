# Harness — Etapa 2: primer agent loop

Cliente conversacional de terminal para `gpt-oss-120b` mediante la API de
Cerebras. El modelo puede responder directamente o pedir herramientas; el
programa ejecuta cada llamada, devuelve su resultado y repite hasta obtener una
respuesta final.

El loop se mantiene deliberadamente directo, sin un framework de agentes:

1. agrega el mensaje del usuario al historial;
2. llama al modelo con los esquemas de herramientas y `tool_choice="auto"`;
3. si hay `tool_calls`, valida y ejecuta cada una;
4. agrega cada resultado con el `tool_call_id` correspondiente;
5. vuelve al paso 2 hasta recibir una respuesta final.

## Herramientas

Las tres herramientas no acceden a red ni modifican archivos:

- `calculator`: evalúa expresiones numéricas con `+`, `-`, `*`, `/`, `//`, `%`
  y `**`. Usa un intérprete limitado basado en el AST de Python, no `eval`.
- `get_current_time`: devuelve la hora actual para una zona horaria IANA, por
  ejemplo `UTC` o `America/Argentina/Mendoza`.
- `echo`: devuelve texto sin modificar. Se incluye como tercera herramienta
  inocua porque la consigna indica tres aunque enumera solo las dos anteriores.

Cada herramienta declara un esquema JSON estricto. Antes de ejecutarla, el
harness comprueba que los argumentos sean JSON, que estén todos los campos
requeridos, que sus tipos sean correctos y que no haya propiedades adicionales.
Un nombre desconocido, JSON inválido, argumentos inválidos o un error durante la
ejecución produce un resultado estructurado con `ok: false`; ese resultado se
envía al modelo para que pueda recuperarse.

## Límites y observabilidad

Cada mensaje del usuario tiene dos guardrails configurables:

- `--max-turns`: máximo de llamadas al modelo, 8 por defecto;
- `--timeout`: límite total de tiempo del loop, 30 segundos por defecto. El
  tiempo restante también se pasa como timeout a cada llamada a la API.

Si se supera un límite o falla la API, se descarta del historial el turno
incompleto para que la conversación conserve una secuencia válida.

El registro JSON Lines incluye el inicio y fin de la sesión, mensajes del
usuario, inicio de cada iteración, solicitudes y respuestas de API, decisiones
del agente, inicio y resultado de herramientas, `call_id`, argumentos, errores,
latencias, métricas y causa de terminación. Las preguntas, respuestas y
argumentos se guardan completos, por lo que el archivo puede contener
información sensible.

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# Editar .env y agregar la API key.
```

La API key se crea en la [consola de Cerebras](https://cloud.cerebras.ai/). El
programa carga `CEREBRAS_API_KEY` desde `.env`; una variable ya exportada tiene
precedencia.

## Uso

Conversación interactiva:

```bash
.venv/bin/python harness.py
```

Durante una conversación interactiva, `/clear` elimina los turnos acumulados
del usuario, el asistente y las herramientas, pero conserva la instrucción de
sistema. El comando no hace una solicitud al modelo. `/help` muestra todos los
comandos disponibles.

Una sola pregunta:

```bash
.venv/bin/python harness.py "¿Cuánto es (27 + 5) * 3?"
```

Configuración de límites y log:

```bash
.venv/bin/python harness.py \
  --max-turns 5 \
  --timeout 20 \
  --log-file logs/sesion.jsonl \
  "¿Qué hora es en America/Argentina/Mendoza?"
```

La utilización mostrada al final suma los tokens de todas las llamadas del loop
y calcula `tokens_totales / ventana_de_contexto * 100`. La ventana
predeterminada es 131.072 tokens y puede ajustarse con `--context-window`. Las
métricas también muestran cuántos tokens del prompt fueron obtenidos desde la
caché del proveedor.

Para ver todas las opciones:

```bash
.venv/bin/python harness.py --help
```

## Pruebas

```bash
.venv/bin/python -m unittest discover -s tests -v
```
