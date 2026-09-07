# AGENTS.md

Este archivo es un estándar legible por máquinas diseñado para Agentes de Codificación de IA (como Cursor, Antigravity, Claude Code, Aider, Copilot, etc.) que trabajan en el proyecto **UNAL-Sistema-Puntaje**.

Los Agentes de IA **DEBEN** leer, respetar y adherirse estrictamente a todas las instrucciones, estándares y restricciones de flujo de trabajo descritas a continuación.

---

## 🤖 Descripción General del Sistema y Reglas de Colaboración

Este repositorio es un **entorno colaborativo multi-agente** donde múltiples ingenieros humanos y sus respectivos agentes de IA operan en paralelo. Para mantener una ventana de contexto limpia y evitar la deriva de instrucciones, **los cálculos de negocio, las reglas legales y las habilidades procedimentales NUNCA deben escribirse en este archivo**.

En su lugar, este archivo actúa como el **controlador global de tráfico de agentes**, proporcionando reglas de interacción y dirigiendo a los agentes a subdirectorios organizados.

### 🛡️ Evitación de Conflictos Multi-Agente (MACA)
Para evitar sobrescribir el trabajo de otros agentes y causar conflictos de fusión (merge conflicts):
1. **Ramificación Aislada (Isolated Branching):** Realiza siempre los cambios en una rama de características (feature branch) dedicada, derivada únicamente de la rama `dev`. Está estrictamente prohibido realizar commits directamente en las ramas `main`, `master` o `dev`. Usa el formato de espacio de nombres para tu rama: `agent/<nombre-agente>/<nombre-caracteristica>` (por ejemplo, `agent/antigravity/setup-framework`).
2. **Commits Atómicos:** Realiza commits pequeños, lógicos y altamente enfocados. Cada commit debe cubrir solo una capacidad específica o corrección de error.
3. **Sin Reescritura Dinámica:** No reescribas grandes bloques de código a menos que se solicite explícitamente. Respeta el estilo y las estructuras establecidas en los módulos existentes.
4. **Coordinación de Bloqueos:** Al editar archivos, verifica si otro agente ha realizado cambios recientemente en el mismo archivo. Coordina con los operadores humanos si los cambios en paralelo se superponen.

---

## 🔄 🤖 Protocolo de Co-Mantenimiento Continuo de Agentes (CACM)

En un entorno multi-agente que introduce cambios en tiempo real, **la degradación del contexto del espacio de trabajo es el principal modo de fallo**. Cada agente activo tiene la responsabilidad estricta de mantener, actualizar y sincronizar los archivos de agentes sobre la marcha, asegurando que se propaguen de manera continua los conocimientos, inquietudes y el estado del contexto a otros agentes.

### 📋 Las Reglas del CACM (Mantenimiento y Guardrails):
1. **Sincronizar Documentación al Cambiar:** Si modificas, agregas o refactorizas cualquier lógica de negocio (por ejemplo, cambiar una regla matemática de puntaje, actualizar un modelo de datos o agregar un nuevo índice de base de datos), **DEBES** actualizar inmediatamente los archivos correspondientes en la carpeta `.agents/`:
   *   **Habilidades (Skills):** Actualiza las entradas, salidas o pasos de ejecución en el `SKILL.md` relevante bajo [`.agents/skills/`](./.agents/skills/).
   *   **Plugins:** Si existen validaciones o integraciones automatizadas, actualiza sus configuraciones en `.agents/plugins/`.
   *   **Recursos (Resources):** Actualiza el diagrama de flujo en formato ASCII Unicode o las referencias en [`.agents/resources/`](./.agents/resources/).
2. **Mutación de Skills y Guardrails:** Los "Guardrails" descritos en cada skill restringen el comportamiento del agente **durante la ejecución** de tareas rutinarias para evitar desviaciones. SIN EMBARGO, si el usuario explícitamente solicita actualizar el sistema, cambiar implementaciones o refactorizar dependencias, el agente **tiene total autoridad para mutar, evolucionar o eliminar** estos guardrails y archivos de sistema, siempre y cuando actualice el `SKILL.md` reflejando el nuevo estado y las nuevas reglas.
3. **Propagación Activa de Conocimiento e Inquietudes:** Todo agente que descubra una ambigüedad en la normativa (Decreto 1279 y el Acuerdo 023 del CSU), un caso extremo (*edge case*) no cubierto, o inquietudes sobre la validación de un documento de soporte (p. ej., sellos fraudulentos, indexación desactualizada), **DEBES** registrar esta advertencia o aprendizaje de forma clara en la sección de "Consideraciones" o "FAQs" de la habilidad (`SKILL.md`) correspondiente y en la documentación del recurso. Ningún conocimiento técnico útil debe quedar oculto.
4. **Implementaciones Auto-Documentadas:** Si implementas una nueva característica que representa una capacidad reutilizable, **DEBES** redactar un nuevo `SKILL.md` bajo `.agents/skills/` utilizando el diseño de `template-skill` establecido como tu plantilla base.
5. **Notificar en Recorridos (Walkthroughs):** Después de realizar actualizaciones, documenta lo que cambiaste en el recorrido local para que los agentes posteriores tengan un registro cronológico de las evoluciones del marco de trabajo.

*El incumplimiento de mantener el directorio `.agents/` sincronizado con el código base activo se considera una regresión de ingeniería.*

---

## 🛠️ Entorno y Comandos

*   **Lenguaje de Programación:** **Python 3.12**
*   **Entorno Virtual (OBLIGATORIO):** Todo agente de IA **DEBE** verificar que existe un entorno virtual (`.venv/`) en la raíz del repositorio antes de ejecutar cualquier comando Python. Si no existe, debe crearlo con `python -m venv .venv`. Todo comando `python`, `pip`, `pytest`, `ruff` o `uvicorn` **DEBE** ejecutarse dentro del entorno virtual activado. Nunca instales dependencias en el intérprete global del sistema.
    *   **Activar (Windows):** `.venv\Scripts\activate`
    *   **Activar (Linux/macOS):** `source .venv/bin/activate`
*   **Gestión de Dependencias (OBLIGATORIO):** El archivo `requirements.txt` es la fuente de verdad para las dependencias del proyecto. Las versiones **DEBEN** estar fijadas con `==` (por ejemplo, `fastapi==0.115.6`), nunca con rangos (`>=`, `~=`). Todo agente que agregue, actualice o elimine una dependencia **DEBE** actualizar `requirements.txt` inmediatamente con la versión exacta instalada. Usa `pip freeze | findstr <paquete>` (Windows) o `pip freeze | grep <paquete>` (Linux/macOS) para obtener la versión exacta.
    *   🚨 **Compilación nativa en Windows:** si `pip install -r requirements.txt` falla por falta de un *wheel* / Build Tools de un paquete con extensión C/C++, dile al usuario que instale **Microsoft C++ Build Tools** y detén la ejecución. **No** cambies versiones de Python ni inventes pins alternativos sin pedirlo. El RAG en producción usa **Cosmos DB** (`vectordb`, 768 dims); **no** hay ChromaDB en `requirements.txt`.
*   **Restricciones de Desacoplamiento:** Los módulos de evaluación central deben ser **completamente agnósticos de proveedor (vendor-agnostic) e independientes de SARA**. No importes SDKs de Azure o AWS, controladores de bases de datos o clientes de API dentro del código de cálculo de negocio. El cálculo de topes normativos y acumulados salariales **NUNCA** debe delegarse al prompt del LLM; debe implementarse de forma 100% determinista en Python (`src/scoring/rules_titulos.py` y `src/scoring/optimizer.py`).
    *   **Directriz Mutable (Root.pdf) como Raíz de Conocimiento:** Las reglas operativas de prioridades de experiencia, equivalencias de horas semanales, criterios de experiencia investigativa, límites para docentes cursando posgrado y validación de requisitos de categoría están consolidadas en el manual mutable [Root.pdf](./data/knowledge/Root.pdf). **ESTÁ ESTRICTAMENTE PROHIBIDO** programar estas reglas en código duro de Python. Deben ser interpretadas y mapeadas dinámicamente mediante el orquestador del LLM que lee la directriz mutable en tiempo de ejecución. Python únicamente debe actuar como ejecutor matemático de las directrices y anulaciones (overrides) devueltas por el LLM.
*   **Ámbito del Sistema:** El sistema soporta actualmente solo procesos de **ingreso y reingreso** docente. La categoría escalafonaria pretendida se infiere dinámicamente desde el parámetro de contexto libre.
*   **Estructura de Carpetas:** `src/`, `tests/`, `dashboard/` (Nuxt), `infra/` (Terraform) y `.agents/` son las estructuras canónicas.
*   **Verificación y Pruebas:** Escribe siempre pruebas unitarias para cualquier cambio en las reglas matemáticas o lógicas dentro del módulo `src.scoring`.
*   **Ventana de procesamiento:** Con `PROCESSING_WINDOW_ENABLED` (on por defecto), los envíos que tocan SARA o encolan trabajo pesado (`POST /process`, `POST /profile/session` + commit, `POST /dossiers/{id}/stow`) solo se aceptan **lun–vie 7:00–18:00 COT**. Ver `src/core/schedule.py`.
*   **RAG:** entrenamiento vía `POST /knowledge/train` → `knowledge-queue` (skill `/train`). Índice en Cosmos `vectordb` (768 dims), no ChromaDB.

### Comandos Estándar
> ⚠️ **Todos los comandos asumen que el entorno virtual está activado.**
*   **Ejecutar Suite de Pruebas:** `pytest -v`
*   **Instalar Dependencias:** `pip install -r requirements.txt`
*   **API local:** `uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload` (skill `/serve`)
*   **Lint y Formato:** `ruff check . && ruff format --check .`
*   **Deploy a Azure dev:** skill `/deploy` — scripts `.github/scripts/deploy-{func,swa,container}.sh`

---

## 📚 Descubrimiento de Habilidades y Conocimiento de Agentes

Los agentes de IA deben localizar las reglas contextuales dinámicamente en lugar de asumirlas.

### 🔍 Cómo Descubrir Capacidades y Reglas
Antes de generar operaciones complejas o scripts personalizados, **DEBES** escanear los siguientes directorios:
1. **Referencia Normativa (PDFs):** Revisa [`data/knowledge/`](./data/knowledge/) para decretos oficiales (Decreto 1279, Acuerdo 023 del CSU, acuerdos internos, circulares). Léelos para resolver ambigüedades en las reglas.
2. **Flujos operativos:** Lee [`.agents/resources/workflow_diagram.md`](./.agents/resources/workflow_diagram.md) — secciones **A** (`/process`) y **B** (`/profile`). La mitad inferior (Drive/consolidación) es aspiracional.
3. **Habilidades Activas (Skills):** Revisa [`.agents/skills/`](./.agents/skills/):

| Skill | Comando | Descripción |
|---|---|---|
| [`setup`](./.agents/skills/setup/SKILL.md) | `/setup` | Inicializar entorno (venv, dependencias, .env) |
| [`test`](./.agents/skills/test/SKILL.md) | `/test` | Ejecutar suite de tests unitarios (pytest) |
| [`ci`](./.agents/skills/ci/SKILL.md) | `/ci` | Pipeline CI completo (ruff check + format + pytest) |
| [`serve`](./.agents/skills/serve/SKILL.md) | `/serve` | Arrancar servidor FastAPI local |
| [`stop`](./.agents/skills/stop/SKILL.md) | `/stop` | Detener servidor y liberar puerto/recursos |
| [`deploy`](./.agents/skills/deploy/SKILL.md) | `/deploy {func\|swa\|container\|all}` | Publicar a Azure `dev` (scripts + ChatOps `bot: deploy-*`) |
| [`manage-infrastructure`](./.agents/skills/manage-infrastructure/SKILL.md) | `/manage-infrastructure` | Terraform / recursos Azure |
| [`train`](./.agents/skills/train/SKILL.md) | `/train` | Entrenar RAG normativo (`POST /knowledge/train` → Cosmos) |
| [`retrieve`](./.agents/skills/retrieve/SKILL.md) | `/retrieve {id}` | Descargar expediente docente desde SARA (scraping CLI) |
| [`diff`](./.agents/skills/diff/SKILL.md) | `/diff {id}` | Genera patch_profile.json (Grupo 5 off salvo `--include-group5`) |
| [`stow`](./.agents/skills/stow/SKILL.md) | `/stow {id}` | Carga en SARA desde patch local (CLI); cloud HITL vía dashboard |
| [`NewHV`](./.agents/skills/NewHV/SKILL.md) | `/NewHV {id}` | Crear HV en SARA (CLI; dry-run default, `--live`) |
| [`patch`](./.agents/skills/patch/SKILL.md) | `/patch {id}` | Modificar/crear registros en SARA con reconciliación LLM |
| [`generate-registers`](./.agents/skills/generate-registers/SKILL.md) | `/generate-registers` | Regenerar `registers.json` (mapa de campos SARA) |
| [`upload-dossier`](./.agents/skills/upload-dossier/SKILL.md) | `/upload-dossier` | Subir expediente local a Blob + OCR (sin scrape) |
| [`process`](./.agents/skills/process/SKILL.md) | `/process {ids[]}` | Evaluar expediente(s) docente(s) (lote + email) |
| [`profile`](./.agents/skills/profile/SKILL.md) | `/profile` | Creación de Perfil (subida progresiva → OCR → new_profile → stow) |
| [`compare-findings`](./.agents/skills/compare-findings/SKILL.md) | `/compare-findings` | Comparar puntajes del motor vs Excel histórico y/o un commit anterior |
| [`usage-report`](./.agents/skills/usage-report/SKILL.md) | `/usage-report` | Reporte de uso/costo real desde Azure Monitor |
| [`scoring-analysis`](./.agents/skills/scoring-analysis/SKILL.md) | — | Referencia normativa del motor (invocar vía `/process`) |
| [`template-skill`](./.agents/skills/template-skill/SKILL.md) | — | Plantilla para crear nuevas skills |

---

## ✍️ Estilo de Código y Directrices

1. **Sé Conciso y Exacto:** Minimiza las ediciones de archivos a las líneas exactas que requieren cambios. No introduzcas modificaciones no relacionadas.
2. **Preserva la Documentación:** Preserva siempre los comentarios existentes, la documentación en línea y los encabezados de derechos de autor.
3. **Redacta Explicaciones Libremente:** Usa markdown estándar para explicar tu razonamiento al usuario, pero mantén los cambios de código altamente limpios y enfocados.
4. **Límites de las Habilidades (Skills) y Ejecución Estricta:** La ejecución de cualquier habilidad (skill) debe limitarse exclusivamente a su EJECUCIÓN y VERIFICACIÓN. Si ocurren errores, el agente debe reportarlos al usuario, pero **ESTÁ ESTRICTAMENTE PROHIBIDO** realizar de forma proactiva cambios en el código del sistema, mejoras o refactorizaciones automáticas para solucionar o mejorar el comportamiento, a menos que el usuario lo solicite de manera explícita. Asimismo, si una skill depende de otra, el agente debe notificar al usuario para que ejecute la skill dependiente de forma manual; no debe encadenar ni ejecutar automáticamente otras skills en segundo plano.
