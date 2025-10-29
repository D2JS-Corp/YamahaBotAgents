# Yamaha Bot Agents - Jetson Deployment

Este directorio contiene la configuración para desplegar los agentes de Yamaha Bot en dispositivos NVIDIA Jetson utilizando Jetson Containers.

## 📋 Requisitos Previos

- **Hardware:** NVIDIA Jetson (Nano, Xavier NX, AGX Xavier, Orin, etc.)
- **JetPack:** 5.1.2 o superior (recomendado 6.0+)
- **Docker:** Preinstalado con JetPack
- **NVIDIA Container Runtime:** Configurado automáticamente con JetPack

## 📁 Estructura del Proyecto

```
Jetson/
├── app/
│   └── server.py          # Servidor principal de Pipecat
├── docker/
│   ├── Dockerfile.app     # Dockerfile para la aplicación
│   └── compose.yaml       # Docker Compose multi-servicio
├── requirements.txt       # Dependencias Python
├── .env.example          # Variables de entorno de ejemplo
└── README.md             # Este archivo
```

## 🚀 Instalación y Configuración

### 1. Preparar Modelos

Antes de ejecutar, necesitas descargar los modelos necesarios:

```bash
# Crear directorio para modelos
mkdir -p docker/models/piper
mkdir -p docker/models/ollama

# Copiar el modelo de Piper desde la carpeta ES o PROD_ES
cp ../ES/es_MX-claude-high.onnx docker/models/piper/
cp ../ES/es_MX-claude-high.onnx.json docker/models/piper/
```

### 2. Configurar Variables de Entorno (Opcional)

```bash
# Copiar el archivo de ejemplo
cp .env.example .env

# Editar según tus necesidades
nano .env
```

### 3. Construir y Ejecutar con Docker Compose

```bash
cd docker

# Construir las imágenes
docker compose build

# Iniciar todos los servicios
docker compose up -d

# Ver logs
docker compose logs -f app
```

## 🐳 Servicios en Docker Compose

### 1. **Ollama** (LLM)
- **Puerto:** 11434
- **Imagen:** ollama/ollama:latest
- **Modelo por defecto:** llama3.2:1b
- **Volumen:** `./docker/models/ollama` → `/root/.ollama`

### 2. **Piper TTS** (Text-to-Speech)
- **Puerto:** 5000
- **Imagen:** dustynv/piper-tts:r36.2.0
- **Modelo:** es_MX-claude-high
- **Volumen:** `./docker/models/piper` → `/data/models`

### 3. **App** (Pipecat Server)
- **Puerto:** 8000
- **Imagen:** pipecat-jetson:latest
- **Base:** dustynv/l4t-pytorch:r36.2.0
- **Incluye:** FAISS-GPU, PyTorch, Pipecat

## 🔧 Comandos Útiles

### Descargar modelo Ollama

```bash
# Entrar al contenedor de Ollama
docker exec -it ollama bash

# Descargar modelo
ollama pull llama3.2:1b

# Salir
exit
```

### Ver logs de un servicio específico

```bash
docker compose logs -f ollama
docker compose logs -f piper
docker compose logs -f app
```

### Reiniciar un servicio

```bash
docker compose restart app
docker compose restart ollama
```

### Detener todos los servicios

```bash
docker compose down
```

### Reconstruir después de cambios en el código

```bash
docker compose down
docker compose build app
docker compose up -d
```

## 🛠️ Desarrollo

### Editar el código

El código de `app/server.py` está montado como volumen, por lo que puedes editarlo directamente y reiniciar el contenedor:

```bash
# Editar el archivo
nano app/server.py

# Reiniciar solo la app
docker compose restart app
```

### Agregar dependencias Python

1. Editar `requirements.txt`
2. Reconstruir la imagen:

```bash
docker compose build app
docker compose up -d app
```

## 📊 Monitoreo

### Verificar estado de los servicios

```bash
docker compose ps
```

### Verificar uso de GPU

```bash
# En el host
tegrastats

# Dentro del contenedor
docker exec -it pipecat-app nvidia-smi
```

### Verificar endpoints

```bash
# Ollama
curl http://localhost:11434/api/tags

# Piper TTS
curl http://localhost:5000/

# App (si tiene endpoint de health)
curl http://localhost:8000/health
```

## ⚠️ Troubleshooting

### Error: "CUDA out of memory"

- Reducir el tamaño del modelo de Ollama
- Usar `llama3.2:1b` en lugar de modelos más grandes
- Reducir `LLM_MAX_TOKENS` en `.env`

### Error: "Failed to connect to Ollama"

```bash
# Verificar que Ollama esté corriendo
docker compose logs ollama

# Verificar que el modelo esté descargado
docker exec -it ollama ollama list
```

### Error: "Piper model not found"

```bash
# Verificar que los archivos del modelo existan
ls -la docker/models/piper/

# Deben estar:
# - es_MX-claude-high.onnx
# - es_MX-claude-high.onnx.json
```

### Rendimiento lento

- Asegúrate de que `runtime: nvidia` esté configurado
- Verifica que la GPU esté siendo utilizada: `tegrastats`
- Considera usar modelos más pequeños
- Ajusta `OMP_NUM_THREADS` y otros parámetros de CUDA en el Dockerfile

## 🔐 Seguridad

Para producción:

1. Cambiar puertos expuestos solo a localhost
2. Agregar autenticación a los endpoints
3. Usar variables de entorno para secretos
4. Configurar firewall en el Jetson

## 📝 Notas Adicionales

- El primer inicio puede tomar 30-40 segundos mientras carga los modelos de Silero VAD
- Los modelos de Ollama se descargan bajo demanda en el primer uso
- FAISS-GPU se compila durante la construcción del Dockerfile (puede tardar ~10 minutos)
- Se recomienda al menos 4GB de RAM libre para operación estable

## 🆘 Soporte

Para problemas o preguntas:
1. Revisa los logs: `docker compose logs -f`
2. Verifica la documentación de Jetson Containers
3. Consulta la documentación de Pipecat-AI

---

**Versión:** 1.0  
**Última actualización:** Octubre 2025
