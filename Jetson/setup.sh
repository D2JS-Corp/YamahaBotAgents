#!/bin/bash

# Script de inicialización para Yamaha Bot en Jetson
# Este script prepara el entorno y descarga los modelos necesarios

set -e

echo "🚀 Iniciando configuración de Yamaha Bot para Jetson..."

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para mensajes
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar que estamos en el directorio correcto
if [ ! -f "docker/compose.yaml" ]; then
    error "Por favor ejecuta este script desde el directorio Jetson/"
    exit 1
fi

# Crear directorios necesarios
info "Creando estructura de directorios..."
mkdir -p docker/models/piper
mkdir -p docker/models/ollama

# Copiar modelos de Piper si existen
if [ -f "../ES/es_MX-claude-high.onnx" ]; then
    info "Copiando modelo de Piper desde ../ES/..."
    cp ../ES/es_MX-claude-high.onnx docker/models/piper/
    cp ../ES/es_MX-claude-high.onnx.json docker/models/piper/
elif [ -f "../PROD_ES/es_MX-claude-high.onnx" ]; then
    info "Copiando modelo de Piper desde ../PROD_ES/..."
    cp ../PROD_ES/es_MX-claude-high.onnx docker/models/piper/
    cp ../PROD_ES/es_MX-claude-high.onnx.json docker/models/piper/
else
    warn "No se encontró el modelo de Piper. Deberás descargarlo manualmente."
    warn "Ejecuta: python -m piper.download_voices es_MX-claude-high"
fi

# Verificar archivos del modelo
if [ -f "docker/models/piper/es_MX-claude-high.onnx" ]; then
    info "✅ Modelo de Piper encontrado"
else
    error "❌ Modelo de Piper no encontrado en docker/models/piper/"
    error "Por favor descarga el modelo antes de continuar"
    exit 1
fi

# Crear archivo .env si no existe
if [ ! -f ".env" ]; then
    info "Creando archivo .env desde .env.example..."
    cp .env.example .env
else
    info "Archivo .env ya existe, saltando..."
fi

# Verificar que Docker esté corriendo
if ! docker info > /dev/null 2>&1; then
    error "Docker no está corriendo. Por favor inicia Docker primero."
    exit 1
fi

info "✅ Verificación de Docker completa"

# Verificar NVIDIA Container Runtime
if ! docker run --rm --runtime=nvidia nvidia/cuda:11.4.0-base nvidia-smi > /dev/null 2>&1; then
    warn "No se pudo verificar NVIDIA Container Runtime"
    warn "Asegúrate de que JetPack esté instalado correctamente"
fi

echo ""
info "📋 Configuración completada!"
echo ""
echo "Siguiente paso: construir e iniciar los servicios"
echo ""
echo "  cd docker"
echo "  docker compose build"
echo "  docker compose up -d"
echo ""
echo "Para ver los logs:"
echo "  docker compose logs -f app"
echo ""
