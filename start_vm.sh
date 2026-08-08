#!/bin/bash
# Acompanhe o progresso via SSH: tail -f /var/log/cloud-init-output.log

set -e  # para o script se algum comando falhar

REPO_URL="github.com/MatheusMoselli/mouse-dynamics"
REPO_DIR="/root/projeto"

echo ">>> Atualizando sistema..."
apt-get update -y
apt-get upgrade -y

echo ">>> Instalando dependências do sistema..."
apt-get install -y python3-pip python3-venv git tmux htop

echo ">>> Clonando repositório..."
rm -rf /root/projeto
git clone "https://${REPO_URL}" "$REPO_DIR"

cd "$REPO_DIR"

echo ">>> Criando ambiente virtual e instalando dependências Python..."
python3 -m venv venv
source venv/bin/activate

if [ -f "pyproject.toml" ]; then
    # Fallback: pip padrão instalando o pacote a partir do pyproject.toml
    echo ">>> Nenhum lockfile específico encontrado - instalando via pip padrão"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -e .
else
    echo "AVISO: pyproject.toml não encontrado em $REPO_DIR - instale manualmente depois"
fi

echo ">>> Ambiente pronto em $REPO_DIR"

echo ">>> Setup finalizado em $(date)"