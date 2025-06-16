#!/bin/bash

# Claude Code Development Environment Setup Script

echo "🚀 Setting up Claude Code Development Environment..."

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo "Please don't run this script as root"
   exit 1
fi

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential tools
echo "🔧 Installing essential tools..."
sudo apt install -y \
    curl \
    wget \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    nodejs \
    npm \
    sqlite3 \
    redis-server \
    htop \
    tree \
    jq \
    dos2unix

# Install Python tools
echo "🐍 Installing Python tools..."
pip3 install --user \
    pipx \
    virtualenv \
    black \
    isort \
    flake8 \
    mypy \
    pytest \
    pre-commit

# Add pipx to PATH
export PATH="$PATH:$HOME/.local/bin"
echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc

# Install Claude Code (if available)
echo "🤖 Checking for Claude Code..."
if command -v claude &> /dev/null; then
    echo "✅ Claude Code already installed"
else
    echo "❌ Claude Code not found. Please install from Anthropic"
fi

# Setup git
echo "📝 Configuring git..."
if [ -z "$(git config --global user.name)" ]; then
    read -p "Enter your git username: " git_username
    git config --global user.name "$git_username"
fi

if [ -z "$(git config --global user.email)" ]; then
    read -p "Enter your git email: " git_email
    git config --global user.email "$git_email"
fi

git config --global init.defaultBranch main

# Create .env from example
if [ -f ".env.example" ] && [ ! -f ".env" ]; then
    echo "🔐 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys"
fi

# Setup Claude MCP settings
if [ ! -d "$HOME/.claude" ]; then
    echo "📁 Creating Claude config directory..."
    mkdir -p "$HOME/.claude"
fi

if [ -f "instance-docs/mcp-settings-template.json" ]; then
    echo "⚙️  Setting up MCP configuration..."
    cp instance-docs/mcp-settings-template.json "$HOME/.claude/settings.local.json"
    echo "✅ MCP settings configured"
fi

# Setup aliases
echo "🎯 Setting up aliases..."
bash setup-aliases.sh

# Create example project structure
echo "📂 Creating example project..."
if [ ! -d "01-projects/CC-001-inverse-edge-lab" ]; then
    mkdir -p 01-projects/CC-001-inverse-edge-lab/{DEV,ARCHIVE}
    echo "✅ Created example project structure"
fi

# Final setup
echo ""
echo "✨ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your API keys"
echo "2. Source your bashrc: source ~/.bashrc"
echo "3. Type 'cchelp' to see available commands"
echo "4. Create your first project: ccnew CC-002-my-project"
echo "5. Read instance-docs/ for guidelines and protocols"
echo ""
echo "Happy coding with Claude Code! 🚀"