#!/bin/bash

# Claude Code Bash Aliases Setup Script

echo "🎯 Setting up Claude Code aliases..."

# Backup existing .bashrc
if [ -f ~/.bashrc ]; then
    cp ~/.bashrc ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)
    echo "✅ Backed up existing .bashrc"
fi

# Create aliases file
cat > ~/.claude_aliases << 'EOL'
# Claude Code Development Aliases

# Directory Navigation
alias cc='cd ~/Claude_Code'
alias cdp='cd ~/Claude_Code/01-projects'
alias cdl='cd ~/Claude_Code/02-libraries'
alias cdt='cd ~/Claude_Code/03-native-tools'
alias cdi='cd ~/Claude_Code/04-infrastructure'
alias cdd='cd ~/Claude_Code/05-documentation'
alias cdo='cd ~/Claude_Code/08-operations'
alias cda='cd ~/Claude_Code/archives'

# Common Operations
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'

# Python Virtual Environment
alias pyenv='python3 -m venv venv && source venv/bin/activate && pip install --upgrade pip'
alias pyact='source venv/bin/activate'
alias pydeact='deactivate'
alias pipr='pip install -r requirements.txt'
alias pipf='pip freeze > requirements.txt'

# Python Development
alias pytest='python -m pytest'
alias pycov='python -m pytest --cov=src --cov-report=html'
alias pyformat='black . && isort .'
alias pylint='flake8 src tests'
alias pytype='mypy src'
alias serve='python -m http.server 8000'
alias jup='jupyter notebook'

# FastAPI
alias fastapi='uvicorn main:app --reload'
alias fastapidev='uvicorn main:app --reload --host 0.0.0.0 --port 8000'

# Streamlit
alias strun='streamlit run'
alias stclear='streamlit cache clear'

# Git Aliases
alias gs='git status'
alias ga='git add'
alias gaa='git add .'
alias gc='git commit -m'
alias gca='git commit -am'
alias gp='git push'
alias gpl='git pull'
alias gl='git log --oneline --graph --decorate'
alias gd='git diff'
alias gds='git diff --staged'
alias gb='git branch'
alias gco='git checkout'
alias gcb='git checkout -b'
alias gm='git merge'
alias gr='git remote -v'
alias gst='git stash'
alias gstp='git stash pop'

# Docker (if using)
alias dps='docker ps'
alias dpsa='docker ps -a'
alias dimg='docker images'
alias dcu='docker compose up -d'
alias dcd='docker compose down'
alias dcr='docker compose restart'
alias dcl='docker compose logs -f'

# System Monitoring
alias ports='netstat -tulanp 2>/dev/null | grep LISTEN'
alias meminfo='free -h'
alias cpuinfo='lscpu'
alias diskinfo='df -h'
alias topten='ps aux | sort -nrk 3,3 | head -10'

# Claude Code Specific
alias ccstatus='date && pwd && whoami && echo "Claude Code Instance"'
alias ccrecovery='cat ~/Claude_Code/instance-docs/recovery-protocol.md'
alias cctemporal='cat ~/Claude_Code/instance-docs/temporal-awareness.md'
alias ccnew='~/Claude_Code/new-project.sh'
alias ccenv='cp ~/Claude_Code/.env.example .env && echo "Created .env file - please edit with your keys"'

# Safety Aliases
alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'

# Quick Edit
alias editbash='nano ~/.bashrc'
alias editaliases='nano ~/.claude_aliases'
alias sourcebash='source ~/.bashrc'

# Search Functions
findpy() { find . -name "*.py" -type f | grep -i "$1"; }
findfile() { find . -name "*$1*" -type f; }
greppy() { grep -r --include="*.py" "$1" .; }
todos() { grep -r "TODO\|FIXME\|XXX" --include="*.py" --include="*.md" .; }

# Project Functions
newproject() {
    if [ -z "$1" ]; then
        echo "Usage: newproject CC-XXX-name"
        return 1
    fi
    ~/Claude_Code/new-project.sh "$1"
}

activateproject() {
    if [ -z "$1" ]; then
        echo "Usage: activateproject CC-XXX-name"
        return 1
    fi
    cd ~/Claude_Code/01-projects/"$1"/DEV && source venv/bin/activate
}

# Environment Check Function
cccheck() {
    echo "🔍 Claude Code Environment Check"
    echo "================================"
    date
    echo "User: $(whoami)"
    echo "Directory: $(pwd)"
    echo "Python: $(python3 --version)"
    echo "Git branch: $(git branch --show-current 2>/dev/null || echo 'Not in git repo')"
    if [ -n "$VIRTUAL_ENV" ]; then
        echo "Virtual env: $(basename $VIRTUAL_ENV)"
    else
        echo "Virtual env: None active"
    fi
    echo "================================"
}

# Quick commit function
qcommit() {
    if [ -z "$1" ]; then
        echo "Usage: qcommit 'commit message'"
        return 1
    fi
    git add .
    git commit -m "$1"
}

# Archive project function
archiveproject() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Usage: archiveproject CC-XXX-name version"
        echo "Example: archiveproject CC-001-api v1.0.0"
        return 1
    fi
    
    PROJECT_PATH="~/Claude_Code/01-projects/$1"
    if [ ! -d "$PROJECT_PATH" ]; then
        echo "Project $1 not found"
        return 1
    fi
    
    cp -r "$PROJECT_PATH/DEV" "$PROJECT_PATH/ARCHIVE/$2"
    echo "✅ Archived $1 as version $2"
}

# Show available Claude Code commands
cchelp() {
    echo "🚀 Claude Code Commands:"
    echo "========================"
    echo "Navigation:"
    echo "  cc         - Go to Claude Code directory"
    echo "  cdp        - Go to projects directory"
    echo "  ccstatus   - Show current environment status"
    echo ""
    echo "Projects:"
    echo "  ccnew      - Create new project"
    echo "  newproject - Create new project (function)"
    echo "  activateproject - Activate project environment"
    echo "  archiveproject - Archive project version"
    echo ""
    echo "Python:"
    echo "  pyenv      - Create new virtual environment"
    echo "  pyact      - Activate virtual environment"
    echo "  pyformat   - Format code with black & isort"
    echo "  pylint     - Run flake8 linter"
    echo "  pytest     - Run tests"
    echo ""
    echo "Git:"
    echo "  gs         - Git status"
    echo "  qcommit    - Quick commit with message"
    echo "  gl         - Git log graph"
    echo ""
    echo "Info:"
    echo "  cccheck    - Environment check"
    echo "  ccrecovery - Show recovery protocol"
    echo "  cctemporal - Show temporal awareness"
    echo "  cchelp     - Show this help"
}

# Initialize on terminal start
echo "🚀 Claude Code aliases loaded! Type 'cchelp' for commands."
EOL

# Add source command to .bashrc if not already present
if ! grep -q "source ~/.claude_aliases" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Claude Code Aliases" >> ~/.bashrc
    echo "if [ -f ~/.claude_aliases ]; then" >> ~/.bashrc
    echo "    source ~/.claude_aliases" >> ~/.bashrc
    echo "fi" >> ~/.bashrc
fi

echo "✅ Aliases setup complete!"
echo ""
echo "To activate aliases, run:"
echo "  source ~/.bashrc"
echo ""
echo "Or start a new terminal session."
echo ""
echo "Type 'cchelp' to see available commands!"