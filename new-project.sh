#!/bin/bash

# New Project Creation Script for Claude Code

if [ $# -eq 0 ]; then
    echo "Usage: $0 CC-XXX-project-name"
    echo "Example: $0 CC-004-web-scraper"
    exit 1
fi

PROJECT_NAME=$1
PROJECT_DIR="01-projects/$PROJECT_NAME"

# Validate project name format
if [[ ! $PROJECT_NAME =~ ^CC-[0-9]{3}-[a-z-]+$ ]]; then
    echo "Error: Project name must follow format CC-XXX-descriptive-name"
    echo "Example: CC-001-web-scraper"
    exit 1
fi

# Check if project already exists
if [ -d "$PROJECT_DIR" ]; then
    echo "Error: Project $PROJECT_NAME already exists"
    exit 1
fi

echo "🚀 Creating new project: $PROJECT_NAME"

# Create project structure
mkdir -p "$PROJECT_DIR"/{DEV,ARCHIVE}
cd "$PROJECT_DIR/DEV"

# Initialize git
git init
echo "venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo ".env" >> .gitignore
echo "*.log" >> .gitignore
echo ".DS_Store" >> .gitignore

# Create source directories
mkdir -p src tests docs

# Create README
cat > README.md << EOF
# $PROJECT_NAME

## Description
Brief description of what this project does.

## Features
- Feature 1
- Feature 2
- Feature 3

## Quick Start

\`\`\`bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python src/main.py
\`\`\`

## Configuration

Copy \`.env.example\` to \`.env\` and configure:
\`\`\`
cp .env.example .env
\`\`\`

## Development

\`\`\`bash
# Run tests
pytest

# Format code
black src tests

# Lint
flake8 src tests
\`\`\`

## API Documentation

(Add API endpoints here if applicable)

## License

Proprietary
EOF

# Create .env.example
cat > .env.example << EOF
# Environment Configuration
DEBUG=True
LOG_LEVEL=INFO

# API Keys (get from respective services)
# OPENAI_API_KEY=your-key-here
# DATABASE_URL=postgresql://user:pass@localhost/dbname

# Application Settings
APP_NAME=$PROJECT_NAME
APP_VERSION=0.1.0
EOF

# Create requirements.txt
cat > requirements.txt << EOF
# Core dependencies
python-dotenv==1.0.0
pydantic==2.5.0
httpx==0.25.2

# Web framework (uncomment as needed)
# fastapi==0.104.1
# uvicorn[standard]==0.24.0
# streamlit==1.29.0

# Data processing (uncomment as needed)
# pandas==2.1.4
# numpy==1.26.2

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0

# Development tools
black==23.12.0
flake8==6.1.0
mypy==1.7.1
EOF

# Create basic source file
cat > src/main.py << EOF
"""
Main application entry point for $PROJECT_NAME
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Main application logic"""
    print(f"Starting {os.getenv('APP_NAME', '$PROJECT_NAME')}...")
    # Add your application logic here

if __name__ == "__main__":
    main()
EOF

# Create __init__.py files
touch src/__init__.py
touch tests/__init__.py

# Create basic test file
cat > tests/test_main.py << EOF
"""
Tests for $PROJECT_NAME
"""
import pytest
from src.main import main

def test_main():
    """Test main function"""
    # Add your tests here
    assert True

if __name__ == "__main__":
    pytest.main([__file__])
EOF

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install basic dependencies
pip install python-dotenv pytest black flake8

# Initial commit
git add .
git commit -m "Initial project structure for $PROJECT_NAME"

echo "✅ Project $PROJECT_NAME created successfully!"
echo ""
echo "Next steps:"
echo "1. cd $PROJECT_DIR/DEV"
echo "2. source venv/bin/activate"
echo "3. pip install -r requirements.txt"
echo "4. Edit README.md and add project details"
echo "5. Start coding in src/"