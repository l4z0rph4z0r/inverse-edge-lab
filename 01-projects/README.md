# Active Projects Directory

This directory contains all active development projects following the naming convention:
```
CC-XXX-descriptive-name
```

Where:
- `CC` = Claude Code prefix
- `XXX` = Three-digit number (001, 002, etc.)
- `descriptive-name` = Clear project purpose

## Project Structure

Each project follows this structure:
```
CC-001-project-name/
├── DEV/              # Active development
│   ├── src/          # Source code
│   ├── tests/        # Test files
│   ├── README.md     # Project documentation
│   ├── requirements.txt
│   └── .env.example
└── ARCHIVE/          # Production releases
    ├── v1.0.0/
    ├── v1.1.0/
    └── v2.0.0/
```

## Creating a New Project

Use the provided script:
```bash
./new-project.sh CC-004-my-new-api
```

Or manually:
```bash
mkdir -p CC-XXX-project-name/{DEV,ARCHIVE}
cd CC-XXX-project-name/DEV
git init
python3 -m venv venv
source venv/bin/activate
```

## Current Projects

<!-- List active projects here -->
- CC-001-inverse-edge-lab (Trading Analysis Platform)