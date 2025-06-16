# Claude Code Instructions

## Efficiency Guidelines

For maximum efficiency, whenever you need to perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially.

## Important Instance Documentation

You should proactively reference these files in `instance-docs/` when relevant:

- **When doing web searches or checking APIs**: Read `instance-docs/temporal-awareness.md` first
- **When starting a new session or lost**: Read `instance-docs/recovery-protocol.md`
- **When writing Python code**: Reference `instance-docs/python-standards.md`
- **When creating/organizing projects**: Check `instance-docs/project-guidelines.md`

## Development Environment Structure

### Directory Organization
```
Claude_Code/
├── 01-projects/          # Active projects (CC-XXX-name format)
├── 02-libraries/         # Shared code libraries
├── 03-native-tools/      # CLI tools and utilities
├── 04-infrastructure/    # Scripts and configs
├── 05-documentation/     # Project documentation
├── 06-data/             # Data storage
├── 07-testing/          # Test suites
├── 08-operations/       # Ops scripts
├── instance-docs/       # Context persistence docs
└── archives/            # Old projects
```

### Quick Commands
- `cchelp` - Show all available commands
- `ccnew CC-XXX-name` - Create new project
- `ccstatus` - Check environment status
- `cccheck` - Full environment check
- `activateproject CC-XXX-name` - Activate project

### Context Recovery
If you lose context:
1. Read `instance-docs/recovery-protocol.md`
2. Run `ccstatus` to check current environment
3. Review recent git commits and project structure

## Current Active Projects

### CC-001: Inverse Edge Lab
- **GitHub**: https://github.com/l4z0rph4z0r/inverse-edge-lab
- **Type**: Multilingual trading analysis web app (Streamlit)
- **Languages**: English & Italian
- **Stack**: Python, Streamlit, Docker, Redis, Perplexity AI
- **Key Files**: 
  - `app.py` - Main application
  - `translations.py` - Multilingual support
  - `PROJECT_INFO.md` - Developer reference
- **Login**: admin/admin123
- **Docker**: `docker compose up -d`

## Important Instruction Reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.