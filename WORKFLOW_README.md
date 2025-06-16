# Claude Code Development Environment

A structured development environment optimized for Claude Code with best practices for project organization, context persistence, and efficient workflows.

## 🚀 Quick Setup

```bash
# Clone or navigate to this directory
cd ~/Claude_Code

# Run the setup script
./setup.sh

# Activate aliases
source ~/.bashrc

# See available commands
cchelp
```

## 📁 Directory Structure

```
Claude_Code/
├── 01-projects/              # Active development projects
│   └── CC-XXX-project-name/
│       ├── DEV/              # Active development
│       └── ARCHIVE/          # Production releases
├── 02-libraries/             # Shared code libraries
├── 03-native-tools/          # CLI tools and utilities
├── 04-infrastructure/        # Scripts and configs
├── 05-documentation/         # Project documentation
├── 06-data/                  # Data storage
├── 07-testing/               # Test suites
├── 08-operations/            # Ops scripts and monitoring
├── instance-docs/            # Claude instance memory
├── archives/                 # Old projects
├── .env                      # Environment variables
└── *.pem                     # Private keys (gitignored)
```

## 🧠 Instance Documentation

The `instance-docs/` folder helps maintain context across Claude Code sessions:

- **temporal-awareness.md** - Reminder to check current date/time before API lookups
- **recovery-protocol.md** - Steps to recover context after session loss
- **python-standards.md** - Python coding standards and best practices
- **project-guidelines.md** - Project structure and workflow guidelines
- **mcp-settings-template.json** - MCP server configuration template

## 🏗️ Project Management

### Creating a New Project

```bash
# Using the script
ccnew CC-004-my-new-api

# Or manually
newproject CC-004-my-new-api
```

### Project Naming Convention

```
CC-XXX-descriptive-name
```
- `CC` = Claude Code prefix
- `XXX` = Three-digit number (001, 002, etc.)
- `descriptive-name` = Clear project purpose

### Working with Projects

```bash
# Navigate to projects
cdp

# Activate a project
activateproject CC-001-inverse-edge-lab

# Archive a release
archiveproject CC-001-api v1.0.0
```

## 🎯 Key Aliases

### Navigation
- `cc` - Go to Claude Code directory
- `cdp` - Go to projects directory
- `cdl` - Go to libraries directory

### Python Development
- `pyenv` - Create new virtual environment
- `pyact` - Activate virtual environment
- `pyformat` - Format code with black & isort
- `pytest` - Run tests

### Git Commands
- `gs` - Git status
- `qcommit` - Quick commit with message
- `gl` - Git log graph

### Claude Code Specific
- `ccstatus` - Show environment status
- `cccheck` - Full environment check
- `cchelp` - Show all available commands

## 🔧 Development Workflow

1. **Start New Project**
   ```bash
   ccnew CC-005-web-scraper
   cd 01-projects/CC-005-web-scraper/DEV
   pyact
   ```

2. **Development**
   - Work in the `DEV/` directory
   - Commit frequently
   - Follow project guidelines

3. **Testing**
   ```bash
   pytest
   pyformat
   pylint
   ```

4. **Archive Release**
   ```bash
   archiveproject CC-005-web-scraper v1.0.0
   git tag v1.0.0
   ```

## 🔐 Security Best Practices

1. **Never commit**:
   - `.env` files
   - API keys
   - Private keys
   - Customer data

2. **Always use**:
   - Environment variables
   - `.gitignore`
   - Input validation

## 🛠️ Troubleshooting

### Lost Context?
```bash
ccrecovery  # Show recovery protocol
ccstatus    # Check current environment
```

### Check Environment
```bash
cccheck     # Full environment check
```

### API/Package Versions
```bash
cctemporal  # Show temporal awareness reminder
```

## 📚 Additional Resources

- Read `instance-docs/` for detailed guidelines
- Use `cchelp` for command reference
- Check `.env.example` for required API keys

---

Built for developers using Claude Code who value organization and efficient workflows.