# Project Guidelines

## Project Lifecycle

### 1. Project Creation
```bash
# New project setup
cd ~/Claude_Code/01-projects
./new-project.sh CC-XXX-project-name
# Or use alias: ccnew CC-XXX-project-name

cd CC-XXX-project-name/DEV

# Initialize git
git init
git add .
git commit -m "Initial commit"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 2. Active Development (DEV)
- All active work happens in `DEV/`
- Commit frequently
- Test continuously
- Document changes

### 3. Archiving Releases (ARCHIVE)
```bash
# When ready for production
cd ..
cp -r DEV ARCHIVE/v1.0.0
cd ARCHIVE/v1.0.0
rm -rf venv  # Don't archive virtual environments
cd ../..
git tag -a v1.0.0 -m "First production release"
```

## Naming Conventions

### Projects
```
CC-001-web-scraper-engine
CC-002-pdf-intelligence
CC-003-trading-dashboard
CC-004-api-gateway
```

### Python Files
```python
# snake_case for files
user_service.py
data_processor.py
api_handler.py

# PascalCase for classes
class UserService:
    pass

class DataProcessor:
    pass
```

### Constants and Environment Variables
```python
# UPPER_SNAKE_CASE
API_TIMEOUT = 30
MAX_RETRIES = 3
DEFAULT_PAGE_SIZE = 100

# In .env
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
```

## Documentation Requirements

### Every Project Must Have:

1. **README.md**
```markdown
# Project Name

Brief description of what this project does.

## Features
- Feature 1
- Feature 2

## Quick Start
\```bash
# Setup instructions
\```

## API Endpoints (if applicable)
- GET /health
- POST /process

## Configuration
- Required environment variables
- Optional settings

## Development
- How to run tests
- How to contribute
```

2. **requirements.txt**
- Pin major versions
- Separate dev requirements
- Update regularly

3. **.env.example**
- All required variables
- Example values
- Clear descriptions

## Git Workflow

### Branch Names
```
main          # Production-ready code
dev           # Development branch
feature/add-auth     # New features
fix/memory-leak      # Bug fixes
refactor/clean-api   # Code improvements
```

### Commit Messages
```
feat: Add user authentication
fix: Resolve memory leak in processor
docs: Update API documentation
test: Add unit tests for auth service
refactor: Clean up API routes
chore: Update dependencies
```

### Commit Frequency
- Commit when a logical unit is complete
- At least once per day if actively working
- Before switching contexts
- After successful test runs

## Code Organization

### Separation of Concerns
```
src/
├── models/      # Data models only
├── services/    # Business logic
├── api/         # HTTP endpoints
├── utils/       # Shared utilities
└── db/          # Database operations
```

### Single Responsibility
- Each function does one thing
- Each class has one purpose
- Each module has clear scope
- Each service is independent

### Dependency Direction
```
API Routes
    ↓
Services
    ↓
Models/DB
```

## Testing Strategy

### Test Categories
1. **Unit Tests** - Test individual functions
2. **Integration Tests** - Test component interaction
3. **API Tests** - Test HTTP endpoints
4. **E2E Tests** - Test full workflows

### Test Coverage Goals
- New code: 80%+ coverage
- Critical paths: 95%+ coverage
- Utilities: 90%+ coverage
- Models: 100% coverage

### Test File Organization
```
tests/
├── unit/
│   ├── test_models.py
│   └── test_utils.py
├── integration/
│   └── test_services.py
├── api/
│   └── test_endpoints.py
└── conftest.py  # Shared fixtures
```

## Performance Guidelines

### Async by Default
- Use async/await for I/O
- Don't block the event loop
- Use connection pooling
- Implement proper timeouts

### Resource Management
```python
# Good: Context managers
async with httpx.AsyncClient() as client:
    response = await client.get(url)

# Good: Connection pooling
from functools import lru_cache

@lru_cache()
def get_db_pool():
    return create_pool(...)
```

### Monitoring
- Log important operations
- Track response times
- Monitor memory usage
- Set up alerts

## Security Principles

### Never Commit:
- API keys
- Passwords
- Private keys
- Customer data

### Always:
- Validate input
- Sanitize output
- Use HTTPS
- Implement auth
- Log security events

### Dependencies
- Regular updates
- Security audits
- License checks
- Vulnerability scans

## Deployment Readiness

### Checklist:
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Environment variables documented
- [ ] Logging configured
- [ ] Error handling complete
- [ ] Performance tested
- [ ] Security reviewed
- [ ] Backup strategy defined

## Code Review Guidelines

### Before Submitting:
1. Run all tests
2. Format code (black, isort)
3. Update documentation
4. Check for secrets
5. Review git diff

### Review Focus:
- Logic correctness
- Performance implications
- Security concerns
- Code readability
- Test coverage

Remember: Clean code is written for humans to read, not just computers to execute!