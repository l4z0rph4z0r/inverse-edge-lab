# Recovery Protocol

## When You Lose Context

If you experience:
- Session timeout
- Memory loss
- Confusion about current task
- Uncertainty about project state

Follow these steps:

## 1. Immediate Context Check

```bash
# Check environment
date && pwd && whoami

# Check git status
git status
git log --oneline -10

# Check running processes
ps aux | grep python
ps aux | grep node
```

## 2. Read Recovery Documents

1. Check this folder for context
2. Read project README files
3. Check recent commits
4. Look for TODO.md or PLAN.md files

## 3. Project State Recovery

```bash
# Find active projects
ls -la ~/01-projects/

# Check for running services
lsof -i :8000  # FastAPI default
lsof -i :3000  # Frontend default
lsof -i :8501  # Streamlit default

# Check virtual environments
find . -name "venv" -type d

# Look for recent logs
find . -name "*.log" -mtime -1
```

## 4. Re-establish Working Context

### For Python Projects:
```bash
cd [project]/DEV
source venv/bin/activate
pip list
python --version
```

### For Node Projects:
```bash
cd [project]/DEV
npm list
node --version
```

## 5. Common Recovery Scenarios

### Lost in Middle of Coding
1. Check git diff
2. Read last few commits
3. Check IDE history
4. Look for TODO comments

### Forgot Project Purpose
1. Read README.md
2. Check project folder name
3. Look at main application file
4. Review test files

### Unknown Error State
1. Check error logs
2. Review recent changes
3. Revert if necessary
4. Start from last known good state

## 6. Recovery Checklist

- [ ] Current date/time verified
- [ ] Working directory confirmed
- [ ] Git status checked
- [ ] Virtual environment activated
- [ ] Dependencies verified
- [ ] Last task identified
- [ ] Next steps clear

## Prevention Tips

1. Commit frequently with descriptive messages
2. Update README.md regularly
3. Keep TODO.md current
4. Use meaningful branch names
5. Document decisions in code comments

Remember: It's okay to lose context. These protocols help you recover quickly!