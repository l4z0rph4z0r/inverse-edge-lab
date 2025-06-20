# Inverse Edge Lab - Project Information

## Project Overview
A multilingual (English/Italian) web application for analyzing trading inversions and measuring statistical edge with robust metrics and significance tests.

## Repository
- **GitHub URL**: https://github.com/l4z0rph4z0r/inverse-edge-lab
- **Branch**: main
- **Latest Commit**: Multilingual support implementation

## Key Features
1. **Multi-user Authentication**: Redis-based session management
2. **Trade Log Analysis**: Excel/CSV file processing with automatic column mapping
3. **Inverse Simulation**: Customizable TP/SL brackets for betting against traders
4. **Statistical Analysis**: Basic metrics + advanced significance tests (t-test, bootstrap CI, Jarque-Bera)
5. **AI Assistant**: Perplexity AI integration for result interpretation
6. **Multilingual**: Full English and Italian support

## Technology Stack
- **Frontend**: Streamlit
- **Backend**: Python 3.11
- **Database**: Redis (Upstash)
- **AI**: Perplexity AI API
- **Containerization**: Docker & Docker Compose
- **CI/CD**: GitHub Actions

## Environment Variables
```bash
# Redis connection (Upstash)
UPSTASH_REDIS_REST_URL=your_redis_url
UPSTASH_REDIS_REST_TOKEN=your_redis_token

# Perplexity AI (currently hardcoded in app.py - should be moved to env)
PERPLEXITY_API_KEY=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
```

## Local Development
```bash
# Clone the repository
git clone https://github.com/l4z0rph4z0r/inverse-edge-lab.git
cd inverse-edge-lab

# Run with Docker
docker compose up -d

# Or run locally
pip install -r requirements.txt
streamlit run app.py
```

## Default Credentials
- Username: `admin`
- Password: `admin123`

## File Structure
```
inverse-edge-lab/
├── app.py                 # Main application
├── translations.py        # Multilingual support (EN/IT)
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-container setup
├── README.md            # User documentation
├── PROJECT_INFO.md      # This file - developer reference
├── .gitignore           # Git exclusions
├── .github/
│   └── workflows/
│       └── ci.yml       # GitHub Actions CI/CD
└── tests/
    └── test_e2e.py      # End-to-end tests
```

## Key Algorithms

### Inverse Trading Logic
When betting AGAINST a trader:
- Their run-up → Our drawdown (adverse movement)
- Their drawdown → Our run-up (favorable movement)

Based on D/R flag:
- **RP (Run-up first)**: Trader profits first → We face loss first
- **DD (Drawdown first)**: Trader loses first → We profit first

### Statistical Tests
1. **T-Test**: Tests if mean P/L ≠ 0
2. **Bootstrap CI**: 1000 resamples for robust confidence intervals
3. **Jarque-Bera**: Tests for normal distribution

## How to Resume Work in Future Claude Code Sessions

### Method 1: Direct GitHub Clone (Recommended)
```bash
# In a new Claude Code session:
git clone https://github.com/l4z0rph4z0r/inverse-edge-lab.git
cd inverse-edge-lab
claude code .
```

### Method 2: From Your Local Directory
```bash
# Navigate to where the project is saved
cd /mnt/c/Claude_Code/inverse-edge-lab
claude code .
```

### Method 3: Reference This Session
When starting a new Claude Code session, you can say:
"I have a project called 'Inverse Edge Lab' in my GitHub at https://github.com/l4z0rph4z0r/inverse-edge-lab. Please clone it and continue working on it."

## Important Context for Future Sessions

### What to Tell Claude Code:
1. "This is a Streamlit trading analysis app with multilingual support"
2. "It uses Docker for containerization and Redis for auth"
3. "The Perplexity AI key is already in the code"
4. "Default login is admin/admin123"

### Current Status:
- ✅ Core functionality complete
- ✅ Multilingual support (EN/IT)
- ✅ Docker containerization
- ✅ Statistical analysis with significance tests
- ✅ AI assistant integration

### Potential Future Enhancements:
1. Move API keys to environment variables
2. Add user registration system
3. Implement real Redis authentication
4. Add more languages
5. Enhanced visualizations (candlestick charts, heatmaps)
6. Export reports as PDF
7. Batch processing for multiple files
8. Historical analysis tracking
9. Performance optimizations for large datasets
10. Additional statistical tests (Monte Carlo, Walk-forward analysis)

## Common Issues & Solutions

### Docker Build Cache
If changes aren't reflected:
```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

### File Upload Errors
Check column mapping in `load_data()` function. The app expects:
- P/L (points) or P/L (P)
- Drawdown (points) or DD (P)
- Run-up (points) or RP (P)
- D/R flag

### Language Not Switching
Clear browser cache or use incognito mode.

### Potential Future Enhancements:
1. ~~Move API keys to environment variables~~ ✅ DONE
2. Add user registration system
3. Implement real Redis authentication
4. Add more languages
5. Enhanced visualizations (candlestick charts, heatmaps)
6. ~~Export reports as PDF~~ ✅ DONE
7. Batch processing for multiple files
8. Historical analysis tracking
9. Performance optimizations for large datasets
10. Additional statistical tests (Monte Carlo, Walk-forward analysis)
11. ~~Add commission calculations~~ ✅ DONE
12. Add support for different contract types (futures, forex)
13. Implement position sizing calculations
14. Add risk management metrics (VaR, CVaR)

## Recent Updates (2025-06-16)

### Commission Feature Implementation
- **Problem Identified**: Commission calculations were mixing units (dollars vs points)
- **Solution**: Proper conversion using point value ($2 for MNQ, $5 for MES, etc.)
- **New Features**:
  - Contract size input
  - Commission per side input
  - Point value configuration
  - Display both points and dollars
  - Commission impact analysis section
  - Updated Mathematical Framework with commission explanations

### Repository Cleanup
- Removed Claude Code workflow files that were accidentally included
- Updated .gitignore to be project-specific
- Added comprehensive README with hosting instructions
- Made repository focused solely on Inverse Edge Lab

### Key Learnings for Future Development
1. **Unit Consistency**: Always track whether values are in points or dollars
2. **Commission Impact**: Small edges can be eliminated by commissions - must account for this
3. **Docker Performance**: Use `docker system prune -f` to clean cache when slow
4. **Repository Hygiene**: Keep project files separate from development environment files

## Important for Resuming Work

### Current Docker Setup
```bash
cd 01-projects/CC-001-inverse-edge-lab/DEV
docker compose up -d
# Access at http://localhost:8501
```

### Testing Commission Feature
1. Upload the SL5TP30.xlsx file
2. Set commission to $0 first to see gross P/L
3. Set commission to $1.30 per side ($2.60 round trip)
4. Note the dramatic difference in net P/L
5. Commission in points = Commission in dollars ÷ Point value

### Mathematical Framework Update
The dashboard now shows:
- **Step 1**: Calculate gross P/L from inverse positions (in points)
- **Step 2**: Convert commission to points and subtract to get net P/L
- All metrics (Sharpe, MAR, etc.) use net P/L for accuracy

## Last Updated
2025-06-16 - Implemented commission handling and cleaned up repository