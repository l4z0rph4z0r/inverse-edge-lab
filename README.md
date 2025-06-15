# Inverse Edge Lab

A Python + Streamlit web application for analyzing trading inversions and measuring statistical edge with robust metrics and significance tests.

## Features

- **Multi-user authentication** with Redis session management
- **Trade log analysis** supporting Excel/CSV files with automatic column mapping
- **Inverse simulation** with customizable bracket pairs (TP/SL)
- **Statistical analysis** including:
  - Basic metrics: P/L, Hit Rate, Sharpe Ratio, Profit Factor, etc.
  - Advanced tests: t-tests, bootstrap confidence intervals, normality tests
- **Interactive visualizations** with equity curves and P/L distributions
- **AI Assistant** for exploring results

## Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd inverse-edge-lab

# Start the application
docker compose up -d

# Open in browser
# http://localhost:8501
```

Default credentials:
- Username: `admin`
- Password: `admin123`

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Input File Format

The application expects trade log files with these 6 columns:
1. **Symbol** (or will use default)
2. **Entry Time** (or will use current time)
3. **P/L (points)** or **P/L (P)**
4. **Drawdown (points)** or **DD (P)**
5. **Run-up (points)** or **RP (P)**
6. **D/R flag** (RP = run-up first, DD = drawdown first)

## Technologies Used

- **Frontend**: Streamlit
- **Backend**: Python with pandas, numpy, scipy, statsmodels
- **Database**: Redis (Upstash)
- **Containerization**: Docker & Docker Compose
- **Testing**: Pytest with Puppeteer for E2E tests

## Environment Variables

- `UPSTASH_REDIS_REST_URL`: Redis connection URL
- `UPSTASH_REDIS_REST_TOKEN`: Redis authentication token

## License

MIT