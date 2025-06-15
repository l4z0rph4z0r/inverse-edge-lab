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
- **AI Assistant**: Perplexity AI
- **Containerization**: Docker & Docker Compose
- **Testing**: Pytest with Puppeteer for E2E tests

## Language Support

The application supports:
- 🇬🇧 English
- 🇮🇹 Italian

Switch languages using the selector in the top-right corner.

## Environment Variables

- `UPSTASH_REDIS_REST_URL`: Redis connection URL
- `UPSTASH_REDIS_REST_TOKEN`: Redis authentication token

## Production Deployment

### Using Docker (Recommended)

1. **On a VPS/Cloud Server**:
```bash
# Clone the repository
git clone https://github.com/l4z0rph4z0r/inverse-edge-lab.git
cd inverse-edge-lab

# Create .env file with your credentials
echo "UPSTASH_REDIS_REST_URL=your_url_here" > .env
echo "UPSTASH_REDIS_REST_TOKEN=your_token_here" >> .env

# Run in production
docker compose up -d

# To update to latest version
git pull
docker compose down
docker compose up -d --build
```

2. **Using a Cloud Platform**:
- **Heroku**: Use the included Dockerfile
- **AWS ECS**: Push to ECR and deploy
- **Google Cloud Run**: Deploy directly from GitHub
- **DigitalOcean App Platform**: Connect GitHub repo

### Security Considerations

For production:
1. Change default admin credentials in `app.py`
2. Move API keys to environment variables
3. Use HTTPS (reverse proxy with nginx/traefik)
4. Set up proper Redis authentication
5. Implement rate limiting

## License

MIT