# Inverse Edge Lab 🔄

A sophisticated trading backtesting dashboard that analyzes inverse position strategies with fixed risk-reward brackets. Built with Streamlit, Docker, and Redis.

![Python](https://img.shields.io/badge/python-3.11-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.29.0-red)
![Docker](https://img.shields.io/badge/docker-ready-blue)

## Features

- **Inverse Position Simulation**: Test the profitability of taking opposite positions from losing traders
- **Commission Handling**: Accurate commission calculations with point-to-dollar conversion
- **Multi-language Support**: Available in English and Italian
- **Statistical Analysis**: Advanced significance tests (T-test, Bootstrap CI, Jarque-Bera)
- **PDF Reports**: Generate comprehensive reports with charts and metrics
- **Real-time Visualization**: Interactive charts showing P/L distribution and equity curves
- **AI Assistant**: Integrated Perplexity AI for result analysis

## Quick Start

### Using Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/l4z0rph4z0r/inverse-edge-lab.git
cd inverse-edge-lab/01-projects/CC-001-inverse-edge-lab/DEV
```

2. Create a `.env` file:
```bash
cp .env.example .env
# Edit .env and add your Perplexity API key (optional)
```

3. Run with Docker Compose:
```bash
docker compose up -d
```

4. Access the dashboard at `http://localhost:8501`
   - Default login: `admin` / `admin123`

### Manual Installation

1. Install Python 3.11+

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Hosting Instructions

### Deploy on Streamlit Cloud

1. Fork this repository
2. Sign up at [share.streamlit.io](https://share.streamlit.io)
3. Create new app and point to your fork
4. Set up secrets in the app settings:
   ```toml
   PERPLEXITY_API_KEY = "your-api-key"
   ```

### Deploy on Heroku

1. Create `Procfile`:
```
web: sh setup.sh && streamlit run app.py
```

2. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
```

3. Deploy:
```bash
heroku create your-app-name
heroku git:remote -a your-app-name
git push heroku main
```

### Deploy on AWS/GCP/Azure

Use the provided Docker image:

```bash
# Build image
docker build -t inverse-edge-lab .

# Run container
docker run -p 8501:8501 -e PERPLEXITY_API_KEY=your-key inverse-edge-lab
```

### Deploy with Docker on VPS

1. SSH to your server
2. Install Docker and Docker Compose
3. Clone the repository
4. Set up SSL with nginx-proxy or Traefik
5. Run:
```bash
docker compose up -d
```

## Configuration

### Environment Variables

- `PERPLEXITY_API_KEY`: API key for AI assistant (optional)
- `UPSTASH_REDIS_REST_URL`: Redis URL for session storage (optional)
- `UPSTASH_REDIS_REST_TOKEN`: Redis token (optional)

### Commission Settings

- **Contract Size**: Number of contracts per trade
- **Commission per Side**: Commission charged per contract per side
- **Point Value**: Dollar value per point (e.g., $2 for MNQ)

## Usage

1. **Upload Data**: Upload CSV/Excel files with trading data
2. **Configure Brackets**: Set Take Profit (TP) and Stop Loss (SL) levels
3. **Set Commissions**: Configure realistic commission rates
4. **Run Simulation**: Analyze results with comprehensive metrics
5. **Export Results**: Download CSV data or PDF reports

## Data Format

Required columns in uploaded files:
- `p/l (points)`: Profit/loss in points
- `run up (points)`: Maximum favorable excursion
- `drawdown (points)`: Maximum adverse excursion
- `d/r flag`: Which came first (RP/DD)
- `symbol`: Trading symbol (optional)
- `entry time`: Trade timestamp (optional)

## Development

### Project Structure
```
01-projects/CC-001-inverse-edge-lab/DEV/
├── app.py              # Main Streamlit application
├── translations.py     # Multi-language support
├── requirements.txt    # Python dependencies
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose setup
└── .env.example       # Environment variables template
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Open an issue on [GitHub](https://github.com/l4z0rph4z0r/inverse-edge-lab/issues)
- Check the [Mathematical Framework](https://github.com/l4z0rph4z0r/inverse-edge-lab#mathematical-framework) section for calculation details