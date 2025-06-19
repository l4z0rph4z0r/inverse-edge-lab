# DigitalOcean Deployment Guide for Inverse Edge Lab

## Prerequisites
1. A DigitalOcean account
2. The DigitalOcean CLI (`doctl`) installed and authenticated
3. Your environment variables ready:
   - `UPSTASH_REDIS_REST_URL` (optional - uses DigitalOcean Redis if not provided)
   - `UPSTASH_REDIS_REST_TOKEN` (optional)
   - `PERPLEXITY_API_KEY` (required for AI features)

## Deployment Steps

### Option 1: Deploy from GitHub (Recommended)

1. Push your code to GitHub:
   ```bash
   git add .
   git commit -m "Prepare for DigitalOcean deployment"
   git push origin main
   ```

2. Create the app using the DigitalOcean CLI:
   ```bash
   doctl apps create --spec app.yaml
   ```

3. Set your environment variables in the DigitalOcean dashboard:
   - Go to your app in the DigitalOcean dashboard
   - Navigate to Settings > App-Level Environment Variables
   - Add your secrets:
     - `UPSTASH_REDIS_REST_URL`
     - `UPSTASH_REDIS_REST_TOKEN`
     - `PERPLEXITY_API_KEY`

### Option 2: Deploy using DigitalOcean MCP

If you have the DigitalOcean MCP configured, you can deploy programmatically:

```python
# This will be handled by the Claude Code MCP integration
```

### Option 3: Manual Deployment via Dashboard

1. Go to [DigitalOcean App Platform](https://cloud.digitalocean.com/apps)
2. Click "Create App"
3. Connect your GitHub repository
4. Select the branch (main)
5. DigitalOcean will auto-detect the Dockerfile
6. Configure environment variables
7. Choose instance size (Basic XXS is sufficient for testing)
8. Deploy!

## Post-Deployment

1. Access your app at: `https://inverse-edge-lab-xxxxx.ondigitalocean.app`
2. Default credentials:
   - Username: `l4z0rph4z0r`
   - Password: `HR58k$!B!8D@gQyT`

## Monitoring

- Check app logs: `doctl apps logs <app-id>`
- View metrics in the DigitalOcean dashboard
- Health check endpoint: `/_stcore/health`

## Scaling

To scale your app:
```bash
doctl apps update <app-id> --spec app.yaml
```

Update `instance_count` in app.yaml to add more instances.

## Cost Estimate

- Basic XXS instance: ~$5/month
- Redis database: ~$15/month (or use external Upstash)
- Total: ~$20/month for basic deployment

## Troubleshooting

1. If the app fails to start, check logs:
   ```bash
   doctl apps logs <app-id> --type=run
   ```

2. Common issues:
   - Missing environment variables
   - Port configuration (must be 8501)
   - Health check failures (increase timeout if needed)

3. For Redis connection issues:
   - Ensure the Redis URL is properly formatted
   - Check firewall rules if using external Redis