# Quick Deployment Instructions

## Step 1: Push to GitHub
```bash
cd /mnt/c/Claude_Code/01-projects/CC-001-inverse-edge-lab/DEV
git add .
git commit -m "Add DigitalOcean deployment configuration"
git push origin main
```

## Step 2: Deploy via DigitalOcean Dashboard

1. Go to https://cloud.digitalocean.com/apps
2. Click "Create App"
3. Select "GitHub" as source
4. Authorize and select the repository: `l4z0rph4z0r/inverse-edge-lab`
5. Select branch: `main`
6. Click "Next"
7. DigitalOcean will detect the Dockerfile automatically
8. Click "Edit Plan" and select "Basic - $5/mo"
9. Click "Next"

## Step 3: Configure Environment Variables

Add these in the DigitalOcean dashboard during setup:

1. `UPSTASH_REDIS_REST_URL` - Your Upstash Redis URL
2. `UPSTASH_REDIS_REST_TOKEN` - Your Upstash Redis token
3. `PERPLEXITY_API_KEY` - Your Perplexity API key (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)

## Step 4: Deploy

Click "Create Resources" and wait for deployment (usually 5-10 minutes).

## Alternative: Use DigitalOcean CLI

```bash
# Install doctl if not already installed
# https://docs.digitalocean.com/reference/doctl/how-to/install/

# Authenticate
doctl auth init

# Create the app
doctl apps create --spec app.yaml

# Get your app ID
doctl apps list

# Update environment variables
doctl apps update <APP_ID> --spec app.yaml
```

## Access Your App

Once deployed, your app will be available at:
`https://inverse-edge-lab-<random>-<hash>.ondigitalocean.app`

Default login:
- Username: `l4z0rph4z0r`
- Password: `HR58k$!B!8D@gQyT`

## Monitoring

View logs:
```bash
doctl apps logs <APP_ID> --type=run
```

Check deployment status:
```bash
doctl apps get <APP_ID>
```