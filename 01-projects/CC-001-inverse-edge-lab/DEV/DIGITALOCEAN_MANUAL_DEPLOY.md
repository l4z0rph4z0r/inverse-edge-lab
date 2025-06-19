# DigitalOcean Manual Deployment Guide

Since GitHub integration requires OAuth authentication through the DigitalOcean dashboard, please follow these manual steps:

## Step 1: Access DigitalOcean Apps

1. Go to https://cloud.digitalocean.com/apps
2. Click the blue "Create App" button

## Step 2: Connect GitHub Repository

1. Select "GitHub" as your source
2. Click "Manage Access" if this is your first time
3. Authorize DigitalOcean to access your GitHub account
4. Select the repository: `l4z0rph4z0r/inverse-edge-lab`
5. Select branch: `development` (important: NOT main)
6. Check "Autodeploy" - Deploy on push
7. Click "Next"

## Step 3: Resources Configuration

DigitalOcean will automatically detect your Dockerfile. You should see:
- **Type**: Web Service
- **Resource Name**: web
- **Build Context**: /
- **Dockerfile Path**: Dockerfile

Click "Next"

## Step 4: Environment Variables

Click "Edit" next to the web service, then add these environment variables:

1. **UPSTASH_REDIS_REST_URL**
   - Type: Secret
   - Value: (your Upstash Redis URL)

2. **UPSTASH_REDIS_REST_TOKEN**
   - Type: Secret
   - Value: (your Upstash Redis token)

3. **PERPLEXITY_API_KEY**
   - Type: Secret
   - Value: `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`

4. **PYTHONUNBUFFERED**
   - Type: Plain text
   - Value: `1`

Click "Save" and then "Next"

## Step 5: Info & Billing

1. **App Name**: inverse-edge-lab (or keep auto-generated)
2. **Region**: New York (NYC) - closest to you
3. **Plan**: Basic - $5.00/month
   - Instance Size: Basic XXS (512 MB RAM, 1 vCPU)

Click "Next"

## Step 6: Review & Create

Review all settings:
- GitHub repo connected
- Dockerfile detected
- Environment variables set
- Basic plan ($5/month)

Click "Create Resources"

## Step 7: Deployment

The initial deployment will take 5-10 minutes. You can watch the build logs in real-time.

Once deployed, you'll get a URL like:
`https://inverse-edge-lab-xxxxx.ondigitalocean.app`

## Step 8: Verify Deployment

1. Visit your app URL
2. You should see the Streamlit app loading
3. Login with:
   - Username: `l4z0rph4z0r`
   - Password: `HR58k$!B!8D@gQyT`

## Troubleshooting

If the app fails to start:
1. Check the "Runtime Logs" tab
2. Common issues:
   - Missing environment variables
   - Port mismatch (must be 8501)
   - Health check timeout (increase if needed)

## Next Steps

1. Set up a custom domain (optional)
2. Configure alerts
3. Monitor app performance
4. Set up backups if needed

Your app will automatically redeploy whenever you push to the `development` branch!