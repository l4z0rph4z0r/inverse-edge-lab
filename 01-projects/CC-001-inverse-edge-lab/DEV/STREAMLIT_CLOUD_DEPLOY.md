# Deploy to Streamlit Community Cloud (FREE)

## Steps:

1. **Sign up at** https://streamlit.io/cloud
   - Use your GitHub account

2. **New App Deployment**
   - Click "New app"
   - Repository: `l4z0rph4z0r/inverse-edge-lab`
   - Branch: `development`
   - Main file path: `01-projects/CC-001-inverse-edge-lab/DEV/app.py`

3. **Advanced Settings** (click before deploying)
   - Python version: 3.11
   
4. **Secrets Management**
   Click "Advanced settings" > "Secrets" and add:
   ```toml
   UPSTASH_REDIS_REST_URL = "your-redis-url"
   UPSTASH_REDIS_REST_TOKEN = "your-redis-token"
   PERPLEXITY_API_KEY = "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"
   ```

5. **Deploy!**

## Your app will be available at:
`https://inverse-edge-lab.streamlit.app`

## Advantages:
- ✅ Completely FREE
- ✅ No sleep/timeout
- ✅ Custom domain support
- ✅ Automatic deployments from GitHub
- ✅ Built for Streamlit
- ✅ Handles secrets properly

## Sharing with Colleague:
Once deployed, share the URL. They can login with:
- Username: `l4z0rph4z0r`
- Password: `HR58k$!B!8D@gQyT`

Or you can create a token for them in the admin panel!