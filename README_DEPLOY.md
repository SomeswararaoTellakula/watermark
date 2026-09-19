# Render deployment guide

This project includes a Flask backend and a Next.js frontend. For Render, keep them as two separate web services.

## 1) Push this repo to GitHub

Initialize git in this folder if needed:

```bash
git init
git add .
git commit -m "Initial Pentamark deploy setup"
git branch -M main
git remote add origin https://github.com/SomeswararaoTellakula/watermark.git
git push -u origin main
```

## 2) Create the backend service on Render

- Use the GitHub repo you linked
- Root directory: `DiffMark-main/webapp`
- Build command:

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

- Start command:

```bash
PORT=$PORT python app.py
```

## 3) Create the frontend service on Render

- Use the same repo
- Root directory: `diffmark-frontend`
- Build command:

```bash
npm install && npm run build
```

- Start command:

```bash
npm run start -- --hostname 0.0.0.0 --port $PORT
```

- Environment variable:

```bash
NEXT_PUBLIC_API_URL=https://YOUR_BACKEND_URL.onrender.com/api
```

## 4) Important notes

- The backend listens on `PORT` from Render.
- The frontend is configured to send API calls to `NEXT_PUBLIC_API_URL` when present.
- The Flask app also accepts CORS requests for `/api/*` to work with the frontend on a different host.
- If MongoDB is required, add a MongoDB service and set `MONGO_URI`.

## 5) Deployment status check

After deployment, confirm:

- Frontend loads at your Render URL
- Backend health route responds: `https://YOUR_BACKEND_URL.onrender.com/api/health`
- Auth and watermark endpoints respond without localhost assumptions
