# Hosting the bakup.ai Landing Page — Step by Step (0 → 100)

> **End result:** A public URL (e.g. `https://salilsankritya.github.io/bakup.ai`)
> where anyone can visit the landing page, enter the access key, and download
> the installer.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  GitHub Pages (FREE)                                     │
│  Serves: index.html, styles.css, fonts.css, app.js       │
│  URL: https://salilsankritya.github.io/bakup.ai          │
└───────────────────────┬──────────────────────────────────┘
                        │ "Download" button links to ↓
┌───────────────────────▼──────────────────────────────────┐
│  GitHub Releases (FREE, up to 2 GB per file)             │
│  Hosts: bakup-ai-installer.exe (241 MB)                  │
│  URL: github.com/…/releases/latest/download/…            │
└──────────────────────────────────────────────────────────┘
```

**Cost: $0.** GitHub Pages and Releases are free for public repos.

---

## Step 0 — Prerequisites

- A GitHub account (you already have: `salilsankritya`)
- Git installed on your machine
- The bakup.ai repo cloned locally (you have this)
- The built installer at `dist/bakup-ai-installer.exe` or `downloads/bakup-ai-installer.exe`

---

## Step 1 — Push latest code to GitHub

```powershell
cd c:\Users\91876\Documents\bakup.ai
git add -A
git commit -m "update landing page download URL to GitHub Releases"
git push origin main
```

---

## Step 2 — Enable GitHub Pages

1. Go to: **https://github.com/salilsankritya/bakup.ai/settings/pages**

2. Under **"Build and deployment"**:
   - **Source:** select **"Deploy from a branch"**
   - **Branch:** select **`main`**
   - **Folder:** select **`/ (root)`**

3. Click **Save**

4. Wait 1–2 minutes. GitHub builds and deploys

5. Your landing page is now live at:
   ```
   https://salilsankritya.github.io/bakup.ai
   ```

> **Note:** The `.nojekyll` file in the repo root tells GitHub Pages to serve
> files as-is without Jekyll processing. This is already included.

---

## Step 3 — Create a GitHub Release (upload the installer)

The installer is 241 MB — too large for git. GitHub Releases supports
files up to **2 GB**, perfect for this.

### Option A: Via GitHub Web UI (easiest)

1. Go to: **https://github.com/salilsankritya/bakup.ai/releases/new**

2. Fill in:
   - **Tag:** `v0.2.0` (or whatever version you're releasing)
   - **Release title:** `v0.2.0 — Developer Preview`
   - **Description:** Paste the changelog or a summary

3. **Drag-and-drop** the installer file into the "Attach binaries" area:
   ```
   c:\Users\91876\Documents\bakup.ai\downloads\bakup-ai-installer.exe
   ```
   (or from `dist\bakup-ai-installer.exe` — same file)

4. Check **"Set as the latest release"**

5. Click **"Publish release"**

### Option B: Via GitHub CLI (faster for future releases)

```powershell
# Install GitHub CLI (one-time)
winget install GitHub.cli

# Authenticate (one-time)
gh auth login

# Create release and upload installer
cd c:\Users\91876\Documents\bakup.ai
gh release create v0.2.0 `
  --title "v0.2.0 — Developer Preview" `
  --notes "Reasoning engine, log-to-code analysis, confidence scoring. See FEATURES.md for details." `
  downloads/bakup-ai-installer.exe
```

---

## Step 4 — Verify everything works

### 4a. Landing page loads
Open in browser:
```
https://salilsankritya.github.io/bakup.ai
```
You should see the full landing page with hero, features, changelog, etc.

### 4b. Access key gate works
1. Scroll to "Developer Preview" section
2. Enter key: `tango`
3. Click "Validate"
4. The download button should appear

### 4c. Download works
Click "Download Developer Preview" — it should download the 241 MB installer
from GitHub Releases.

### 4d. Direct download URL works
```
https://github.com/salilsankritya/bakup.ai/releases/latest/download/bakup-ai-installer.exe
```
This URL always points to the **latest release**, so future uploads
automatically become the default download.

---

## Step 5 — Share with your team

Send this to your developers and testers:

```
Landing page:  https://salilsankritya.github.io/bakup.ai
Access key:    <share privately via Slack/email>

Steps:
1. Open the link above
2. Enter the access key
3. Click "Download Developer Preview"
4. Run the installer
5. Launch bakup.ai from the Start menu
6. Point it at your project and ask questions
```

---

## Step 6 (Optional) — Custom domain

Want `bakup.ai` or `app.bakup.ai` instead of the GitHub URL?

### 6a. Buy the domain
- [Namecheap](https://namecheap.com) — ~$10/year for `.ai` domains
- [Cloudflare Registrar](https://www.cloudflare.com/products/registrar/) — at-cost pricing

### 6b. Configure DNS
Add a CNAME record pointing to GitHub Pages:

| Type  | Name | Value                              |
|-------|------|------------------------------------|
| CNAME | `@`  | `salilsankritya.github.io`         |

Or for a subdomain:

| Type  | Name  | Value                              |
|-------|-------|------------------------------------|
| CNAME | `app` | `salilsankritya.github.io`         |

### 6c. Tell GitHub Pages about the domain
1. Go to **Settings → Pages**
2. Under **Custom domain**, enter your domain (e.g. `bakup.ai`)
3. Click **Save**
4. Check **"Enforce HTTPS"** (free SSL from GitHub)

### 6d. Add CNAME file to repo
```powershell
echo "bakup.ai" > c:\Users\91876\Documents\bakup.ai\CNAME
git add CNAME
git commit -m "add custom domain CNAME"
git push
```

---

## Step 7 (Optional) — Update the installer for new releases

When you build a new version:

```powershell
# 1. Build the new installer (from build/ directory)
cd c:\Users\91876\Documents\bakup.ai\build
.\build.ps1

# 2. Copy to downloads/
Copy-Item dist\bakup-ai-installer.exe ..\downloads\bakup-ai-installer.exe

# 3. Create a new GitHub Release
cd c:\Users\91876\Documents\bakup.ai
gh release create v0.3.0 `
  --title "v0.3.0 — <Release Name>" `
  --notes "Changelog here" `
  downloads/bakup-ai-installer.exe

# The download URL in app.js uses /releases/latest/download/...
# so it automatically points to the newest release. No code change needed.
```

---

## Step 8 — What users do after downloading

This is the flow from the user's perspective:

```
1. Visit landing page → enter access key → download installer
2. Run bakup-ai-installer.exe
3. Installer creates:
   - C:\Users\<user>\AppData\Local\bakup-ai\   (main application)
   - Desktop shortcut                            (bakup.ai)
   - Start menu entry                            (bakup.ai)
4. Double-click the shortcut
5. bakup.ai desktop app launches with a splash screen
6. The backend server starts automatically in the background
7. The app UI loads in the desktop window (no browser needed)
8. User clicks "Index Project" → provides project path (or picks from recent projects)
9. User asks questions → gets structured answers
```

---

## Quick Reference

| What | URL |
|------|-----|
| Landing page | `https://salilsankritya.github.io/bakup.ai` |
| GitHub repo | `https://github.com/salilsankritya/bakup.ai` |
| Latest installer | `https://github.com/salilsankritya/bakup.ai/releases/latest/download/bakup-ai-installer.exe` |
| Pages settings | `https://github.com/salilsankritya/bakup.ai/settings/pages` |
| Create release | `https://github.com/salilsankritya/bakup.ai/releases/new` |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Page shows 404 | Wait 2–3 min after enabling Pages; check branch is `main`, folder is `/ (root)` |
| CSS/JS not loading | Ensure `styles.css`, `fonts.css`, `app.js` are committed and pushed |
| Download fails | Ensure you created a GitHub Release AND uploaded the .exe as an asset |
| Key validation fails on Pages | GitHub Pages uses HTTPS by default — Web Crypto works fine |
| "Validate" button error | Clear browser cache; ensure `app.js` is the latest version |
| Custom domain not working | DNS propagation takes up to 24 hours; verify CNAME record |
