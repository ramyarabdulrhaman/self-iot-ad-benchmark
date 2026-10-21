# Step-by-Step Instructions to Upload This Repository to GitHub

This guide walks you through uploading the `self-iot-ad-benchmark` folder to your GitHub account (`https://github.com/ramyarteimoor/self-iot-ad-benchmark`).

---

## 📋 Method 1: Using the Command Line (Recommended & Fastest)

### Step 1: Create an Empty Repository on GitHub
1. Open your browser and go to [https://github.com/new](https://github.com/new).
2. Set **Repository name**: `self-iot-ad-benchmark`
3. Description: `A Leakage-Free Benchmark of Self-Supervised and Semi-Supervised Representation Learning for IoT Intrusion Detection Under Label Scarcity`
4. Set visibility to **Public**.
5. **IMPORTANT:** Do **NOT** check "Add a README file", "Add .gitignore", or "Choose a license" (we have already created all of these files inside this folder).
6. Click **Create repository**.

---

### Step 2: Open Terminal / PowerShell in This Folder
1. Open PowerShell or Command Prompt.
2. Navigate directly into this directory:
   ```powershell
   cd E:\Paper\article_11_iot_self_supervised_anomaly\self-iot-ad-benchmark
   ```

---

### Step 3: Initialize Git and Push to GitHub
Copy and paste the following commands one by one (or all at once):

```powershell
# 1. Initialize local git repository
git init

# 2. Add all files in this directory to staging
git add .

# 3. Create the initial commit
git commit -m "Initial release: Leakage-free benchmark of SSL and SSL-consistency for IoT intrusion detection"

# 4. Ensure the default branch is named main
git branch -M main

# 5. Connect your local repository to your GitHub remote repository
git remote add origin https://github.com/ramyarteimoor/self-iot-ad-benchmark.git

# 6. Push your code to GitHub
git push -u origin main
```

*(If prompted to authenticate, log in via your browser or use your GitHub Personal Access Token.)*

---

## 🖥️ Method 2: Using GitHub Desktop (Graphical User Interface)

If you prefer a visual interface without typing commands:

1. Download and open **GitHub Desktop** ([desktop.github.com](https://desktop.github.com/)).
2. Click **File** $\to$ **Add Local Repository...** (or press `Ctrl + O`).
3. Browse to: `E:\Paper\article_11_iot_self_supervised_anomaly\self-iot-ad-benchmark`
4. If it says "This directory does not appear to be a Git repository", click **create a repository here**.
   - Name: `self-iot-ad-benchmark`
   - Git Ignore: None (we already have `.gitignore`)
   - License: None (we already have `LICENSE`)
   - Click **Create Repository**.
5. In the summary box at the bottom-left, enter:
   - Summary: `Initial release of Self-IoT-AD benchmark`
   - Click **Commit to main**.
6. Click **Publish repository** in the top bar:
   - Name: `self-iot-ad-benchmark`
   - Uncheck "Keep this code private" (to make it public for review).
   - Click **Publish Repository**.

---

## ⚡ Method 3: Using GitHub CLI (`gh`)

If you have GitHub CLI installed:

```powershell
cd E:\Paper\article_11_iot_self_supervised_anomaly\self-iot-ad-benchmark
git init
git add .
git commit -m "Initial release: Leakage-free benchmark of SSL and SSL-consistency for IoT intrusion detection"
gh repo create self-iot-ad-benchmark --public --source=. --remote=origin --push
```

---

## ✅ Verification Checklist After Upload

Once pushed, visit your repository page:
👉 **[https://github.com/ramyarteimoor/self-iot-ad-benchmark](https://github.com/ramyarteimoor/self-iot-ad-benchmark)**

Check the following:
1. [x] **README.md** renders properly with author details, affiliations, ORCID badge, abstract, and results table.
2. [x] `code/` contains all 6 reproduction scripts (`prepare_data.py`, `model_and_algorithms.py`, etc.).
3. [x] `data/` contains `raw/rt_iot2022_curated.csv` and `processed/rt_iot2022_splits.npz`.
4. [x] `results/` contains `experiment_results_20reps.json` and models (`self_iot_ad_fp32.pt`, `self_iot_ad_int8.pt`).
5. [x] `tables/` and `figures/` display all paper artifacts cleanly.

---

## 🏛️ Permanent Archiving via Zenodo (Optional for Journal Review)

To obtain a permanent DOI (Digital Object Identifier) for your code repository to include in journal camera-ready papers:
1. Go to [https://zenodo.org](https://zenodo.org) and sign in with your GitHub account.
2. Navigate to your Zenodo GitHub settings: [https://zenodo.org/account/settings/github/](https://zenodo.org/account/settings/github/).
3. Flip the switch next to `ramyarteimoor/self-iot-ad-benchmark` to **ON**.
4. On GitHub, create a new Release (e.g. `v1.0.0`).
5. Zenodo will automatically archive the repository and mint a citable DOI (e.g., `10.5281/zenodo.XXXXXXX`).
