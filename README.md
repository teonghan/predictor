# 🔮 Model Prediction App

A flexible Streamlit tool to **load a trained model (.pkl)** and generate predictions via single entry or batch upload — all within a friendly web interface.

Try it online 👉 *https://predictor-pickle.streamlit.app/*

---

## 🚀 Overview

This app helps you perform fast predictions using your trained machine learning model (classification or regression) by:

- 📤 Uploading your `.pkl` model (with metadata)
- 🔢 Entering a single data row (with input UI)
- 📄 Uploading a full dataset (CSV or Excel)
- 🧼 Auto-preprocessing your inputs (e.g., one-hot encoding, missing features)

---

## ✅ Key Features

- Supports both **regression** and **classification**
- Visual mapping of class labels
- Shows prediction probabilities (if available)
- Downloadable batch predictions (CSV)
- Works with **categorical**, **numerical**, and **mixed** features

---

## 🛠 Installation

### Option 1: One-Click macOS Installer

```bash
bash installer-macos-universal.sh
```

What it does:
- Detects Apple Silicon or Intel
- Installs Miniforge if not found
- Creates conda env (`modelprediction`)
- Adds Desktop shortcut with Automator icon

---

### Option 2: One-Click Windows Installer

```powershell
Right-click → Run with PowerShell → installer-windows.ps1
```

What it does:
- Detects Anaconda/Miniconda
- Creates or updates `modelprediction` env from `__environment__.yml`
- Creates launcher (`start-streamlit-app.ps1`)
- Adds Desktop shortcut (`Start Model Prediction App`)
- Generates uninstaller (`uninstall-streamlit-app.ps1`)

> 💡 **Note**: Ensure Conda is installed before running.

---

### Option 3: Manual Setup

```bash
git clone https://github.com/teonghan/predictor.git
cd predictor
conda env create -f __environment__.yml
conda activate modelprediction
streamlit run app.py
```

---

## 📦 Model Format Requirements

The uploaded `.pkl` must be a dictionary like:

```python
{
  'model': trained_model,
  'feature_names': [...],
  'target_column': 'Your Target',
  'is_regression': True or False,
  'label_encoder': encoder_or_None,
  'original_predictor_cols': [...],
  'categorical_unique_values': {...},
  'one_hot_encoded_feature_map': {...}
}
```

If something’s missing or malformed, the app will show a helpful error.

---

## 🧪 Input Data

You can either:
1. **Manually input** values for prediction (single-row)
2. **Upload batch files** (`.csv`, `.xlsx`) and download prediction results

Uploaded files must have all expected feature columns. Missing values will be auto-filled.

---

## 📦 Dependencies

Included in `__environment__.yml` or `requirements.txt`:

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `lightgbm`

---

## 📃 License

MIT License — free for personal, academic, or commercial use.

---

> 🧠 Predict smarter, faster — without writing another line of code.
