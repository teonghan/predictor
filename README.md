# 🔮 Model Prediction App  
*Your model, your rules — predict away!*

Welcome to the **Model Prediction App**, where `.pkl` files meet Streamlit magic. Got a trained model? Just toss it in here and watch the predictions flow. Single row? Whole file? We got you.

This is a companion app to https://github.com/teonghan/stat_quickie

---

## 🚀 What This App Does

- 📤 Upload your **trained model (.pkl)**  
- 📊 Predict using **a single data point** or **batch files (CSV/Excel)**  
- 🧠 Supports **classification** _and_ **regression**  
- 🧼 Handles preprocessing like a pro (categoricals, missing columns, OHE logic)

---

## 💡 Why Use This?

Because:
- You don’t want to rebuild your pipeline from scratch
- You want a slick interface for stakeholders (or yourself 😎)
- You like pressing buttons more than writing code

---

## 🛠️ How to Use

1. Train and export your model (with metadata!) from your pipeline or notebook
2. Clone this repo:

   ```bash
   git clone https://github.com/teonghan/predictor.git
   cd predictor
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

5. Upload your `.pkl` and go wild.

---

## 📦 Model Format Requirements

Your `.pkl` file should be a dictionary like:

```python
{
  'model': trained_model,
  'feature_names': [...],
  'target_column': 'Appraisal 2023',
  'is_regression': True or False,
  'label_encoder': encoder_or_None,
  'original_predictor_cols': [...],
  'categorical_unique_values': {...},
  'one_hot_encoded_feature_map': {...}
}
```

> 🧙 No worries — you’ll get a friendly error if something’s off.

---

## 📸 Screenshots (Add Yours!)

- Upload interface  
- Prediction output  
- Probability tables for classifiers  
- Download button for batch results

---

## 🤘 Made with

- [Streamlit](https://streamlit.io)
- [scikit-learn](https://scikit-learn.org)

---

> 🎉 Happy modeling!
