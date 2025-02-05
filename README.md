# 🤖 No-Code ML Model Training

This project provides a **Streamlit-based web application** that allows users to train machine learning models without writing code. Users can select a dataset, choose a target column, apply scaling, train a model, and download the trained model for later use.

---

## 🚀 Features
- **Upload and Select a Dataset** from the available files.
- **Automatic Data Preprocessing** (Train-Test Split and Scaling options).
- **Supports Multiple ML Models:**
  - Logistic Regression
  - Support Vector Classifier
  - Random Forest Classifier
  - XGBoost Classifier
- **Evaluate Model Accuracy** after training.
- **Download Trained Model** as a `.pkl` file.

---

## 🛠️ Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-repo/no-code-ml.git
cd no-code-ml
```

### 2️⃣ Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App
```bash
streamlit run src/main.py
```

---

## 📂 Project Structure
```
no-code-ml/
│── data/                  # Folder containing datasets
│── models/                # Folder where trained models are saved
│── main.py                # Streamlit app (UI & interactions)
│── ml_utility.py          # Helper functions for ML pipeline
│── requirements.txt       # List of dependencies
│── README.md              # Project documentation
```

---

## ⚙️ How It Works
1. **Select a dataset** from the dropdown menu.
2. **View the dataset preview** in the Streamlit UI.
3. **Choose:**
   - Target column
   - Scaler type (`Standard` or `MinMax`)
   - ML model
   - Model name
4. Click **"Train the Model"** to train and evaluate the model.
5. **Download** the trained model as a `.pkl` file.



## ✉️ Contact
For questions, reach out via **[LinkedIn](https://www.linkedin.com/in/subhash-bishnoi-a068a42b1/)** or email: **me22b2044@iiitdm.ac.in**

