# SMS Spam Detection

## 📌 Project Overview
SMS Spam Detection is a machine learning project that classifies text messages as **Spam** or **Not Spam (Ham)**. It uses Natural Language Processing (NLP) and machine learning techniques to identify unwanted and potentially fraudulent messages.

## 🎯 Objectives
- Detect spam SMS messages automatically.
- Classify messages using machine learning.
- Process text data for model training.
- Provide a simple interface for testing messages.

## ✨ Features
- SMS text classification.
- Text preprocessing and cleaning.
- TF-IDF feature extraction.
- Machine learning model prediction.
- Spam and ham classification.


## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib



*Note: Keep only the files that exist in your project. Your dataset and saved model filenames may differ.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd SMS_Spam_Detection
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Application

If your Streamlit application is named `app.py`, run:

```bash
streamlit run app.py
```

## 🔍 How It Works
1. Load and prepare the SMS dataset.
2. Clean and preprocess the message text.
3. Convert text into numerical features using TF-IDF.
4. Train a machine learning classifier.
5. Predict whether a message is spam or ham.
6. Display the prediction to the user.

## 📊 Expected Output
The application accepts an SMS message and predicts one of the following:
- **Spam:** The message is classified as unwanted or suspicious.
- **Ham:** The message is classified as a normal message.

Predictions depend on the training dataset and model performance.

## 🚀 Future Improvements
- Test multiple machine learning algorithms.
- Improve model accuracy through evaluation and tuning.
- Add a confusion matrix and classification report.
- Improve the Streamlit interface.
- Test the model on new, unseen messages.

## 👨‍💻 Author
      Happy Kumar
Developed as a machine learning project to explore SMS classification using Python and NLP.
