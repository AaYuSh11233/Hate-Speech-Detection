# 📝 Hate Speech Detection

A robust, lightweight **Hate Speech Detection** system built with Python and Scikit-Learn.  
This project leverages NLP and machine learning techniques to identify hate speech in textual data, trained on a dataset of labeled tweets.

---

## 📖 Table of Contents

- [About](#about)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Train the Model](#train-the-model)
  - [Run Predictions](#run-predictions)
- [Examples](#examples)
- [Evaluation](#evaluation)

---

## 📌 About

**Hate speech** refers to offensive or discriminatory speech directed toward individuals or groups based on race, religion, ethnicity, gender, sexual orientation, or other protected characteristics.  
This project aims to detect such language in text automatically and accurately, using classic machine learning pipelines and a well-curated dataset.

The goal of this project is to demonstrate how simple and transparent models (like Logistic Regression with TF-IDF) can be effectively used for detecting hate speech in small and balanced datasets.

---

## 🌟 Features

✅ Classifies text as:
- Hate Speech
- Not Hate Speech

✅ Includes:
- Data cleaning and text preprocessing pipeline
- Handling class imbalance via oversampling
- Training-validation split evaluation
- Reproducible ML pipeline saved as a `.joblib` model
- CLI tools for training and inference

✅ Extensible to larger datasets and deep learning models.

---

## 📊 Dataset

- **Training data:** 300 labeled text samples (balanced, synthetic)
- **Test data:** 50 labeled text samples (unseen during training)
- Format: `CSV` with columns:
  - `id`: Unique identifier
  - `tweet`: The text content
  - `label`: `1` for hate speech, `0` for not hate speech

Datasets are located in the `data/` folder:
data/
├── train.csv
├── test.csv

---

## 🖥️ Technologies

- 🐍 Python 3.10+
- 📦 scikit-learn
- 📦 pandas
- 📦 joblib

---

## 📂 Project Structure

Hate_Speech-Detect/
├── data/
│ ├── train.csv # Training dataset
│ └── test.csv # Testing dataset
├── train.py # Training script
├── predict.py # Inference script
├── model.joblib # Trained model (generated after training)
├── README.md # Project documentation

---

## ⚙️ Installation

1️⃣ Clone this repository:
```bash
git clone https://github.com/AaYuSh11233/Hate-Speech-Detection.git
cd Hate-Speech-Detection

````

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
Or individually:
pip install pandas scikit-learn joblib
```

🚀 Usage
🧑‍💻 Train the Model
Train the hate speech detection model using the provided training and test datasets:

```bash
python train.py --train data/train.csv --test data/test.csv --model model.joblib
Output includes:

Validation F1 score

Saved model at model.joblib
```

🔍 Run Predictions
Once trained, you can test the model on new, unseen text:

```bash
python predict.py --model model.joblib --text "I hope you burn in hell"
Sample output:

Prediction: Hate Speech
```

🔬 Examples

Text	Prediction
You are amazing ---- Not Hate Speech
I hate your kind ---- Hate Speech
Wishing you a joyful and successful life ---- Not Hate Speech
You are nothing but a worthless parasite ---- Hate Speech

📈 Evaluation
The model achieves the following on the test set:

F1 Score: 0.89

Accuracy: ~90%

Note: Performance is limited by dataset size. Larger and more diverse datasets can improve generalization.

🌱 Future Work
✅ Increase dataset size with real-world examples
✅ Integrate deep learning models (e.g., BERT)
✅ Build a web or mobile frontend for easier use
✅ Support multi-lingual hate speech detection
✅ Fine-tune hyperparameters for even better accuracy

🤝 Contributing
Contributions are welcome!
To contribute:

Fork the repo

Create your feature branch (git checkout -b feature/new-feature)

Commit your changes (git commit -m 'Add new feature')

Push to the branch (git push origin feature/new-feature)

Open a Pull Request

Please follow the Contributor Covenant guidelines.

Inspiration from public hate speech detection datasets

Everyone contributing to ethical AI research
