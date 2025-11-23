
# Fraud Detection with PyOD AutoEncoder

This project trains an AutoEncoder-based fraud detection model using the
[PyOD](https://pyod.readthedocs.io/) library and the anonymized credit card
transaction dataset from Kaggle.

## Project Structure

```text
fraud_autoencoder_pyod/
├── data/
│   └── creditcard.csv           # Kaggle dataset (already included)
├── src/
│   └── fraud_autoencoder.py     # main Python script
├── manifest.txt
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Running the Experiment

From the project root (`fraud_autoencoder_pyod/`), run:

```bash
python -m src.fraud_autoencoder
```

or

```bash
python src/fraud_autoencoder.py
```

The script will:

- Load the Kaggle credit card fraud dataset from `data/creditcard.csv`
- Preprocess and scale the features
- Train a PyOD `AutoEncoder` model
- Print confusion matrix, classification report, and ROC-AUC
- Display a ROC curve plot (you can take a screenshot of this for your report)

## GitHub Submission

1. Initialize a Git repository in this folder:

   ```bash
   git init
   git add .
   git commit -m "Initial commit - PyOD AutoEncoder fraud detection"
   ```

2. Create a new repository on GitHub (e.g., `fraud-autoencoder-pyod`),
   then add the remote and push:

   ```bash
   git remote add origin https://github.com/<your-username>/fraud-autoencoder-pyod.git
   git push -u origin main
   ```

3. Include the GitHub URL in your Word document as required by the assignment.
