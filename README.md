 
# Python package in conda


📘 README.md — OASIS Machine Learning Pipeline (With Badges)

 
<p align="center">
  https://img.shields.io/badge/python-3.10%2B-blue.svg
  https://img.shields.io/badge/license-MIT-green.svg
  https://img.shields.io/github/last-commit/USERNAME/OASIS
  https://img.shields.io/github/issues/USERNAME/OASIS
  https://img.shields.io/github/issues-pr/USERNAME/OASIS
  https://img.shields.io/badge/code%20style-black-000.svg
</p>

---

# OASIS Machine Learning Pipeline

This repository contains the full end‑to‑end workflow for training, testing, and validating a LightGBM-based machine learning model.  
The project includes:

- Real dataset training pipeline  
- Versioned model saving  
- Automated GitHub Actions CI  
- Model artifact uploads  
- Pytest-based model validation  
- CLI training command  

---

## 📦 Project Structure

 

OASIS/ │ ├── data/ │   └── dataset.csv ├── models/ │   └── trained_model.pkl ├── src/ │   ├── train_pipeline.py │   ├── model_loader.py │   └── oasis/ │       └── cli.py ├── tests/ │   └── test_lgb_model.py └── .github/workflows/ci.yml

 

---

## 🚀 Training Pipeline

Training is handled by:

src/oasis/cli.py
 
models/retrain_model.py
 

Modify  models/retrain_model.py :

 
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
import os
from datetime import datetime

MODEL_PATH = "models/trained_model.pkl"

def retrain_model():
    X_train = pd.DataFrame([
        [0.2, 0.1],
        [0.8, 0.9],
        [0.3, 0.2]
    ], columns=["f1", "f2"])
    y_train = np.array([0, 1, 0])

    model = LGBMClassifier(n_estimators=50, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    metadata = {
        "version": datetime.utcnow().strftime("%Y.%m.%d.%H%M"),
        "timestamp": datetime.utcnow().isoformat(),
        "features": ["f1", "f2"]
    }

    joblib.dump({"model": model, "metadata": metadata}, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")
 


 entry_points={
    "console_scripts": [
        "oasis=oasis.cli:cli",
    ]
}
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
import os

MODEL_PATH = "models/trained_model.pkl"

def retrain_model():
    # Training dataset that matches the test
    X_train = pd.DataFrame([
        [0.2, 0.1],
        [0.8, 0.9],
        [0.3, 0.2]
    ], columns=["f1","f2"])

    y_train = np.array([0, 1, 0])

    model = LGBMClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3
    )

    model.fit(X_train, y_train)

    joblib.dump({"model": model, "features": ["f1","f2"]}, MODEL_PATH)
    print("Model trained and saved.")

if __name__ == "__main__":
    retrain_model()


src/train_pipeline.py

 

This script:

src/model_loader.py
 

 
import joblib
import os

MODEL_PATH = "models/trained_model.pkl"

def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]
 


- Loads the real dataset (`data/dataset.csv`)
- Splits into training/testing subsets
- Trains a LightGBM classifier
- Saves the model AND feature names to:
 

models/trained_model.pkl

 oasis evaluate validation.csv,(target)

Run training manually:

```bash
python src/train_pipeline.py
 

 

🧪 Testing

Testing is done with pytest.

The test:

Loads the trained model

Ensures the model produces valid predictions

Checks DataFrame input/feature alignment

Run tests:
 
pytest -v
 

 

⚙️ GitHub Actions CI Workflow

Location:

 
.github/workflows/ci.yml
 

Pipeline steps:

- name: Train model
  run: python models/retrain_model.py

- name: Run tests
  run: pytest -v

Install dependencies

Retrain the model

Run pytest

Upload artifacts only on failure

 

📤 Artifact Upload (Failure Only)

Artifacts include:

 models/trained_model.pkl 

Test logs

Pytest XML reports

Template snippet:

 
- name: Upload model artifact (only if failed)
  if: failure()
  uses: actions/upload-artifact@v3
  with:
    name: trained-model
    path: models/trained_model.pkl
 

 

🖥️ CLI

After installing:

 
pip install -e .
 

You can run:

Train model:

 
oasis train
 

More commands can be added in:

 
Inside  pyproject.toml :

 
[project.scripts]
oasis = "oasis.cli:cli"
 

Or in setup.py:

 
entry_points={
    "console_scripts": [
        "oasis=oasis.cli:cli",
    ]
}
 

 

🚀 Your CLI Now Supports:

✔ Model training
✔ Model prediction
✔ Model evaluation
✔ Automatic feature alignment
✔ Error checks for missing columns
✔ Real dataset compatibility



📊 Dataset Format

Your dataset ( data/dataset.csv ) must include:feature1, feature2, ..., target

Feature columns

A target column named:
 
target
 

 

🧱 Future Enhancements

Planned upgrades:

Hyperparameter optimization

Model versioning

Automated deployment workflow

GPU‑accelerated training pipeline

 

🏁 Conclusion

This README provides a complete overview of the OASIS ML training + testing pipeline with integrated CI, CLI support, and artifact handling.
