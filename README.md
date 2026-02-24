 
# python package in conda


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

 

src/train_pipeline.py

 

This script:

- Loads the real dataset (`data/dataset.csv`)
- Splits into training/testing subsets
- Trains a LightGBM classifier
- Saves the model AND feature names to:
 

models/trained_model.pkl

 

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

 
src/oasis/cli.py
 

 

📊 Dataset Format

Your dataset ( data/dataset.csv ) must include:

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
