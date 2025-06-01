# NLP: Resume Classification System

## Overview

**Resume-Analysis-NLP** is a full-stack application that uses Natural Language Processing (NLP) and Deep Learning to classify resumes into job categories. It provides a modern web interface for users to upload resumes (PDF, DOCX, DOC) or paste text, and instantly receive a predicted job category and confidence score.

- **Backend:** FastAPI, TensorFlow/Keras, NLTK, joblib
- **Frontend:** HTML, CSS, JavaScript (vanilla)
- **Deployment:** Docker-ready

---

## Features

- **Classifies resumes** into 25+ job categories using a trained deep learning model.
- **Accepts input** as either:
  - Uploaded file (`.pdf`, `.docx`, `.doc`)
  - Pasted text
- **Displays results**: Category and confidence score.
- **Preview**: Shows extracted text from uploaded files.
- **Retraining endpoint** (for advanced users).
- **Modern, responsive frontend**.

---

## Directory Structure

```
Resume-Analysis-NLP/
│
├── main.py                  # FastAPI backend (API, model loading, endpoints)
├── requirements.txt         # Python dependencies
├── Dockerfile               # For containerized deployment
│
├── src/                     # Core backend modules
│   ├── model/               # Model architecture, saved model, tokenizer, encoder
│   ├── utils/               # Utilities (logging, helpers, file extraction)
│   ├── training/            # Training pipeline and scripts
│   ├── preprocessing/       # Data preprocessing logic
│   └── inference/           # Inference and prediction logic
│
├── dataset/                 # Resume datasets (CSV, sample PDFs)
│
├── frontend/                # Frontend (static files)
│   ├── index.html           # Main UI
│   ├── styles.css           # Styling
│   └── script.js            # Interactivity/API calls
│
├── logs/                    # Log files
├── tests/                   # Test scripts
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd Resume-Analysis-NLP
```

### 2. Install dependencies

**Recommended:** Use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download/prepare model files

Ensure the following files exist in `src/model/`:
- `best_model.keras`
- `trained_tokenizer.json`
- `OHEncoder.joblib`

If not, retrain the model using the provided notebook or scripts.

### 4. Run the app

```bash
uvicorn main:app --reload
```

Visit [http://localhost:8000](http://localhost:8000) in your browser.

---

## Docker Deployment

Build and run the app in a container:

```bash
docker build -t resume-analyser .
docker run -p 8000:8000 resume-analyser
```

---

## API Endpoints

### `POST /classify_resume/text/`
- **Input:** JSON: `{ "resume_text": "..." }`
- **Output:** Category, confidence, extracted text

### `POST /classify_resume/file/`
- **Input:** Form-data: `file` (PDF, DOCX, DOC)
- **Output:** Category, confidence, extracted text

### `POST /classify_resume/train/`
- **Input:** (Advanced) Triggers retraining (see code for details)

### `GET /`
- **Frontend UI** (index.html)

### `GET /logs/status`
- **Returns logging status**

---

## Frontend Usage

- **Paste resume text** or **upload a file** in the left column.
- Click **Classify Resume**.
- See the **predicted category** and **confidence** below.
- The right column shows the uploaded file name and extracted text.

---

## Dataset

- Place your resume datasets (CSV, PDF) in the `dataset/` directory.
- Example files: `resume_new.csv`, `resume_dataset.csv`, `DummyResume.pdf`

---

## Model & Training

- Model is a custom Keras text classifier.
- Preprocessing uses NLTK, custom tokenization, and one-hot encoding.
- Training scripts and logic are in `src/training/` and `src/preprocessing/`.
- See `NLP_Resume_Classification.ipynb` for EDA and prototyping.

---

## Customization

- **Add new categories:** Update your dataset and retrain the model.
- **Change model architecture:** Edit `src/model/model.py`.
- **Logging:** Configured in `src/utils/logger.py`.

---

## Contributing

1. Fork the repo
2. Create a feature branch
3. Commit your changes
4. Open a pull request

---

## License

MIT License.

---

## Acknowledgements

- Inspired by open datasets and NLP research.
- Built with FastAPI, TensorFlow, and NLTK.

---

**For more details, see the code and comments in each module.**
