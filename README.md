# VITAL: Virtual Interface for Assessment and Treatment in Lifecare

VITAL is a Django-based web application designed to provide a virtual interface for healthcare assessment and treatment. It integrates Electronic Health Records (EHR), machine learning risk prediction, and FHIR interoperability in a modular, extensible architecture.

---

## Table of Contents

-   [Features](#features)
-   [Project Structure](#project-structure)
-   [Setup Instructions](#setup-instructions)
-   [Running the Application](#running-the-application)
-   [Machine Learning Integration](#machine-learning-integration)
-   [Django App Modules](#django-app-modules)
-   [Static and Template Files](#static-and-template-files)
-   [Environment Variables](#environment-variables)
-   [Requirements](#requirements)
-   [License](#license)

---

## Features

-   User authentication for doctors and patients
-   Patient record management and observation entry
-   Machine learning risk prediction for chronic diseases (Diabetes, Liver, Kidney, CVD)
-   Admin dashboard and role-based access
-   FHIR-compliant project structure for interoperability
-   Modular, extensible codebase

---

## Project Structure

```
Application/
├── db.sqlite3                # SQLite database (default)
├── manage.py                 # Django management script
├── requirements.txt          # Python dependencies
├── Procfile                  # For deployment (e.g., Heroku)
├── .env                      # Environment variables (not committed)
├── ehr/                      # Main Django app (EHR, ML, views, models)
│   ├── ml_nhanes_module/     # ML models, training, prediction, artifacts
│   ├── migrations/           # Django migrations
│   ├── templates/            # HTML templates (see below)
│   └── ...                   # Models, views, forms, etc.
├── fhir_project/             # Django project settings (WSGI, ASGI, URLs)
├── static/                   # Static files (images, CSS, JS)
├── staticfiles/              # Collected static files (for deployment)
└── templates/                # Main HTML templates
```

---

## Setup Instructions

1. **Clone the repository**
    ```bash
    git clone <repo-url>
    cd Application
    ```
2. **Create and activate a virtual environment**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
4. **Apply migrations**
    ```bash
    python manage.py migrate
    ```
5. **Create a superuser (admin)**
    ```bash
    python manage.py createsuperuser
    ```
6. **(Optional) Load initial data**
    - Place your data CSVs in the appropriate folders (see ML section).

---

## Running the Application

-   **Development server:**
    ```bash
    python manage.py runserver
    ```
-   **Admin panel:**
    Visit `http://localhost:8000/admin/` and log in with your superuser credentials.

---

## Machine Learning Integration

-   ML code and models are in `ehr/ml_nhanes_module/`.
-   Model artifacts are saved in `ehr/ml_nhanes_module/model_files/`.
-   Prediction endpoint: `/ml/predict/` (see `ml_views.py` for API details).
-   Supported diseases: Diabetes, Liver Condition, Weak/Failing Kidney, CVD (multi-label)

---

## Django App Modules

-   **ehr/**: Main app for EHR, ML, and business logic
    -   `models.py`: Patient, Practitioner, Observation, etc.
    -   `views.py`: Web views for dashboard, login, patient detail, etc.
    -   `ml_nhanes_module/`: ML training, prediction, and model files
    -   `serializers.py`: For API endpoints (if any)
    -   `forms.py`: Django forms for data entry
    -   `urls.py`: URL routing for the app
-   **fhir_project/**: Django project settings, root URLs, WSGI/ASGI

---

## Static and Template Files

-   **static/**: Custom static assets (images, CSS, JS)
-   **staticfiles/**: Collected static files for deployment
-   **templates/**: HTML templates for base, home, doctor, and patient views
    -   `base.html`, `home.html`, etc.
    -   `doctor/`, `patient/` subfolders for role-specific pages

---

## Environment Variables

-   `.env` file (not committed) can be used to store secrets and configuration, e.g.:
    ```env
    SECRET_KEY=your-secret-key
    DEBUG=True
    DATABASE_URL=sqlite:///db.sqlite3
    ```

---

## Requirements

-   Python 3.8+
-   Django 3.2+
-   pandas, scikit-learn, xgboost, seaborn (for ML)
-   gunicorn, dj-database-url, python-dotenv (for deployment)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## License

This project is for academic and research purposes. See `LICENSE` if present.

---

## Contact

For questions or contributions, please contact the project maintainer.
