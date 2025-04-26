# AI Financial Advisor (India) - Streamlit

This project provides personalized financial advice using AI, tailored for the Indian context. It features user profiling, risk assessment, and explainable recommendations.

## Project Structure

```
ai_financial_advisor_india/
│
├── streamlit_app/
│   ├── Home.py
│   ├── pages/
│   │   ├── 1_🔑_Login_Register.py
│   │   ├── 2_👤_Profile.py
│   │   └── 3_📊_Dashboard_Advice.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auth_service.py
│   │   ├── db_service.py
│   │   └── advice_service.py
│   ├── ai_integration/
│   │   ├── __init__.py
│   │   └── prediction.py
│   ├── db_models.py
│   └── config.py # (Optional)
│   └── utils.py  # (Optional)
│
├── ml_scripts/
│   ├── data_generation/
│   │   └── generate_user_profile.py
│   ├── data_processing/
│   ├── training/
│   │   └── train_risk_model.py
│   └── explainability/
│
├── models/
│   # Populated by ml_scripts
│
├── data/
│   # Populated by ml_scripts
│
├── tests/
│
├── venv/
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd ai_financial_advisor_india
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate
    # MacOS/Linux: source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Initial Data & Model Setup (Person A Task)

1.  **Generate Synthetic Data:**
    ```bash
    # Ensure you are in the project root directory
    python ml_scripts/data_generation/generate_user_profile.py
    ```

2.  **Preprocess Data & Train Risk Model:**
    ```bash
    # Ensure you are in the project root directory
    python ml_scripts/training/train_risk_model.py
    ```

## Running the Streamlit Application (Person B Task / Testing)

1.  **Initialize the Database:**
    *   The first time you run the Streamlit app, the SQLite database (`app_database.db`) should be created automatically by the application in the project root.

2.  **Run the Streamlit App:**
    ```bash
    # Ensure you are in the project root directory
    streamlit run streamlit_app/Home.py
    ```
    Open your browser to the URL provided by Streamlit (usually http://localhost:8501).

## Development Notes

*   **Person A:** Focuses on scripts in `ml_scripts/`, generating artifacts into `data/` and `models/`. Provides functions in `streamlit_app/ai_integration/prediction.py`.
*   **Person B:** Focuses on files within `streamlit_app/`, building the UI, services, database interactions, and calling Person A's functions from `ai_integration`.
*   **Models Directory:** The root `models/` folder is the handoff point for trained models and preprocessors.
*   **Database:** Currently configured for SQLite in the root directory (`app_database.db`). Change `DATABASE_URL` in `streamlit_app/db_models.py` for other databases.
