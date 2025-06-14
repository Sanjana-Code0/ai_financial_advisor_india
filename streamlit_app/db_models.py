# streamlit_app/db_models.py
from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey, MetaData
from sqlalchemy.orm import declarative_base # Corrected import
import os # Import os to read environment variable

# --- Use an environment variable for the database URL ---
# For local testing, you can set this environment variable.
# For Streamlit Cloud, you'll set it as a Secret.
DATABASE_URL = os.environ.get("DATABASE_URL_STREAMLIT", "sqlite:///../app_database.db") # Fallback to SQLite for local
# Example for Supabase: "postgresql://postgres:YOUR_PASSWORD@YOUR_SUPABASE_HOST:5432/postgres"

print(f"Database URL being used: {DATABASE_URL[:DATABASE_URL.find('@') if '@' in DATABASE_URL else len(DATABASE_URL)]}...") # Print safely

engine_args = {}
if DATABASE_URL.startswith("postgresql"):
    # No need for check_same_thread with PostgreSQL
    pass
elif DATABASE_URL.startswith("sqlite"):
    engine_args = {"connect_args": {"check_same_thread": False}}

engine = create_engine(DATABASE_URL, **engine_args)
Base = declarative_base() # Corrected from declarative_base
metadata = MetaData()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    profile_complete = Column(Boolean, default=False)

class UserProfile(Base):
    __tablename__ = "user_profiles"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    AgeRange = Column(String, nullable=True)
    IncomeRange = Column(String, nullable=True)
    SavingsLevel = Column(String, nullable=True)
    DebtLevel = Column(String, nullable=True)
    HasDependents = Column(String, nullable=True)
    PrimaryGoal = Column(String, nullable=True)
    TimeHorizonYears = Column(Integer, nullable=True)
    SelfReportedTolerance = Column(String, nullable=True)

def create_db_tables_internal():
    print("Checking and creating database tables if necessary (from db_models)...")
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables checked/created (from db_models).")
    except Exception as e:
        print(f"Error creating tables: {e}")
        raise