# streamlit_app/db_models.py
from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey, MetaData
from sqlalchemy.orm import declarative_base
import os

DATABASE_URL = os.environ.get("DATABASE_URL_STREAMLIT", "sqlite:///../app_database.db")
if DATABASE_URL and "@" in DATABASE_URL and ":" in DATABASE_URL.split("@")[0]:
    user_pass, rest_of_url = DATABASE_URL.split("://")[1].split("@", 1)
    user, _ = user_pass.split(":", 1)
    safe_to_print_url = f"{DATABASE_URL.split('://')[0]}://{user}:********@{rest_of_url}"
else:
    safe_to_print_url = DATABASE_URL
print(f"Database URL being used: {safe_to_print_url}")

engine_args = {}
if DATABASE_URL.startswith("postgresql"): pass
elif DATABASE_URL.startswith("sqlite"): engine_args = {"connect_args": {"check_same_thread": False}}
engine = create_engine(DATABASE_URL, **engine_args)
Base = declarative_base()
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
    # --- NEW COLUMNS ---
    InvestmentKnowledge = Column(String, nullable=True) # e.g., Beginner, Intermediate, Advanced
    LiquidityNeeds = Column(String, nullable=True)      # e.g., Low, Medium, High
    # --- END NEW COLUMNS ---

def create_db_tables_internal():
    print("Checking and creating database tables if necessary (from db_models)...")
    try: Base.metadata.create_all(bind=engine); print("DB tables checked/created.")
    except Exception as e: print(f"Error creating tables: {e}"); raise