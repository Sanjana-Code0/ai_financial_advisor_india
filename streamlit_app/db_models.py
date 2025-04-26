from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey, MetaData
from sqlalchemy.orm import declarative_base
import os

# Store DB in the project root directory, not inside streamlit_app
# Calculate path relative to this file's location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATABASE_URL = f"sqlite:///{os.path.join(PROJECT_ROOT, 'app_database.db')}"
# Example for PostgreSQL (requires psycopg2-binary):
# DATABASE_URL = "postgresql://user:password@host:port/dbname"

print(f"Database URL set to: {DATABASE_URL}") # For debugging path issues

# check_same_thread=False is needed only for SQLite
engine_args = {"connect_args": {"check_same_thread": False}} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, **engine_args)

Base = declarative_base()
metadata = MetaData() # Useful for checking table existence

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    profile_complete = Column(Boolean, default=False)

    # Define relationship using string name to avoid import issues if UserProfile is defined later
    # profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")

class UserProfile(Base):
    __tablename__ = "user_profiles"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False) # Link to User

    # Store answers directly, matching generation script categories / UI questions
    # Use nullable=True for fields that might not be set initially
    AgeRange = Column(String, nullable=True)
    IncomeRange = Column(String, nullable=True)
    SavingsLevel = Column(String, nullable=True)
    DebtLevel = Column(String, nullable=True)
    HasDependents = Column(String, nullable=True) # Store as 'Yes'/'No' string
    PrimaryGoal = Column(String, nullable=True)
    TimeHorizonYears = Column(Integer, nullable=True) # Store the calculated year number
    SelfReportedTolerance = Column(String, nullable=True)
    # Add any other fields you collect

    # Define relationship back to User
    # user = relationship("User", back_populates="profile")

# Now that both classes are defined, set up relationships properly if needed
# User.profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
# UserProfile.user = relationship("User", back_populates="profile")
# Note: Relationships might not be strictly needed if you only query separately via user_id


# Function to create tables - CAN be called from db_service.init_db()
def create_db_tables_internal():
    print("Checking and creating database tables if necessary (from db_models)...")
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables checked/created (from db_models).")
    except Exception as e:
        print(f"Error creating tables: {e}")
        raise

