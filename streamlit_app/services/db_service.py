from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

# Import models AFTER Base is defined in db_models
# Ensure db_models is importable from the current path
try:
    # Assuming db_models.py is in the parent directory (streamlit_app/)
    from db_models import User, UserProfile, DATABASE_URL, Base, engine, metadata
except ImportError:
    print("Error importing db_models from db_service. Check structure/PYTHONPATH.")
    # Handle this error appropriately, maybe exit or raise
    raise

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Ensure tables are created (idempotent call)
def init_db():
    print("Initializing database and creating tables...")
    try:
        Base.metadata.create_all(bind=engine)
        print("Database initialization complete.")
    except Exception as e:
        print(f"Error during DB initialization: {e}")
        # Decide how critical this is - maybe raise the exception
        raise

# init_db() # Call this once when the app starts, e.g. in Home.py

@contextmanager
def get_db_session():
    """Provides a transactional scope around a series of operations."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        print(f"DB Error: {e}") # Log the error
        db.rollback()
        raise # Re-raise the exception so calling code knows about it
    finally:
        db.close()

# --- User Functions ---

def create_user(username: str, hashed_password: str):
    """Creates a new user and returns their ID."""
    with get_db_session() as db:
        db_user = User(username=username, hashed_password=hashed_password)
        db.add(db_user)
        db.flush()  # Make the DB assign the ID now
        user_id = db_user.id # Get the ID while session is active
        print(f"User '{username}' created with ID {user_id}.")
        # Return only the ID, which is safe data
        return user_id

# CORRECTED FUNCTION: Returns needed data as a dictionary, not the ORM object
def get_user_auth_data_by_username(username: str):
    """Fetches essential user data for authentication as a dictionary."""
    with get_db_session() as db:
        user = db.query(User).filter(User.username == username).first()
        if user:
            # Access needed attributes while the session is active
            user_data = {
                "id": user.id,
                "username": user.username,
                "hashed_password": user.hashed_password
                # Add other fields needed immediately if necessary
            }
            return user_data # Return the dictionary
        else:
            return None # User not found

def get_user_by_id(user_id: int):
     """Gets user ORM object - use result carefully to avoid detached errors."""
     # Warning: Returning the full object can lead to DetachedInstanceError
     # if accessed after the session closes. Consider returning a dict if needed elsewhere.
     with get_db_session() as db:
        return db.query(User).filter(User.id == user_id).first()

# --- Profile Functions ---
def save_or_update_profile(user_id: int, profile_data: dict):
    """Saves or updates a user's profile. Returns the profile dictionary."""
    saved_profile_obj = None # To store the object before session closes
    with get_db_session() as db:
        db_profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if db_profile:
            # Update existing profile
            for key, value in profile_data.items():
                if hasattr(db_profile, key):
                    setattr(db_profile, key, value)
                else:
                    print(f"Warning: Attribute '{key}' not found in UserProfile model during update.")
            print(f"Updating profile for user_id: {user_id}")
        else:
            # Create new profile
            valid_keys = [c.name for c in UserProfile.__table__.columns if c.name not in ['id', 'user_id']]
            filtered_data = {k: v for k, v in profile_data.items() if k in valid_keys}
            db_profile = UserProfile(user_id=user_id, **filtered_data)
            db.add(db_profile)
            print(f"Creating new profile for user_id: {user_id}")

        # Mark profile as complete on the User table
        db_user = db.query(User).filter(User.id == user_id).first()
        if db_user:
            db_user.profile_complete = True
        else:
             print(f"Warning: User with id {user_id} not found when trying to mark profile complete.")

        db.flush() # Ensure data is flushed to DB
        # Capture the state into a dictionary *before* session closes
        saved_profile_obj = db_profile # Keep reference to object
        profile_dict = {c.name: getattr(saved_profile_obj, c.name) for c in saved_profile_obj.__table__.columns} if saved_profile_obj else None

    # Return the dictionary created *before* the session closed
    return profile_dict


def get_profile(user_id: int):
    """Gets a user's profile as a dictionary."""
    with get_db_session() as db:
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if profile:
            # Convert ORM object to a dictionary for safe return
            profile_dict = {c.name: getattr(profile, c.name) for c in profile.__table__.columns}
            return profile_dict
        return None

def is_profile_complete(user_id: int) -> bool:
    """Checks if the user's profile is marked as complete."""
    profile_complete_status = False # Default
    with get_db_session() as db:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            profile_complete_status = user.profile_complete # Access attribute within session
    return profile_complete_status # Return the boolean value