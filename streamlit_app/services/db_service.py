from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

# Import models AFTER Base is defined in db_models
# Ensure db_models is importable from the current path
try:
    # Assuming db_models.py is in the parent directory (streamlit_app/)
    # when services is a package.
    from ..db_models import User, UserProfile, Base, engine # Use relative import
except ImportError:
    # Fallback for direct script execution (less common for structured apps)
    # or if db_models is in the same directory as db_service (not the planned structure)
    print("Warning: Relative import of db_models failed, trying direct import (db_service.py).")
    from db_models import User, UserProfile, Base, engine


# --- Create session factory (Define ONCE) ---
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Ensure tables are created (idempotent call - Define ONCE) ---
def init_db():
    """Initializes the database and creates tables if they don't exist."""
    # Use the print message appropriate for your target (cloud or local)
    # If DATABASE_URL in db_models.py points to cloud, this will act on it.
    print("Initializing database and creating tables (will act on configured DB)...")
    try:
        Base.metadata.create_all(bind=engine)
        print("Database schema initialization complete.")
    except Exception as e:
        print(f"Error during DB schema initialization: {e}")
        raise # Re-raise the error to make it visible

# Call init_db() from your main app script (e.g., Home.py) on startup, not here.
# init_db()


@contextmanager
def get_db_session():
    """Provides a transactional scope around a series of operations."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        print(f"DB Transaction Error: {e}") # Log the error
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
        print(f"User '{username}' created successfully with ID {user_id}.")
        # Return only the ID, which is safe data and doesn't cause DetachedInstanceError
        return user_id

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
            }
            return user_data # Return the dictionary
        else:
            return None # User not found

def get_user_by_id(user_id: int): # Not currently used by other provided code, but can be kept
     """
     Gets user ORM object - use result carefully to avoid detached errors.
     Prefer returning specific fields as a dict if possible.
     """
     with get_db_session() as db:
        return db.query(User).filter(User.id == user_id).first()

# --- Profile Functions ---
def save_or_update_profile(user_id: int, profile_data: dict):
    """Saves or updates a user's profile. Returns the saved/updated profile as a dictionary."""
    returned_profile_dict = None
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
            # Filter to only include keys that are actual columns in UserProfile
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

        db.flush() # Ensure data is flushed to DB so all attributes are populated
        # Convert the ORM object to a dictionary *before* the session closes
        if db_profile: # Check if db_profile was successfully created/found
            returned_profile_dict = {c.name: getattr(db_profile, c.name) for c in db_profile.__table__.columns}

    return returned_profile_dict # Return the dictionary


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