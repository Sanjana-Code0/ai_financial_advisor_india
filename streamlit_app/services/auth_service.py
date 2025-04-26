from passlib.context import CryptContext
# Use RELATIVE import for modules within the same package
try:
    # '.' means import from the current package (services)
    from . import db_service
except ImportError:
    print("Error performing relative import of db_service within auth_service.")
    # You might want to raise the error to make it clear something is wrong
    raise

# Setup password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Function Definitions ---
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a stored hash."""
    if not plain_password or not hashed_password:
        return False
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hashes a plain password."""
    return pwd_context.hash(password)

def authenticate_user(username: str, password: str):
    """
    Authenticates a user by username and password.
    Returns a dictionary {'id': user_id, 'username': username} on success, None otherwise.
    """
    # Call the function that returns a dictionary safely
    # db_service is correctly imported via relative import now
    user_data = db_service.get_user_auth_data_by_username(username)

    if not user_data:
        print(f"Auth failed: User '{username}' not found.")
        return None # User not found

    # Access hashed_password from the dictionary
    if not verify_password(password, user_data["hashed_password"]):
        print(f"Auth failed: Incorrect password for user '{username}'.")
        return None # Incorrect password

    print(f"Auth successful for user '{username}'.")
    # Return only the necessary info for session state (also a dictionary)
    return {"id": user_data["id"], "username": user_data["username"]}