# app/auth/routes.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from starlette.responses import JSONResponse

from app.database.database import supabase
import bcrypt
import jwt
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer

# JWT config
SECRET_KEY = "your-secret-key"  # Use env variable in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

router = APIRouter(
    prefix="/auth",
    tags=["auth"]
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


# ---------------------
# Schemas
# ---------------------
class SignUpRequest(BaseModel):
    username: EmailStr
    password: str


class LoginRequest(BaseModel):
    username: EmailStr
    password: str


# ---------------------
# JWT helpers
# ---------------------
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return token


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        # Optionally, fetch user from DB to verify exists
        response = supabase.table("users").select("*").eq("user_id", user_id).execute()
        if not response.data:
            raise HTTPException(status_code=401, detail="User not found")
        return response.data[0]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ---------------------
# Register
# ---------------------
@router.post("/register")
def register_user(data: SignUpRequest):
    print(data)
    existing = supabase.table("users").select("*").eq("username", data.username).execute()
    if existing.data and len(existing.data) > 0:
        return JSONResponse(status_code=400, content={"message": "Username already exists"})

    hashed_password = bcrypt.hashpw(data.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    response = supabase.table("users").insert({
        "username": data.username,
        "password": hashed_password
    }).execute()

    # Check if there was an error
    if response.error:
        raise HTTPException(status_code=500, detail=f"Failed to create user: {response.error.message}")

    return {"message": "User created", "user": {"user_id": response.data[0]["user_id"], "username": data.username}}


# ---------------------
# Login
# ---------------------
@router.post("/login")
def login_user(data: LoginRequest):
    response = supabase.table("users").select("*").eq("username", data.username).execute()
    if not response.data or len(response.data) == 0:
        raise HTTPException(status_code=400, detail="Invalid username or password")

    user = response.data[0]
    if not bcrypt.checkpw(data.password.encode("utf-8"), user["password"].encode("utf-8")):
        raise HTTPException(status_code=400, detail="Invalid username or password")

    token = create_access_token({"user_id": user["user_id"], "username": user["username"]})
    print(token)
    return {"access_token": token, "token_type": "bearer"}
