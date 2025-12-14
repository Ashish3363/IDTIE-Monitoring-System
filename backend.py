from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from recognition import add_student_images, load_index, reset_faiss_and_db, save_index, index_to_metadata
import os   
import redis
from fastapi import FastAPI, Form, HTTPException
from passlib.context import CryptContext
from pymongo import MongoClient
from pydantic import BaseModel
from typing import List
from datetime import datetime
import bcrypt
from typing import List
from datetime import datetime
import dateutil.parser
from fastapi import HTTPException
from dateutil.parser import parse as parse_date


from fastapi.staticfiles import StaticFiles
import os



MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client['schoolDB']  # replace with your DB name
users_collection = db['users']      # collection name

violations_collection = db['violations']


app = FastAPI()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")




app = FastAPI()


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add after app initialization
VIOLATION_SCREENSHOTS_DIR = "violation_screenshots"
os.makedirs(VIOLATION_SCREENSHOTS_DIR, exist_ok=True)

# Mount static files to serve images
app.mount("/violation_images", StaticFiles(directory=VIOLATION_SCREENSHOTS_DIR), name="violation_images")


UPLOAD_FOLDER = "faces_db"

# Redis setup for event notifications
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# Load FAISS + metadata cache on startup
try:
    load_index()
    print("[INFO] FAISS index + metadata loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load FAISS index: {e}")


@app.post("/upload-student/")
async def upload_student(
    student_id: str = Form(...),
    student_name: str = Form(None),
    files: list[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    student_folder = os.path.join(UPLOAD_FOLDER, student_id)
    os.makedirs(student_folder, exist_ok=True)
    image_paths = []

    try:
        # Save uploaded files
        for file in files:
            file_path = os.path.join(student_folder, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
            image_paths.append(file_path)

        # Update FAISS + MongoDB + in-memory cache
        add_student_images(
            student_id,
            image_paths,
            student_name=student_name,
            photo_folder=student_folder
        )

        # Persist FAISS index and metadata snapshot
        save_index()

        redis_client.publish("reload_index_channel", "reload")

        return {
            "status": "success",
            "student_id": student_id,
            "student_name": student_name,
            "num_images": len(files),
            "cache_size": len(index_to_metadata)
        }

    except Exception as e:
        # Cleanup partially saved files on failure
        for path in image_paths:
            if os.path.exists(path):
                os.remove(path)

        raise HTTPException(status_code=500, detail=f"Failed to add student: {str(e)}")


@app.post("/register-teacher/")
async def register_teacher(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
):
    
    print("Registering teacher with email:", email)

    # Check if email already exists
    existing = users_collection.find_one({"email": email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Truncate and hash password safely
    password_bytes = password[:MAX_BCRYPT_LEN].encode("utf-8")
    hashed_password = bcrypt.hashpw(password_bytes, bcrypt.gensalt())

    teacher_doc = {
        "name": name,
        "email": email,
        "hashed_password": hashed_password.decode("utf-8"),  # store as string
        "role": "teacher"
    }

    users_collection.insert_one(teacher_doc)
    return {"status": "success", "email": email}



@app.post("/reset-password/")
async def reset_teacher_password(
    email: str = Form(...),
    new_password: str = Form(...),
):
    # Check if teacher exists
    teacher = users_collection.find_one({"email": email, "role": "teacher"})
    if not teacher:
        raise HTTPException(status_code=404, detail="Teacher not found")

    # Truncate and hash the new password safely
    password_bytes = new_password[:MAX_BCRYPT_LEN].encode("utf-8")
    hashed_password = bcrypt.hashpw(password_bytes, bcrypt.gensalt())

    # Update the password in MongoDB
    users_collection.update_one(
        {"email": email, "role": "teacher"},
        {"$set": {"hashed_password": hashed_password.decode("utf-8")}}
    )

    return {"status": "success", "email": email}

# Response model
class Violation(BaseModel):
    studentId: str
    cause: List[str]
    timestamp: str
    image: str

@app.get("/violations/", response_model=List[Violation])
async def get_violations():
    # Fetch all documents from MongoDB
    raw_violations = list(violations_collection.find({}, {"_id": 0}))
    
    if not raw_violations:
        raise HTTPException(status_code=404, detail="No violations found")

    # Convert all fields to strings (even timestamp)
    safe_violations = []
    for v in raw_violations:
        safe_violations.append({
            "studentId": str(v.get("studentId", "")),
            "cause": v.get("cause", []), 
            "timestamp": str(v.get("timestamp", "")),  # keep original string
            "image": str(v.get("image", ""))
        })

    print("Fetched violations:", safe_violations)

    return safe_violations



@app.delete("/violations/{studentId}")
async def delete_violation(studentId: str):
    result = violations_collection.delete_one({"studentId": studentId})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Violation not found")
    return {"status": "success"}


class LoginRequest(BaseModel):
    email: str
    password: str


MAX_BCRYPT_LEN = 72  # Bcrypt password byte limit

@app.post("/login")
def login(data: LoginRequest):
    try:
        # Lookup user by email
        user = users_collection.find_one({"email": data.email})
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Verify password safely using bcrypt directly
        password_bytes = data.password[:MAX_BCRYPT_LEN].encode('utf-8')
        stored_hash = user["hashed_password"].encode('utf-8')

        if not bcrypt.checkpw(password_bytes, stored_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Successful login
        return {
            "username": user.get("name"),
            "email": user.get("email"),
            "role": user.get("role")
        }

    except Exception as e:
        print("Login error:", e)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/reset/")
async def reset_system():
    result = reset_faiss_and_db()
    return result



@app.get("/ping")
def ping():
    return {"message": "pong"}