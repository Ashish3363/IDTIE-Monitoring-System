from datetime import datetime
from typing import List
import faiss
from fastapi import HTTPException
import numpy as np
from deepface import DeepFace
import os
from pydantic import BaseModel
from pymongo import MongoClient
import numpy as np
from deepface import DeepFace
import shutil

class Violation(BaseModel):
    studentId: str
    cause: List[str]
    timestamp: str
    image: str


# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["schoolDB"]
students = db["students"]
violations_col = db["violations"]

# Mapping FAISS index -> metadata (MongoDB doc fields)
index_to_metadata = {} 


EMBEDDING_SIZE = 512

# -------------------------------
# FAISS Setup
# -------------------------------
index = faiss.IndexFlatIP(EMBEDDING_SIZE)  # inner product = cosine


def l2_normalize(vec):
    vec = np.array(vec).astype("float32")
    return vec / np.linalg.norm(vec)


def add_student_images(student_id, image_paths, student_name=None, photo_folder=None):
    global index_to_metadata, index

    new_indexes = []

    for img_path in image_paths:
        embedding = DeepFace.represent(
            img_path, model_name="ArcFace", enforce_detection=False
        )[0]["embedding"]
        embedding = l2_normalize(embedding)

        # Add to FAISS
        index.add(np.expand_dims(embedding, axis=0))
        embedding_index = index.ntotal - 1
        new_indexes.append(embedding_index)

        # Update in-memory cache
        index_to_metadata[embedding_index] = {
            "studentId": student_id,
            "studentName": student_name,
            "photoFolder": photo_folder,
            "embeddingIndex": embedding_index
        }

    # Save / update MongoDB (one document per student)
    students.update_one(
        {"studentId": student_id},
        {
            "$set": {
                "studentName": student_name,
                "photoFolder": photo_folder
            },
            "$push": {
                "embeddingIndexes": {"$each": new_indexes}  # append all new embeddings
            }
        },
        upsert=True
    )

def recognize_face(face_img, threshold=0.35):
    embedding = DeepFace.represent(face_img, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
    embedding = l2_normalize(embedding)
    print("Embedding generated:", embedding[:5], "...")  # first 5 values

    D, I = index.search(np.expand_dims(embedding, axis=0), 1)
    similarity = D[0][0]
    idx = I[0][0]
    print(f"Top FAISS index: {idx}, similarity: {similarity}")

    if similarity >= threshold:
        metadata = index_to_metadata.get(idx)
        if metadata:
            print("Match found:", metadata["studentId"])
            return {
                "studentId": metadata["studentId"],
                "studentName": metadata.get("studentName"),
                "photoFolder": metadata.get("photoFolder"),
                "embeddingIndex": idx,
                "similarity": float(similarity)
            }

    print("No match found")
    return {"studentId": "Unknown", "similarity": None}



def save_index(index_path="data/faiss_index/faiss_cosine.index"):
    os.makedirs(os.path.dirname(index_path), exist_ok=True) 
    faiss.write_index(index, index_path)
    print(f"[INFO] FAISS index saved to {index_path}")


def load_index(index_path="data/faiss_index/faiss_cosine.index"):
    global index, index_to_metadata

    if os.path.exists(index_path):
        index = faiss.read_index(index_path)

        index_to_metadata = {}
        global_emb_idx = 0  # unique index across all students
        for doc in students.find({}, {"_id": 1, "studentId": 1, "studentName": 1, "photoFolder": 1, "embeddingIndexes": 1}):
            for embedding in doc.get("embeddingIndexes", []):
                index_to_metadata[global_emb_idx] = {
                    "_id": doc["_id"],
                    "studentId": doc["studentId"],
                    "studentName": doc.get("studentName"),
                    "photoFolder": doc.get("photoFolder"),
                }
                global_emb_idx += 1  # increment for each embedding

        print(f"[INFO] Loaded {len(index_to_metadata)} embeddings into memory.")
        for emb_idx, metadata in index_to_metadata.items():
            print(f"Embedding Index: {emb_idx} -> Metadata: {metadata}")

    else:
        print("[WARNING] No FAISS index found, starting empty.")




def delete_student(student_id, index_path="data/faiss_index/faiss_cosine.index"):
    global index, index_to_metadata

    # 1. Get student folder(s) first
    student_docs = list(students.find({"studentId": student_id}, {"photoFolder": 1}))
    if not student_docs:
        return {"status": "not_found", "student_id": student_id}

    # 2. Delete student from MongoDB
    students.delete_one({"studentId": student_id})
    print(f"[INFO] Deleted student {student_id} from MongoDB.")

    # 3. Delete folders
    for doc in student_docs:
        folder = doc.get("photoFolder")
        if folder and os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)
            print(f"[INFO] Deleted folder: {folder}")

    # 4. Rebuild FAISS + cache
    index = faiss.IndexFlatIP(EMBEDDING_SIZE)
    index_to_metadata = {}

    docs = list(students.find({}, {"studentId": 1, "studentName": 1, "photoFolder": 1, "embeddingIndexes": 1}))
    for doc in docs:
        studentId = doc["studentId"]
        studentName = doc.get("studentName")
        photoFolder = doc.get("photoFolder")
        embeddingIndexes = doc.get("embeddingIndexes", [])

        for idx, embedding in enumerate(embeddingIndexes):
            embedding = np.array(embedding, dtype="float32")
            index.add(np.expand_dims(embedding, axis=0))
            faiss_idx = index.ntotal - 1

            index_to_metadata[faiss_idx] = {
                "studentId": studentId,
                "studentName": studentName,
                "photoFolder": photoFolder
            }

    save_index(index_path)
    return {"status": "deleted", "student_id": student_id}





def reset_faiss_and_db(index_path="data/faiss_index/faiss_cosine.index"):
    global index, index_to_metadata

    # 1. Reset in-memory FAISS index
    index = faiss.IndexFlatIP(EMBEDDING_SIZE)
    index_to_metadata = {}

    # 2. Clear MongoDB
    students.delete_many({})
    print("[INFO] Cleared MongoDB student collection.")

    # 3. Remove FAISS index file
    if os.path.exists(index_path):
        os.remove(index_path)
        print(f"[INFO] Deleted FAISS index file: {index_path}")

    # 4. Delete all student folders
    faces_root = "faces_db"
    if os.path.exists(faces_root):
        shutil.rmtree(faces_root, ignore_errors=True)
        os.makedirs(faces_root, exist_ok=True)
        print(f"[INFO] Cleared all student photo folders in {faces_root}")

    # 5. Save fresh empty FAISS index
    save_index(index_path)

    return {"status": "reset_done"}



def get_violations_for_student_today(student_id: str, date: datetime.date):
    date_str = date.isoformat()  # '2025-11-15'
    
    query = {
        "studentId": student_id,
        "timestamp": {"$regex": f"^{date_str}"}
    }
    return list(violations_col.find(query))




def add_violation(v: Violation):
    from datetime import datetime, date

    # Extract just the date part from the timestamp
    violation_date = datetime.fromisoformat(v.timestamp).date()

    # Get violations for this student on the same day
    todays_violations = get_violations_for_student_today(v.studentId, violation_date)

    # Find any existing record with overlapping cause(s) today
    overlapping_violation = None
    for record in todays_violations:
        if set(record.get("cause", [])) & set(v.cause):
            overlapping_violation = record
            break

    if overlapping_violation:
        existing_causes = set(overlapping_violation["cause"])
        new_causes = set(v.cause)

        if new_causes.issubset(existing_causes):
            # Duplicate violation for today; skip insertion
            print(f"[VIOLATION] Skipped duplicate violation record for {v.studentId} with cause(s) {v.cause} on {violation_date}")
            return None
        else:
            # Partial new causes; insert only unique ones
            unique_causes = list(new_causes - existing_causes)
            if unique_causes:
                v.cause = unique_causes
                violations_col.insert_one(v.dict())
                print(f"[VIOLATION] Added additional violation '{v.cause}' for student {v.studentId} on {violation_date}")
                return v
            else:
                print(f"[VIOLATION] No new unique causes to add for {v.studentId} on {violation_date}")
                return None
    else:
        # No overlapping violation today; insert new record
        violations_col.insert_one(v.dict())
        print(f"[VIOLATION] Added '{v.cause}' for student {v.studentId} on {violation_date}")
        return v
