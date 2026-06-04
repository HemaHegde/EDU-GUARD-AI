from fastapi import APIRouter
from fastapi import UploadFile
from fastapi import File

from pydantic import BaseModel

import shutil
import os

from config.supabase_client import supabase

from services.pdf_learning_service import (
    generate_learning_material
)

router = APIRouter()


# =========================
# QUIZ ATTEMPT MODEL
# =========================

class QuizAttemptRequest(
    BaseModel
):

    user_id: str

    pdf_name: str

    score: int

    total: int


# =========================
# HOME
# =========================

@router.get("/")
def home():

    return {

        "module":
        "PDF Learning Intelligence Running"

    }


# =========================
# PDF UPLOAD
# =========================

@router.post("/upload")
async def upload_pdf(

    file: UploadFile = File(...)

):

    try:

        upload_dir = os.path.join(
            os.getcwd(),
            "uploads"
        )

        os.makedirs(
            upload_dir,
            exist_ok=True
        )

        file_path = os.path.join(

            upload_dir,

            file.filename

        )

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        result = generate_learning_material(

            pdf_path=file_path,

            user_id=None,

            pdf_name=file.filename

        )

        return {

            "status":
            "success",

            "filename":
            file.filename,

            "result":
            result

        }

    except Exception as e:

        return {

            "status":
            "error",

            "message":
            str(e)

        }


# =========================
# SAVE QUIZ ATTEMPT
# =========================

@router.post(
    "/save-attempt"
)
async def save_attempt(

    request:
    QuizAttemptRequest

):

    try:

        supabase.table(
            "quiz_attempts"
        ).insert({

            "user_id":
            request.user_id,

            "quiz_topic":
            request.pdf_name,

            "score":
            request.score,

            "total":
            request.total

        }).execute()

        return {

            "status":
            "success",

            "message":
            "Quiz attempt saved"

        }

    except Exception as e:

        return {

            "status":
            "error",

            "message":
            str(e)

        }