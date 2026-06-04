from fastapi import APIRouter
from sqlalchemy.orm import Session

from database.db import SessionLocal
from models.student import Student
from schemas.student_schema import StudentCreate

router = APIRouter()


# Database Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Get All Students
@router.get("/students")
def get_students():
    db: Session = SessionLocal()

    students = db.query(Student).all()

    db.close()

    return students


# Create Student Dynamically
@router.post("/students")
def create_student(student: StudentCreate):
    db: Session = SessionLocal()

    new_student = Student(
        student_id=student.student_id,
        course=student.course,
        education_level=student.education_level,
        age_band=student.age_band,
        risk_score=0.0,
        risk_level="Low",
        persona="Consistent Learner"
    )

    db.add(new_student)
    db.commit()
    db.refresh(new_student)

    db.close()

    return {
        "message": "Student added successfully",
        "student_id": new_student.student_id,
        "course": new_student.course
    }