from pydantic import BaseModel


class StudentCreate(BaseModel):
    student_id: str
    course: str
    education_level: str
    age_band: str