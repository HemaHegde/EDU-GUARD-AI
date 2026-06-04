from sqlalchemy import Column, Integer, String, Float
from database.db import Base

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, unique=True, index=True)
    course = Column(String)
    education_level = Column(String)
    age_band = Column(String)

    risk_score = Column(Float, default=0.0)
    risk_level = Column(String, default="Low")
    persona = Column(String, default="Consistent Learner")