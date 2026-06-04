from database.db import engine, Base
from models.student import Student

print("Creating EduGuard-AI database tables...")

Base.metadata.create_all(bind=engine)

print("Database tables created successfully!")