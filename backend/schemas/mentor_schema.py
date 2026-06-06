from pydantic import BaseModel


class MentorQuestion(BaseModel):

    user_id: str

    question: str