class UserResponse(BaseModel):
    assistant_message: str
    suggestions: list

class OrgResponse(BaseModel):
    user_id: str
    burnout_level: str
    burnout_score: float
    dominant_signals: list
