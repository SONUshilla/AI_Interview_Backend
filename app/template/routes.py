from fastapi import APIRouter, HTTPException, Depends
from supabase import create_client, Client
from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from app.database.database import supabase



# --- Auth Dependency ---
from app.database.dependencies import get_current_user  # adjust path if needed

# --- Schemas ---
class PromptCreate(BaseModel):
    prompt: str

class PromptUpdate(BaseModel):
    prompt: str

class PromptOut(BaseModel):
    id: UUID
    user_id: int
    prompt: Optional[str]

router = APIRouter()

# ➕ Create a new prompt
@router.post("/prompts", response_model=PromptOut)
async def add_prompt(payload: PromptCreate, user=Depends(get_current_user)):
    print(user)
    response = supabase.table("user_prompts").insert({
        "user_id": user["user_id"],   # 👈 from token
        "prompt": payload.prompt
    }).execute()

    if not response.data:
        raise HTTPException(status_code=500, detail="Failed to add prompt")
    return response.data[0]

# 📖 Get all prompts for the logged-in user
@router.get("/prompts", response_model=list[PromptOut])
async def get_prompts(user=Depends(get_current_user)):
    response = supabase.table("user_prompts").select("*").eq("user_id", user["user_id"]).execute()
    return response.data or []

# ✏️ Update a prompt (make sure it belongs to the user)
@router.put("/prompts/{prompt_id}", response_model=PromptOut)
async def update_prompt(prompt_id: UUID, payload: PromptUpdate, user=Depends(get_current_user)):
    response = (
        supabase.table("user_prompts")
        .update({"prompt": payload.prompt})
        .eq("id", str(prompt_id))
        .eq("user_id", user["user_id"])   # 👈 only update if belongs to user
        .execute()
    )
    if not response.data:
        raise HTTPException(status_code=404, detail="Prompt not found or not owned by user")
    return response.data[0]

# ❌ Delete a prompt (make sure it belongs to the user)
@router.delete("/prompts/{prompt_id}")
async def delete_prompt(prompt_id: UUID, user=Depends(get_current_user)):
    response = (
        supabase.table("user_prompts")
        .delete()
        .eq("id", str(prompt_id))
        .eq("user_id", user["user_id"])   # 👈 ownership check
        .execute()
    )
    if not response.data:
        raise HTTPException(status_code=404, detail="Prompt not found or not owned by user")
    return {"message": "Prompt deleted successfully"}
