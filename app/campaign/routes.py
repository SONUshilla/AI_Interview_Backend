import json
from typing import List

from fastapi import APIRouter, Depends,HTTPException
import requests
from pydantic import BaseModel
import os
from app.database.database import supabase
from app.database.dependencies import get_current_user
from app.interview.routes import get_questions
router = APIRouter()


def create_sample_prompt(
        job_title: str = "Frontend Developer",
        experience_years: int = 5,
        seniority_level: str = "mid",
        skills: list[str] = None,
        industry: str = "General",
        round_type: str = "Technical",
        num_questions: int = 5,
        user_instructions: str = ""
):
    skills = skills or ["JavaScript", "React", "HTML/CSS"]

    system_prompt = f"""
You are an expert recruiter and interviewer. Generate exactly {num_questions} interview questions
based on the candidate profile and the interview round: {round_type}.

Admin instructions: {user_instructions if user_instructions else "No extra instructions provided."}

Rules:
1. Prioritize user_instructions, then skills, then general field knowledge.
2. Questions should be clear, concise, and realistic for a normal human interview.
3. Difficulty should match the candidate's experience and seniority.
4. Include only questions relevant to the specified round type.



Candidate Info:
- Job Title: {job_title}
- Experience: {experience_years} years
- Seniority Level: {seniority_level}
- Skills: {', '.join(skills) if skills else 'Not specified'}
- Industry: {industry if industry else 'General'}

Feedback: updates questions based on the feedbacks
"""
    return system_prompt


from fastapi import Form
from uuid import uuid4
def format_questions_as_numbered_string(ai_response: dict):
    """
    Converts AI dict response into a single numbered string.
    """
    questions = ai_response.get("questions", [])
    numbered_str = "\n".join(
        f"{i+1}. {q['Question']}" for i, q in enumerate(questions)
    )
    return numbered_str

from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import JSONResponse
from uuid import uuid4
import httpx

router = APIRouter()

@router.post("/CreateNewCampaign")
def create_new_campaign(
    user=Depends(get_current_user),
    campaign_name: str = Form(...),
    job_title: str = Form(...),
    experience_years: int = Form(...),
    seniority_level: str = Form(...),
    skills: str = Form(None),  # comma-separated
    industry: str = Form(None),
    num_questions: int = Form(5),
    round_type: str = Form(None),
    user_instructions: str = Form(None)
):
    try:
        # Convert skills string to list
        skills_list = [s.strip() for s in skills.split(",")] if skills else []

        # Generate default final_prompt
        final_prompt = create_sample_prompt(
            job_title=job_title,
            experience_years=experience_years,
            seniority_level=seniority_level,
            skills=skills_list,
            industry=industry,
            round_type=round_type or "Technical",
            num_questions=num_questions,
            user_instructions=user_instructions
        )

        campaign_id = str(uuid4())
        campaign = {
            "id": campaign_id,
            "campaign_name": str(campaign_name),
            "user_id": user['user_id'],
            "job_title": job_title,
            "experience_years": experience_years,
            "seniority_level": seniority_level,
            "skills": skills_list,
            "industry": industry,
            "num_questions": num_questions,
            "round_type": round_type,
            "user_instructions": user_instructions,
            "final_prompt": final_prompt,
            "status": "active"
        }

        # Insert campaign
        try:
            response = supabase.table("interview_campaign").insert(campaign).execute()
        except httpx.ConnectError as e:
            return JSONResponse(status_code=503, content={"error": "Database connection failed", "details": str(e)})

        if not response.data:
            return JSONResponse(status_code=500, content={"error": "Failed to insert campaign"})

        # Generate questions
        questions = get_questions(final_prompt)
        formatted_questions = format_questions_as_numbered_string(questions)

        # Store in chat_messages
        chat_message = {
            "id": str(uuid4()),
            "campaign_id": campaign_id,
            "message": formatted_questions,
            "role": "system"
        }

        try:
            response = supabase.table("chat_messages").insert(chat_message).execute()
        except httpx.ConnectError as e:
            return JSONResponse(status_code=503, content={"error": "Failed to save chat messages", "details": str(e)})

        if not response.data:
            return JSONResponse(status_code=500, content={"error": "Failed to save chat message"})

        return {"id": campaign_id}

    except Exception as e:
        # Catch-all safeguard
        return JSONResponse(status_code=500, content={"error": "Unexpected server error", "details": str(e)})






# Pydantic schema for response
class Message(BaseModel):
    role: str  # 'user' or 'system'
    content: str

class GetMessagesRequest(BaseModel):
    id: str  # the chat/session id

@router.post("/getMessages")
async def get_messages(request: GetMessagesRequest):
    chat_id = request.id
    print(chat_id)
    # Fetch messages from Supabase table 'chat_messages'
    response = supabase.table("chat_messages") \
        .select("role, message") \
        .eq("campaign_id", chat_id) \
        .order("created_at", desc=False) \
        .execute()

    # Supabase returns {'data': [...], 'error': ...}
    if response.data and len(response.data) > 0:
        rows=response.data or []
        # Convert to list of dicts
        messages = [{"role": row["role"], "message": row["message"]} for row in rows]
        return messages
    else:
        return {"error": "Failed to get messages"}


def summarise_prompt(current_prompt: str, feedback: str) -> str:
    """Summarise existing prompt + feedback into a single concise prompt"""
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "system",
                "content": "You merge interview instructions and feedback into one concise final prompt."
            },
            {
                "role": "user",
                "content": f"""
Current Final Prompt:
{current_prompt}

New Feedback:
{feedback}

Task: Rewrite into one clear, short final prompt without repetition.
"""
            }
        ],
        "temperature": 0.3,
        "stream": False,
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


@router.post("/UpdateQuestions")
def update_questions(
    campaign_id: str = Form(...),
    feedback_text: str = Form(...),
    temperature: float = Form(...),
):

    # Store in chat_messages table
    chat_message = {
        "id": str(uuid4()),
        "campaign_id": campaign_id,
        "message": feedback_text,
        "role": "user"
    }
    supabase.table("chat_messages").insert(chat_message).execute()
    # 1. Fetch campaign final_prompt
    campaign = supabase.table("interview_campaign") \
        .select("final_prompt") \
        .eq("id", campaign_id) \
        .single() \
        .execute()

    if not campaign.data:
        raise HTTPException(status_code=404, detail="Campaign not found")

    current_prompt = campaign.data.get("final_prompt") or ""

    # 2. Summarise final_prompt + feedback
    updated_prompt = f"{current_prompt}\n\nUser Feedback:\n{feedback_text}"


    # 3. Update table with summarised prompt
    supabase.table("interview_campaign") \
        .update({
        "final_prompt": updated_prompt,
        "temperature": temperature  # <-- new field
    }) \
        .eq("id", campaign_id) \
        .execute()

    questions = get_questions(updated_prompt,temperature)
    # Extract questions and format
    formatted_questions = format_questions_as_numbered_string(questions)
    # Store in chat_messages table
    chat_message = {
        "id": str(uuid4()),
        "campaign_id": campaign_id,
        "message": formatted_questions,
        "role": "system"
    }
    response = supabase.table("chat_messages").insert(chat_message).execute()

    if response.data and len(response.data) > 0:
        return response.data[0]  # return the row itself
    else:
        return {"error": "Failed to create interview campaign"}



@router.post("/FinalizeQuestions")
def finalize_campaign(campaign_id: str = Form(...), temperature: float = Form(...)):
    # 1️⃣ Fetch current final_prompt
    campaign_res = supabase.table("interview_campaign") \
        .select("final_prompt") \
        .eq("id", campaign_id) \
        .single() \
        .execute()

    campaign_data = campaign_res.data
    if not campaign_data:
        raise HTTPException(status_code=404, detail="Campaign not found")

    current_prompt = campaign_data.get("final_prompt", "")
    if not current_prompt.strip():
        raise HTTPException(status_code=400, detail="Final prompt is empty")

    # 2️⃣ Build the payload for Groq API using the passed temperature
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant that refines prompts."
            },
            {
                "role": "user",
                "content": f"Rewrite the following into one clear, short final prompt without repetition:\n\n{current_prompt}"
            }
        ],
        "temperature": temperature,
        "stream": False
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
            json=payload,
            timeout=60
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Groq API request failed: {str(e)}")

    # 3️⃣ Extract the finalized prompt
    final_prompt = response.json()["choices"][0]["message"]["content"].strip()
    if not final_prompt:
        raise HTTPException(status_code=500, detail="Failed to finalize campaign")

    # 4️⃣ Update final_prompt and temperature in Supabase
    update_res = supabase.table("interview_campaign") \
        .update({
            "final_prompt": final_prompt,
            "temperature": temperature
        }) \
        .eq("id", campaign_id) \
        .execute()

    if update_res.data and len(update_res.data) > 0:
        return {"success": "Questions finalized successfully", "final_prompt": final_prompt, "temperature": temperature}
    else:
        raise HTTPException(status_code=500, detail="Failed to update campaign")


@router.get("/GetCampaigns")
def get_campaigns(user=Depends(get_current_user)):
    """
    Fetch all campaigns for a user and return only selected columns.
    """
    try:
        res = (
            supabase.table("interview_campaign")
            .select("id, campaign_name, job_title, experience_years, seniority_level, status, created_at, skills, industry")
            .eq("user_id", user["user_id"])
            .execute()
        )

        if not res.data:
            return []  # return empty list if no campaigns found

        # Optional: convert skills from comma-separated string to list
        campaigns = []
        for c in res.data:
            campaign = {
                "id": c.get("id"),
                "campaign_name": c.get("campaign_name"),
                "job_title": c.get("job_title"),
                "experience_years": c.get("experience_years"),
                "seniority_level": c.get("seniority_level"),
                "status": c.get("status"),
                "created_at": c.get("created_at"),
                "skills": c.get("skills"),
                "industry": c.get("industry"),
            }
            campaigns.append(campaign)

        return campaigns

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch campaigns: {str(e)}")


@router.get("/GetCampaign/{campaign_id}")
def get_campaign(campaign_id: str, user=Depends(get_current_user)):
    """
    Fetch a specific campaign by ID for the authenticated user.
    """
    try:
        # Fetch the campaign with the given ID that belongs to the current user
        res = (
            supabase.table("interview_campaign")
            .select("*")
            .eq("id", campaign_id)
            .eq("user_id", user["user_id"])
            .execute()
        )

        if not res.data:
            raise HTTPException(status_code=404, detail="Campaign not found")

        campaign = res.data[0]

        # Format the response
        return {
            "id": campaign.get("id"),
            "campaign_name": campaign.get("campaign_name"),
            "job_title": campaign.get("job_title"),
            "experience_years": campaign.get("experience_years"),
            "seniority_level": campaign.get("seniority_level"),
            "status": campaign.get("status"),
            "created_at": campaign.get("created_at"),
            "skills": campaign.get("skills"),
            "industry": campaign.get("industry"),
            "num_questions": campaign.get("num_questions"),
            "round_type": campaign.get("round_type")
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch campaign: {str(e)}")


@router.get("/GetCandidates/{campaign_id}")
def get_candidates(campaign_id: str, user=Depends(get_current_user)):
    """
    Fetch all candidates for a specific campaign that belongs to the authenticated user.
    """
    try:
        # First verify the campaign belongs to the user
        campaign_res = (
            supabase.table("interview_campaign")
            .select("id")
            .eq("id", campaign_id)
            .eq("user_id", user["user_id"])
            .execute()
        )

        if not campaign_res.data:
            raise HTTPException(status_code=404, detail="Campaign not found")

        # Now fetch candidates for this campaign
        res = (
            supabase.table("candidate")  # Assuming your table is named "candidate"
            .select("*")
            .eq("campaign_id", campaign_id)
            .execute()
        )

        return res.data if res.data else []

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch candidates: {str(e)}")



from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pydantic import EmailStr

conf = ConnectionConfig(
    MAIL_USERNAME="sonushilla189@gmail.com",
    MAIL_PASSWORD="dpys vrcu eexy fjlc",  # Use Gmail App Password
    MAIL_FROM="sonushilla189@gmail.com",
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
)

@router.post("/AddCandidate/{campaign_id}")
async def add_candidate(campaign_id: str, candidate_data: dict, user=Depends(get_current_user)):
    """
    Add a candidate to a specific campaign that belongs to the authenticated user.
    """
    print(candidate_data)
    try:
        # First verify the campaign belongs to the user
        campaign_res = (
            supabase.table("interview_campaign")
            .select("id")
            .eq("id", campaign_id)
            .eq("user_id", user["user_id"])
            .execute()
        )

        if not campaign_res.data:
            raise HTTPException(status_code=404, detail="Campaign not found")

        # Prepare candidate data
        candidate = {
            "name": candidate_data.get("name"),
            "email": candidate_data.get("email"),
            "campaign_id": campaign_id,
            "status": "not_started",  # Default status
            "score": None,  # Default score
        }

        # Insert candidate into database
        res = supabase.table("candidate").insert(candidate).execute()
        candidate_id = res.data[0]["id"]

        # Construct interview link
        interview_link = f"http://localhost:3000/interview/{campaign_id}/{candidate_id}"

        # Send email
        message = MessageSchema(
            subject="Your Interview Link",
            recipients=[candidate_data["email"]],
            body=f"Hello {candidate_data["name"]},\n\nHere is your interview link: {interview_link}\n\nBest regards,\nTeam",
            subtype="plain"
        )
        fm = FastMail(conf)
        await fm.send_message(message)
        if not res.data:
            raise HTTPException(status_code=500, detail="Failed to add candidate")

        return {"message": "Candidate added successfully", "candidate": res.data[0]}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add candidate: {str(e)}")