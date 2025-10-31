import io
from fileinput import filename

from fastapi import APIRouter, UploadFile, File, Form
import os
import subprocess
from fastapi import HTTPException
from pydantic import BaseModel
import requests
import json
import uuid
from fastapi import BackgroundTasks
from sympy.integrals.risch import residue_reduce

# routes/session.py
from app.database.database import supabase
from app.database.model import InterviewSession, SessionStatus
from datetime import datetime, timedelta, timezone
from typing import List, Optional

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
router = APIRouter()

# ---------- Helper functions ----------

def extract_audio(video_path: str) -> bytes:
    """
    Extract mono 16kHz WAV audio from a video and return it as bytes.
    """
    print("video_path:", video_path)
    command = [
        "ffmpeg",
        "-i", video_path,       # input video
        "-vn",                  # no video
        "-acodec", "pcm_s16le", # raw WAV
        "-ar", "16000",         # 16 kHz
        "-ac", "1",             # mono
        "-f", "wav",            # force wav format
        "pipe:1"                # output to stdout
    ]

    # Run ffmpeg and capture stdout
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    print(result)
    return result.stdout  # this is the full audio file as bytes


async def answer12(
    file: UploadFile = File(...),
    questionId: str = Form(...),
    questionText: str = Form(...)
):
    try:
        # --- Step 1: Save uploaded file ---
        contents = await file.read()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  # e.g., 20250914123045
        file_path = os.path.join(UPLOAD_DIR, f"{questionId}_{timestamp}_{file.filename}")

        with open(file_path, "wb") as f:
            f.write(contents)

        # --- Step 2: Extract audio (convert to wav) ---
        audio_path = os.path.splitext(file_path)[0] + ".wav"
        extract_audio(file_path, audio_path)

        # --- Step 3: Transcribe using Groq Whisper API ---
        with open(audio_path, "rb") as audio_file:
            response = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
                files={"file": audio_file},
                data={"model": "whisper-large-v3"}  # can use medium/small if you want faster
            )
            response.raise_for_status()
            transcript = response.json()["text"].strip()

        # --- Step 4: Send transcript to AI evaluation ---
        ai_response_str = get_ai_evaluation(questionText, transcript)

        # --- Step 5: Parse AI evaluation JSON ---
        try:
            evaluation_data = json.loads(ai_response_str)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="AI response was not valid JSON")

        # --- Step 6: Return combined response ---
        return {
            "status": "ok",
            "questionId": questionId,
            "questionText": questionText,
            "fileName": file.filename,
            "fileSize": len(contents),
            "audioFile": os.path.basename(audio_path),
            "transcript": transcript,
            "evaluation": evaluation_data,
        }

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {e}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Groq API: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_ai_evaluation(question, user_answer, code):
    """
    Sends the prompt to the Groq/OpenAI API and returns its response as JSON.
    """
    system_prompt = """
    You are a fair and consistent technical interviewing evaluator.
    You must evaluate the candidate's answer to the given question and return scores consistently.

    Scoring Rules:
    - If the answer is completely empty, whitespace, or just filler (e.g., "I don’t know"), 
      set all scores to 0 and feedback = "No answer provided."
    - If the answer is unrelated to the question, 
      set all scores to 0 and feedback = "Answer is irrelevant to the question."
    - Otherwise, score based on:
      - Correctness: Logic and facts (0–10)
      - Completeness: Covers all requirements (0–10)
      - Clarity: Explanation understandable (0–10)
      - Grammar: Writing clarity and correctness (0–10)
      - Confidence: Apparent confidence in response (0–10)

    Additional Rules:
    - Short factual answers (like numeric or single-statement results) are acceptable if they match the correct value.
      These should score 7–8 for correctness, even if brief.
    - Partially correct but incomplete answers → 4–6.
    - Mostly correct with minor gaps → 7–8.
    - Fully correct, clear, and well-explained with examples → 9–10.

    Output strictly as JSON:
    {
      "score": int,
      "confidence": int,
      "grammar": int,
      "clarity": int,
      "completeness": int,
      "feedback": string
    }
    """

    full_prompt = f"""
    ### Question:
    {question}

    ### Candidate's Code (if any):
    {code}

    ### Candidate's Answer:
    {user_answer}

    ### Evaluation:
    Evaluate correctness even if the answer is short but factually right (like numeric results). 
    """

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ],
        "temperature": 0.2,
        "stream": False,
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        ai_text = response.json()["choices"][0]["message"]["content"]
        return ai_text

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with the AI model: {e}")


def get_ai_questions(
    job_title,
    experience_years,
    seniority_level,
    skills=None,
    industry=None,
    num_questions=5,
    round_type=None,
    user_instructions=None,
):
    """
    Generates a tailored number of interview questions based on candidate profile and user instructions.

    Parameters:
    - job_title (str): The job role for which questions are needed.
    - experience_years (int): Candidate's years of experience.
    - seniority_level (str): e.g., "entry", "Junior", "Mid", "Senior", "Lead".
    - skills (list of str, optional): Key skills or technologies.
    - industry (str, optional): Industry context.
    - num_questions (int): Number of questions to generate (default: 5)
    - round_type (str, optional): Which round of interview (technical, HR, coding, etc.)
    - user_instructions (str, optional): Free-form instructions from the admin about how to generate the questions.

    Returns:
    - List of questions (list of str)
    """

    system_prompt = f"""
    You are an expert recruiter and interviewer. Generate exactly {num_questions} interview questions
    based on the candidate profile and the interview round: {round_type}.

    Admin instructions: {user_instructions if user_instructions else "No extra instructions provided."}

    Rules:
    1. Prioritize user_instructions, then skills, then general field knowledge.
    2. Questions should be clear, concise, and realistic for a normal human interview.
    3. Difficulty should match the candidate's experience and seniority.
    4. Include only questions relevant to the specified round type.
    5. Output only valid JSON in the format below — never wrap it in quotes or escape it:

    {{
      "questions": [
        {{
          "isCodingRequired": true/false,
          "MaxTimeToAnswer": <integer seconds>,
          "Question": "Question text here"
        }},
        ...
      ]
    }}

    Candidate Info:
    - Job Title: {job_title}
    - Experience: {experience_years} years
    - Seniority Level: {seniority_level}
    - Skills: {', '.join(skills) if skills else 'Not specified'}
    - Industry: {industry if industry else 'General'}
    """

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Generate interview questions based on the above info."}
        ],
        "temperature": 0.5,
        "stream": False,
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
            json=payload,
            timeout=120
        )
        print(response)
        response.raise_for_status()

        questions = response.json()["choices"][0]["message"]["content"]

        return questions

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error generating questions from AI: {e}")





def get_questions(user_prompt: str, temperature: float = 0.5):
    """Generate interview questions with a fixed output schema"""

    system_prompt = """
You are an AI that generates interview questions in JSON format.
Always follow this schema strictly:

{{
  "questions": [
    {{
      "isCodingRequired": true/false,
      "MaxTimeToAnswer": <integer seconds>,
      "Question": "Question text here"
    }},
    ...
  ]
}}

- Do not include anything outside the JSON.
- Keep the questions clear and relevant to the user prompt.
- Ensure valid JSON that can be parsed directly.
"""

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "stream": False,
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]

        # Try parsing as JSON
        try:
            questions = json.loads(content)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Model did not return valid JSON")

        return questions

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error generating questions from AI: {e}")



class CampaignRequest(BaseModel):
    campaign_id: str

class GetQuestions(BaseModel):
    candidate_id: str
    campaign_id: str
    session_id:str
import logging
from fastapi import APIRouter, BackgroundTasks, Form, File, UploadFile
from datetime import datetime
import os

logger = logging.getLogger("uvicorn.error")  # use uvicorn logger for console output

@router.post("/answer")
async def get_questions_route(
    background_tasks: BackgroundTasks,
    candidate_id: str = Form(...),
    campaign_id: str = Form(...),
    session_id: str = Form(...),
    file: UploadFile = File(...),
    finished: str = Form(...),
    code: str = Form(...),
):
    # Step 1: Verify candidate
    candidate_result = (
        supabase.table("candidate")
        .select("*")
        .eq("id", candidate_id)
        .eq("campaign_id", campaign_id)
        .execute()
    )

    if not candidate_result.data:
        return {"error": "Invalid session or candidate not found"}

    candidate = candidate_result.data[0]

    if candidate.get("status") == "completed":
        return {
            "success": True,
            "status": "interview-completed",
            "message": "Your interview has already been completed. Evaluation is running in the background."
        }

    # Step 2: Verify session
    verify_session({"session_id": session_id})

    # Step 3: Save uploaded video
    contents = await file.read()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{candidate_id}_{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    video_link = filename

    with open(file_path, "wb") as f:
        f.write(contents)

    # Step 4: Get current question
    current_question_number = candidate.get("current_question", 1)

    total_questions_result = (
        supabase.table("interview_question")
        .select("id")
        .eq("candidate_id", candidate_id)
        .eq("session_id", session_id)
        .execute()
    )
    total_questions = len(total_questions_result.data)

    # Step 5: Get current question data
    question_result = (
        supabase.table("interview_question")
        .select("*")
        .eq("candidate_id", candidate_id)
        .eq("session_id", session_id)
        .eq("question_number", current_question_number)
        .execute()
    )

    if not question_result.data:
        return {"error": "No question found for this candidate"}

    current_question = question_result.data[0]

    # ✅ Step 6: Record answer before checking finish
    eval_data = {
        "id": current_question.get("id"),
        "candidate_id": candidate_id,
        "answer": None,
        "code": code,
        "score": None,
        "video_link": video_link,
        "session_id": session_id,
    }
    supabase.table("interview_evaluation").insert(eval_data).execute()

    # ✅ Step 7: Handle last question properly
    if finished.lower() == "true" or current_question_number >= total_questions:
        supabase.table("candidate").update({"status": "completed"}).eq("id", candidate_id).execute()
        background_tasks.add_task(finish_interview, session_id, candidate_id)
        return {
            "success": True,
            "status": "interview-completed",
            "message": "Your interview has been recorded. Evaluation will run in the background."
        }

    # Step 8: Move to next question
    new_question_number = current_question_number + 1
    supabase.table("candidate").update({
        "current_question": new_question_number
    }).eq("id", candidate["id"]).execute()

    next_question_result = (
        supabase.table("interview_question")
        .select("*")
        .eq("candidate_id", candidate_id)
        .eq("session_id", session_id)
        .eq("question_number", new_question_number)
        .execute()
    )

    if not next_question_result.data:
        return {"error": "No question found for this candidate"}

    next_question = next_question_result.data[0]
    is_last_question = new_question_number >= total_questions

    # Step 9: Return next question
    return {
        "question": {
            "question_number": next_question["question_number"],
            "question": next_question["question"],
            "coding": next_question["coding"],
            "time_limit": next_question["time_limit"]
        },
        "is_last_question": is_last_question,
        "status": "in-progress"
    }







@router.post("/start-session")
def start_session(payload: dict):
    user_id = payload["candidate_id"]
    session = InterviewSession(
        user_id=user_id,
        start_time=datetime.utcnow(),
        expiry_time=datetime.utcnow() + timedelta(minutes=30),
        status=SessionStatus.ACTIVE
    )

    # Convert to JSON-safe dict
    data = session.model_dump(mode="json", exclude={"id"})
    response = supabase.table("interview_sessions").insert(data).execute()
    return response.data[0] if response.data else {"error": "Failed to create session"}



# ✅ End Session
@router.post("/end-session")
def end_session(payload: dict):
    session_id = payload["session_id"]

    # Update status to completed
    response = (
        supabase.table("interview_sessions")
        .update({"status": SessionStatus.COMPLETED.value})
        .eq("id", session_id)
        .execute()
    )

    if response.data:
        return {"message": "Session ended successfully", "session": response.data[0]}
    return {"error": "Session not found or failed to update"}


# ✅ Verify Session
@router.post("/verify-session")
def verify_session(payload: dict):
    session_id = payload["session_id"]

    # Fetch the session
    response = (
        supabase.table("interview_sessions")
        .select("*")
        .eq("id", session_id)
        .execute()
    )

    if not response.data:
        return {"valid": False, "message": "Session not found"}

    session = response.data[0]

    # Check expiry
    expiry_time = datetime.fromisoformat(session["expiry_time"].replace("Z", "+00:00"))
    if datetime.utcnow() > expiry_time:
        # Expire the session in DB
        supabase.table("interview_sessions").update(
            {"status": SessionStatus.EXPIRED.value}
        ).eq("id", session_id).execute()
        return {"valid": False, "message": "Session expired"}

    # Check if active
    if session["status"] != SessionStatus.ACTIVE.value:
        return {"valid": False, "message": f"Session is {session['status']}"}

    return {"valid": True, "message": "Session is active", "session": session}


class CandidateStatusResponse(BaseModel):
    success: bool
    message: str
    status: str | None = None


@router.get("/verify/{campaign_id}/{candidate_id}", response_model=CandidateStatusResponse)
async def verify_candidate(campaign_id: str, candidate_id: str):
    # Query candidate table
    result = supabase.table("candidate") \
        .select("status, created_at") \
        .eq("id", candidate_id) \
        .eq("campaign_id", campaign_id) \
        .execute()

    if result.data is None:
        raise HTTPException(status_code=404, detail="Candidate not found")

    status = result.data[0]["status"]
    created_at = result.data[0]["created_at"]

    # Convert to datetime
    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

    # ⏳ Check if link expired (older than 2 days)
    if created_at < datetime.now(timezone.utc) - timedelta(days=2):
        raise HTTPException(status_code=400, detail="Candidate link expired")

    # ❌ If already started
    if status != "not_started":
        raise HTTPException(status_code=400, detail="Candidate interview already started or completed")

    # ✅ Candidate is valid
    return CandidateStatusResponse(
        success=True,
        message="Candidate verified successfully",
        status=status)


class StartInterviewRequest(BaseModel):
    candidate_id: str
    campaign_id: str

@router.post("/start-interview")
def start_interview(request: StartInterviewRequest):
    candidate_id = request.candidate_id
    campaign_id = request.campaign_id

    # Fetch candidate
    candidate_result = (
        supabase.table("candidate")
        .select("*")
        .eq("id", candidate_id)
        .eq("campaign_id", campaign_id)
        .execute()
    )
    candidate_data = candidate_result.data[0]
    if not candidate_result.data:
        return {"error": "Candidate not found"}
    if candidate_data["status"] == "completed":
        return {"success": True, "status": "interview-completed",
                "message": "Your interview has been recorded. Evaluation will run in the background."}


    # If interview already completed
    if candidate_data["status"] == "completed":
        return {"message": "Interview already completed"}

    # If interview already in progress
    if candidate_data["status"] == "processing":
        current_question_number = candidate_data.get("current_question", 1)

        # Fetch the current question
        question_result = (
            supabase.table("interview_question")
            .select("*")
            .eq("candidate_id", candidate_id)
            .eq("question_number", current_question_number)
            .execute()
        )

        if not question_result.data:
            return {"error": "No question found for this candidate"}

        current_question = question_result.data[0]

        # Check total questions
        total_questions_result = (
            supabase.table("interview_question")
            .select("id")
            .eq("candidate_id", candidate_id)
            .eq("session_id", current_question["session_id"])
            .execute()
        )
        total_questions = len(total_questions_result.data)
        is_last_question = current_question_number >= total_questions

        # Update candidate table to point to NEXT question
        if not is_last_question:
            supabase.table("candidate").update({
                "current_question": current_question_number + 1,
            }).eq("id", candidate_data["id"]).execute()

        return {
            "message": "Interview already in progress",
            "session_id": current_question["session_id"],
            "question": {
                "question_number": current_question["question_number"],
                "question": current_question["question"],
                "coding": current_question["coding"],
                "time_limit": current_question["time_limit"]
            },
            "isLastQuestion": is_last_question
        }

    # If starting fresh
    session_data = start_session({"candidate_id": candidate_id})
    session_id = session_data["id"]

    # Fetch campaign prompt & temperature
    campaign_result = (
        supabase.table("interview_campaign")
        .select("final_prompt, temperature")
        .eq("id", campaign_id)
        .execute()
    )

    if not campaign_result.data:
        return {"error": "Campaign not found"}

    final_prompt = campaign_result.data[0]["final_prompt"]
    temperature = campaign_result.data[0]["temperature"]

    # Generate questions
    que = get_questions(final_prompt, temperature)
    questions = que.get("questions", [])
    if not questions:
        return {"error": "No questions generated"}

    # Insert questions into interview_question table
    insert_payload = []
    for idx, q in enumerate(questions, start=1):
        insert_payload.append({
            "id": str(uuid.uuid4()),
            "question_number": idx,
            "question": q.get("Question"),
            "coding": q.get("isCodingRequired", False),
            "time_limit": q.get("MaxTimeToAnswer", 60),
            "candidate_id": candidate_id,
            "session_id": session_id
        })

    supabase.table("interview_question").insert(insert_payload).execute()

    # Send first question
    first_question = insert_payload[0]

    # Check total questions
    total_questions_result = (
        supabase.table("interview_question")
        .select("id")
        .eq("candidate_id", candidate_id)
        .eq("session_id", session_id)
        .execute()
    )
    total_questions = len(total_questions_result.data)
    is_last_question = total_questions <= 1

    # Update candidate table to NEXT question only if not last
    supabase.table("candidate").update({
        "status": "processing",
        "current_question": 1,
    }).eq("id", candidate_data["id"]).execute()

    return {
        "message": "Interview started",
        "session_id": session_id,
        "question": {
            "question_number": first_question["question_number"],
            "question": first_question["question"],
            "coding": first_question["coding"],
            "time_limit": first_question["time_limit"]
        },
        "isLastQuestion": is_last_question
    }
@router.post("/finish")
async def finish_interview(session_id: str, candidate_id: str):
    # 1️⃣ Fetch interview evaluations filtered by session_id and candidate_id
    evaluations = supabase.table("interview_evaluation") \
        .select("id, video_link, code") \
        .eq("session_id", session_id) \
        .eq("candidate_id", candidate_id) \
        .execute()

    if not evaluations.data:
        return {"error": "No evaluations found for this session/candidate."}

    # 2️⃣ Collect all question IDs
    question_ids = [e["id"] for e in evaluations.data]

    # 3️⃣ Fetch the corresponding questions
    questions = supabase.table("interview_question") \
        .select("id, question") \
        .in_("id", question_ids) \
        .execute()

    if not questions.data:
        return {"error": "No matching questions found."}

    # 4️⃣ Merge evaluations with questions, keeping evaluation ID
    merged = []
    for eval_item in evaluations.data:
        question = next((q for q in questions.data if q["id"] == eval_item["id"]), None)
        if question:
            merged.append({
                "evaluation_id": eval_item["id"],  # needed for update
                "code":eval_item["code"],
                "question": question["question"],
                "video_link": eval_item["video_link"]
            })

    # 5️⃣ Process all questions and update evaluations
    for item in merged:
        result = await answer(item["video_link"], item["question"],item["code"])
        update_data = {
            "score": result["evaluation"]["score"],
            "confidence": result["evaluation"]["confidence"],
            "grammar": result["evaluation"]["grammar"],
            "clarity": result["evaluation"]["clarity"],
            "completeness": result["evaluation"]["completeness"],
            "feedback": result["evaluation"]["feedback"],
            "answer": result["answer"]
        }
        supabase.table("interview_evaluation") \
            .update(update_data) \
            .eq("id", item["evaluation_id"]) \
            .execute()

    evaluations = supabase.table("interview_evaluation") \
        .select("score") \
        .eq("candidate_id", candidate_id) \
        .eq("session_id", session_id) \
        .execute()

    scores = [e["score"] for e in evaluations.data if e.get("score") is not None]
    avg_score = sum(scores) / len(scores) if scores else None

    print(avg_score)

    print("Average score:", avg_score)
    # Assume avg_score is calculated as before
    if avg_score is not None:
        response = supabase.table("candidate") \
            .update({
            "score": avg_score,
            "status": "completed"
                }) \
            .eq("id", candidate_id) \
            .execute()

        if response.data:
            print(f"Candidate {candidate_id} score updated to {avg_score}")
        else:
            print("Failed to update candidate score")
    else:
        print("No score to update")
    end_session({"session_id":session_id})
    return {"status": "ok"}


async def answer(
    video_link: str,
    questionText: str,
    code: str,
):
    try:
        # --- Step 2: Extract audio (convert to wav) ---
        video_filename = video_link

        # Path to the local file in your uploads/static folder
        video_path = os.path.join("uploads", video_filename)  # adjust based on your folder structure

        # Extract audio
        audio_bytes = extract_audio(video_path)
        response = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
                files={"file": ("audio.wav", io.BytesIO(audio_bytes))},  # wrap bytes
                data={"model": "whisper-large-v3"}  # can use medium/small if you want faster
            )
        response.raise_for_status()
        transcript = response.json()["text"].strip()

        # --- Step 4: Send transcript to AI for light correction ---
        payload = {
            "model": "openai/gpt-oss-120b",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert at lightly correcting messy transcriptions. "
                        "Fix repeated words, filler words, and small grammatical mistakes. "
                        "Preserve all numbers, quantities, and factual information exactly as they appear in the original text. "
                        "Do NOT interpret, change, or correct any numbers, even if they seem unusual. "
                        "Preserve the original wording and meaning as closely as possible. "
                        "Only output the corrected text, no explanations."
                    )
                },
                {
                    "role": "user",
                    "content": f"Rewrite the following raw transcription into clean, correct English:\n\n{transcript}"
                }
            ]
        }

        ai_response =requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
            json=payload,
            timeout=120
        )
        ai_response.raise_for_status()
        ai_cleaned_text = ai_response.json()["choices"][0]["message"]["content"].strip()

        # --- Step 4: Send transcript to AI evaluation ---
        ai_response_str = get_ai_evaluation(questionText, ai_cleaned_text,code)

        # --- Step 5: Parse AI evaluation JSON ---
        try:
            evaluation_data = json.loads(ai_response_str)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="AI response was not valid JSON")
        # --- Step 6: Return combined response ---
        return {
            "answer": transcript,
            "evaluation": evaluation_data,
        }

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {e}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Groq API: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@router.get("/view-details")
def view_details(candidate_id: str):
    # Step 1: Fetch all interview questions for candidate
    print(candidate_id)
    questions = (
        supabase.table("interview_question")
        .select("*")
        .eq("candidate_id", candidate_id)
        .execute()
    )

    # Step 2: Extract question IDs
    question_ids = [q["id"] for q in questions.data]
    print(question_ids)
    # Step 3: Fetch evaluations that match these IDs
    evaluations = (
        supabase.table("interview_evaluation")
        .select("*")
        .in_("id", question_ids)
        .execute()
    )

    # Step 4: Merge both results by ID (1–1 mapping)
    merged = []
    for q in questions.data:
        eval_item = next((e for e in evaluations.data if e["id"] == q["id"]), None)
        merged.append({
            **q,
            "evaluation": eval_item
        })

    return merged
