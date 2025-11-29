# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket,Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import json
import os
from typing import Dict, List, Optional
import base64
import asyncio
from datetime import datetime

# Import our free modules
from interview_agent_free import FreeInterviewAgent
from database import FirebaseManager
from speech_processor import SpeechProcessor

app = FastAPI(title="AI Interview Agent - Free Version", version="1.0.0")

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("interview_agent.html")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components with free models
interview_agent = FreeInterviewAgent()
db_manager = FirebaseManager()
speech_processor = SpeechProcessor()

# Store active interviews
active_interviews: Dict[str, Dict] = {}

@app.get("/")
async def root():
    return {"message": "AI Interview Agent API - Free Version"}

@app.get("/categories")
async def get_categories():
    """Get available interview categories"""
    categories = [
        {
            "id": "technical",
            "name": "Technical Interview",
            "description": "Programming, algorithms, system design questions",
            "duration": "15-20 minutes",
            "difficulty": "Intermediate"
        },
        {
            "id": "behavioral", 
            "name": "Behavioral Interview",
            "description": "Teamwork, leadership, problem-solving scenarios",
            "duration": "10-15 minutes",
            "difficulty": "All Levels"
        },
        {
            "id": "management",
            "name": "Management Interview", 
            "description": "Leadership, strategy, decision-making questions",
            "duration": "20-25 minutes",
            "difficulty": "Advanced"
        },
        {
            "id": "sales",
            "name": "Sales & Marketing",
            "description": "Customer relations, strategy, pitching scenarios", 
            "duration": "15-20 minutes",
            "difficulty": "Intermediate"
        }
    ]
    return categories

def generate_questions(self, category: str, count: int = 8) -> List[str]:
    """Generate 8 random questions from the 20 available per category"""
    try:
        if category not in self.question_templates:
            category = "technical"
        
        # Get all available questions for this category
        all_questions = self.question_templates[category]
        
        # If we have fewer questions than requested, use fallback
        if len(all_questions) < count:
            fallback_questions = self._get_fallback_questions(category, count)
            # Mix with available questions
            mixed_questions = all_questions + fallback_questions
            # Take unique questions up to count
            from random import sample
            return sample(list(set(mixed_questions)), min(count, len(set(mixed_questions))))
        
        # Randomly select 8 questions from the 20 available
        import random
        selected_questions = random.sample(all_questions, min(count, len(all_questions)))
        
        print(f"Selected {len(selected_questions)} questions for {category} category")
        return selected_questions
        
    except Exception as e:
        print(f"Error generating questions: {e}")
        # Fallback to first 8 questions
        fallback = self._get_fallback_questions(category, count)
        print(f"Using fallback questions: {len(fallback)} questions")
        return fallback


@app.post("/start-interview")
async def start_interview(category: str, user_id: str = "default_user"):
    """Start a new interview session with 8 random questions"""
    try:
        interview_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize interview session
        session_data = {
            "interview_id": interview_id,
            "category": category,
            "user_id": user_id,
            "current_question": 0,
            "questions": [],
            "responses": [],
            "scores": [],
            "start_time": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Generate 8 random questions for the category
        questions = interview_agent.generate_questions(category, 8)  # Changed to 8
        session_data["questions"] = questions
        
        # Store session
        active_interviews[interview_id] = session_data
        
        # Get first question
        first_question = questions[0] if questions else "Tell me about yourself."
        
        print(f"=== INTERVIEW STARTED ===")
        print(f"Interview ID: {interview_id}")
        print(f"Category: {category}")
        print(f"Total Questions: {len(questions)}")
        print(f"Questions: {questions}")
        print("========================")
        
        return {
            "interview_id": interview_id,
            "question": first_question,
            "question_index": 0,
            "total_questions": len(questions),
            "category": category
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting interview: {str(e)}")

# In your main.py, update the submit_response function:

@app.post("/submit-response")
async def submit_response(
    interview_id: str = Form(...),
    response_text: str = Form(...)
):
    """Submit text response and get next question"""
    try:
        print(f"=== PROCESSING RESPONSE ===")
        print(f"Interview ID: {interview_id}")
        print(f"Response length: {len(response_text)} characters")
        print(f"Response preview: {response_text[:100]}...")
        
        if interview_id not in active_interviews:
            raise HTTPException(status_code=404, detail="Interview session not found")
        
        session = active_interviews[interview_id]
        current_q_index = session["current_question"]
        current_question = session["questions"][current_q_index]
        
        print(f"Current question: {current_question}")
        print(f"Category: {session['category']}")
        
        # Store response
        session["responses"].append(response_text)
        
        # Analyze response
        analysis = interview_agent.analyze_response(current_question, response_text, session["category"])
        
        # DEBUG: Print scoring details
        print(f"=== SCORING ANALYSIS ===")
        print(f"Overall Score: {analysis['overall_score']}/10")
        print(f"Relevance: {analysis['relevance_score']}/10")
        print(f"Completeness: {analysis['completeness_score']}/10") 
        print(f"Clarity: {analysis['clarity_score']}/10")
        print(f"Technical Accuracy: {analysis['technical_accuracy']}/10")
        print(f"Word Count: {analysis['word_count']}")
        print(f"Strengths: {analysis['strengths']}")
        print(f"Improvements: {analysis['improvement_areas']}")
        print("========================")
        
        # Store analysis
        session["scores"].append(analysis)
        
        # Move to next question
        session["current_question"] += 1
        next_q_index = session["current_question"]
        
        # Check if interview is complete
        if next_q_index >= len(session["questions"]):
            print("Interview completed! Calculating final score...")
            final_result = interview_agent.calculate_final_score(session["scores"])
            session["status"] = "completed"
            session["final_result"] = final_result
            
            await db_manager.save_interview_session(session)
            
            return {
                "interview_complete": True,
                "final_score": final_result["overall_score"],
                "detailed_feedback": final_result["detailed_feedback"],
                "areas_for_improvement": final_result["areas_for_improvement"],
                "strength_analysis": final_result["strength_analysis"],
                "category_breakdown": final_result.get("category_breakdown", {})
            }
        
        # Get next question
        next_question = session["questions"][next_q_index]
        
        return {
            "interview_complete": False,
            "next_question": next_question,
            "question_index": next_q_index,
            "total_questions": len(session["questions"]),
            "current_response_analysis": analysis
        }
        
    except Exception as e:
        print(f"ERROR in submit_response: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing response: {str(e)}")

@app.post("/speech-to-text")
async def speech_to_text(request: dict):
    """Convert speech audio to text using Whisper"""
    try:
        audio_data = request.get('audio_data')
        if not audio_data:
            raise HTTPException(status_code=400, detail="No audio data provided")
        
        print("Received audio data for transcription")
        
        # Convert base64 back to bytes
        import base64
        audio_bytes = base64.b64decode(audio_data)
        
        # Process speech to text
        transcribed_text = await speech_processor.speech_to_text(audio_bytes)
        
        if transcribed_text:
            print(f"Successfully transcribed: {transcribed_text}")
            return {
                "text": transcribed_text, 
                "success": True,
                "confidence": "high"
            }
        else:
            print("Could not transcribe audio")
            return {
                "text": "", 
                "success": False, 
                "error": "Could not transcribe audio. Please try again or type your answer."
            }
            
    except Exception as e:
        print(f"Error in speech-to-text: {e}")
        return {
            "text": "",
            "success": False,
            "error": f"Error processing speech: {str(e)}"
        }
    
@app.get("/debug-interview/{interview_id}")
async def debug_interview(interview_id: str):
    """Debug endpoint to check interview state"""
    if interview_id not in active_interviews:
        return {"error": "Interview not found in active interviews"}
    
    session = active_interviews[interview_id]
    return {
        "interview_id": interview_id,
        "current_question": session["current_question"],
        "total_questions": len(session["questions"]),
        "questions": session["questions"],
        "responses": session["responses"],
        "status": session["status"],
        "category": session["category"]
    }

@app.get("/interview-status/{interview_id}")
async def get_interview_status(interview_id: str):
    """Get current interview status"""
    if interview_id not in active_interviews:
        raise HTTPException(status_code=404, detail="Interview session not found")
    
    session = active_interviews[interview_id]
    return {
        "interview_id": interview_id,
        "category": session["category"],
        "current_question": session["current_question"],
        "total_questions": len(session["questions"]),
        "status": session["status"]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use Render's assigned port
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
    
