import os
import requests
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")


groq_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
)

def chat_with_groq(messages: list):
    lc_messages = []
    for m in messages:
        if m["role"] == "system":
            lc_messages.append(SystemMessage(content=m["content"]))
        elif m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))

    print("[INFO] Invoking Groq LLM...")
    response = groq_llm.invoke(lc_messages)
    print("[DEBUG] Groq response received.")
    return response

def get_confidence_score(response_text: str) -> float:
    """
    Extract confidence score from LLM response.
    Expected format: [CONFIDENCE: 0.85]
    """
    try:
        # Look for confidence pattern
        confidence_pattern = r'\[CONFIDENCE:\s*([0-9]*\.?[0-9]+)\]'
        match = re.search(confidence_pattern, response_text)
        
        if match:
            confidence = float(match.group(1))
            # Ensure confidence is between 0 and 1
            return max(0.0, min(1.0, confidence))
        else:
            # If no confidence score found, analyze response content
            low_confidence_indicators = [
                "i'm not sure", "i don't know", "i can't", "unfortunately",
                "i don't have", "i'm unable", "i cannot", "i'm sorry but",
                "i don't have enough information", "connect you to a human agent",
                "escalate", "human agent", "not enough information"
            ]
            
            response_lower = response_text.lower()
            if any(indicator in response_lower for indicator in low_confidence_indicators):
                return 0.1  # Very low confidence - should escalate
            elif len(response_text.strip()) < 30:
                return 0.3  # Low confidence for very short responses
            elif len(response_text.strip()) < 100:
                return 0.5  # Medium confidence for short responses
            else:
                return 0.7  # Default confidence for substantial responses
                
    except (ValueError, TypeError):
        return 0.3  # Low confidence on error