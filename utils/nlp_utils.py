import google.generativeai as genai
import os
import signal  # Added to fix the 'signal' not defined error
from dotenv import load_dotenv
import json

# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-1.5-flash")

# Timeout wrapper
class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

def run_with_timeout(func, timeout=5):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        result = func()
        signal.alarm(0)
        return result
    except TimeoutException:
        raise TimeoutException("Function timed out")
    finally:
        signal.alarm(0)

import google.generativeai as genai
import os
import threading
from dotenv import load_dotenv
import json

# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-1.5-flash")

# Timeout wrapper using threading
class TimeoutException(Exception):
    pass

def run_with_timeout(func, timeout=5):
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        raise TimeoutException("Function timed out")
    if exception[0]:
        raise exception[0]
    return result[0]

def parse_input_to_features_or_chat(user_input: str):
    """
    First, check if the input includes enough data to extract features.
    If yes: return {"features": {...}}
    If no: return {"chat": "some intelligent Gemini response"}
    """

    # Step 1: Check if it's related to crop/farming
    def call_relevance():
        relevance_prompt = f"""
        Is the following message a crop or farming related question? 
        Message: "{user_input}"
        Respond with only one word: "yes" or "no"
        """
        return model.generate_content(relevance_prompt)

    try:
        relevance_response = run_with_timeout(call_relevance, timeout=5)
        relevance_answer = relevance_response.text.strip().lower()
    except TimeoutException:
        return {"chat": "🌧️ Gemini is slow. Please try again later."}

    if relevance_answer.startswith("yes"):
        # Try to extract features (Step 2)
        def call_extract():
            extract_prompt = f"""
            Convert the following user request into structured features for crop recommendation.
            Text: "{user_input}"
            Return ONLY a valid JSON object with these fields:
            N, P, K, temperature, humidity, ph, rainfall
            Rules:
            - Only include the JSON object — no markdown, no explanations.
            - Each value must be a number.
            - If input is vague (e.g., "moderate weather"), use your best estimate.
            - Do not use quotes around keys unless necessary.
            - Example valid format:
            {{"N": 90, "P": 42, "K": 43, "temperature": 20.5, "humidity": 80.0, "ph": 6.8, "rainfall": 120}}
            """
            return model.generate_content(extract_prompt)

        try:
            response = run_with_timeout(call_extract, timeout=5)
            content = response.candidates[0].content.parts[0].text.strip()

            if content.startswith("```") and content.endswith("```"):
                content = content.strip("```").strip()

            features = json.loads(content)
            return {"features": features}
        except (TimeoutException, json.JSONDecodeError):
            chat_response = model.generate_content(user_input)
            return {"chat": chat_response.text.strip()}

    else:
        # Not crop-related → general chat
        def call_chat():
            return model.generate_content(user_input)

        try:
            chat_response = run_with_timeout(call_chat, timeout=5)
            return {"chat": chat_response.text.strip()}
        except TimeoutException:
            return {"chat": "🌧️ Gemini chat timed out. Please try again later."}