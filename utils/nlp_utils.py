import os
import json
import signal
import requests
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI
from textblob import TextBlob

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HUGGINGFACE_API_KEY")

# Configure APIs
genai.configure(api_key=gemini_api_key)
client = OpenAI()  # OpenAI reads OPENAI_API_KEY from env

model = genai.GenerativeModel("gemini-1.5-flash")


# 1. TextBlob Fallback
def local_fallback_nlp(user_input):
    blob = TextBlob(user_input)
    keywords = blob.noun_phrases
    return f"I'm offline now 🌾 Here's what I understood: {', '.join(keywords)}"


# 2. OpenAI Fallback
def openai_fallback(user_input):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a friendly crop advisor."},
                {"role": "user", "content": user_input},
            ],
            temperature=0.8,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI fallback error:", str(e))
        return huggingface_fallback(user_input)


# 3. Hugging Face Fallback (free)
def huggingface_fallback(user_input):
    try:
        api_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {
            "inputs": f"Answer this user question in a friendly and short way: {user_input}",
            "parameters": {
                "max_new_tokens": 100,
                "temperature": 0.7
            }
        }

        response = requests.post(api_url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()

        output = response.json()
        if isinstance(output, dict) and "error" in output:
            return f"🤖 Hugging Face error: {output['error']}"
        elif isinstance(output, list) and "generated_text" in output[0]:
            return output[0]["generated_text"].strip()
        elif isinstance(output, list) and "generated_text" in output[0]:
            return output[0]["generated_text"].strip()
        elif isinstance(output, list) and len(output) > 0:
            return output[0]["generated_text"].strip()
        return "🤖 Hugging Face gave no response. Try again?"
    except Exception as e:
        print("Hugging Face fallback error:", str(e))
        return local_fallback_nlp(user_input)


# Safe timeout wrapper
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


# 🧠 Main logic
def parse_input_to_features_or_chat(user_input: str):
    try:
        # Step 1: Relevance check
        def call_relevance():
            prompt = f"""
            Is the following message a crop or farming related question? 
            Message: "{user_input}"
            Respond with only one word: "yes" or "no"
            """
            return model.generate_content(prompt)

        try:
            relevance_response = run_with_timeout(call_relevance, timeout=5)
        except TimeoutException:
            return {
                "chat": "🌧️ Gemini is slow. Switching to OpenAI...",
                "fallback": openai_fallback(user_input),
            }

        relevance_answer = relevance_response.text.strip().lower()

        if relevance_answer.startswith("yes"):
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
                - Example:
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

            except TimeoutException:
                return {
                    "chat": "🌧️ Gemini is slow during extraction. Switching to OpenAI...",
                    "fallback": openai_fallback(user_input),
                }

        else:
            def call_chat():
                return model.generate_content(user_input)

            try:
                chat_response = run_with_timeout(call_chat, timeout=5)
                return {"chat": chat_response.text.strip()}
            except TimeoutException:
                return {
                    "chat": "🌧️ Gemini chat timed out. Switching to OpenAI...",
                    "fallback": openai_fallback(user_input),
                }

    except Exception as e:
        error_msg = str(e).lower()
        if "429" in error_msg or "quota" in error_msg:
            return {
                "chat": "⚠️ Gemini quota exceeded. Using OpenAI...",
                "fallback": openai_fallback(user_input),
            }
        else:
            return {
                "chat": "⚠️ Gemini failed. Using OpenAI...",
                "fallback": openai_fallback(user_input),
            }





# import google.generativeai as genai
# import os
# from dotenv import load_dotenv
# import json

# # Load API key from .env
# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")

# # Configure Gemini
# genai.configure(api_key=api_key)

# # Initialize the model (you can use "gemini-pro" or "gemini-1.5-flash")
# model = genai.GenerativeModel("gemini-1.5-flash")


# def parse_input_to_features(user_input: str):
#     prompt = f"""
#     Convert the following user request into structured features for crop recommendation.

#     Text: "{user_input}"

#     Return ONLY a valid JSON object with these fields:
#     N, P, K, temperature, humidity, ph, rainfall

#     Rules:
#     - Only include the JSON object — no markdown, no explanations.
#     - Each value must be a number.
#     - Do not use quotes around keys unless necessary.
#     - Example valid format:
#     {{"N": 90, "P": 42, "K": 43, "temperature": 20.5, "humidity": 80.0, "ph": 6.8, "rainfall": 120}}

#     Ranges:
#     - N, P, K: 0–140
#     - temperature: 10–50 (°C)
#     - humidity: 10–100 (%)
#     - ph: 3.5–9.5
#     - rainfall: 0–300 (mm)

#     Do NOT wrap the output in backticks or markdown.
#     """

#     response = model.generate_content(prompt)

#     try:
#         # Extract raw text
#         content = response.candidates[0].content.parts[0].text.strip()
#         print("Raw Gemini Output:\n", content)

#         # Clean up possible markdown code block
#         if content.startswith("```") and content.endswith("```"):
#             content = content.strip("```").strip()

#         return json.loads(content)
#     except Exception as e:
#         raise ValueError(f"Gemini output is not valid JSON:\n{content}\nError: {e}")



