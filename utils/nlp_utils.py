


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




import google.generativeai as genai
import os
from dotenv import load_dotenv
import json

# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-1.5-flash")


def parse_input_to_features_or_chat(user_input: str):
    """
    First, check if the input includes enough data to extract features.
    If yes: return {"features": {...}}
    If no: return {"chat": "some intelligent Gemini response"}
    """

    # Step 1: Check if it's related to crop/farming
    relevance_prompt = f"""
    Is the following message a crop or farming related question? 

    Message: "{user_input}"

    Respond with only one word: "yes" or "no"
    """

    relevance_response = model.generate_content(relevance_prompt)
    relevance_answer = relevance_response.text.strip().lower()

    if relevance_answer.startswith("yes"):
        # Try to extract features (Step 2)
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

        try:
            response = model.generate_content(extract_prompt)
            content = response.candidates[0].content.parts[0].text.strip()

            if content.startswith("```") and content.endswith("```"):
                content = content.strip("```").strip()

            features = json.loads(content)
            return {"features": features}

        except Exception:
            # Step 3: fallback chat mode
            chat_response = model.generate_content(user_input)
            return {"chat": chat_response.text.strip()}

    else:
        # Not crop-related → general chat
        chat_response = model.generate_content(user_input)
        return {"chat": chat_response.text.strip()}
