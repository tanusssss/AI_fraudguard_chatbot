"""
chatbot_engine.py
Primary: Free Hugging Face LLM (local / HF endpoint)
Fallback: Rule‑based responses
Optional: Easily swap in OpenAI by setting USE_OPENAI=True
"""

import os
from chatbot.prompt_templates import prompt_templates


# CONFIGURATION FLAGS  (adjust in one place only)

USE_HF_LLM   = True          # <- default: Hugging Face Transformers for replies
USE_OPENAI   = False         # <- set True if you later add OpenAI key & want to use GPT
HF_MODEL_ID  = os.getenv("HF_MODEL_ID", "microsoft/DialoGPT-small")  # free, light model
MAX_TOKENS   = 128


# OPTIONAL: set your Hugging Face token here (needed for Inference API;
# not needed if you’re just downloading the small model locally)
# os.environ["HF_TOKEN"] = "hf_..."



# Try to load the preferred LLM.  If it fails, silently fall back.

generator = None
if USE_HF_LLM:
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        generator = pipeline(
            "text-generation",
            model=HF_MODEL_ID,
            max_new_tokens=MAX_TOKENS,
            do_sample=True,
            temperature=0.3,
        )
    except Exception as e:
        print(f"[WARN] Hugging Face model load failed: {e}")
        generator = None
        USE_HF_LLM = False

if USE_OPENAI:
    try:
        import openai, dotenv
        dotenv.load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
    except Exception as e:
        print(f"[WARN] OpenAI init failed: {e}")
        USE_OPENAI = False


# ------------------------------------------------------------------
# Helper: rule‑based fallback
# ------------------------------------------------------------------
def _rule_based_response(role, user_query, txn_data):
    risk = txn_data["riskNote"]
    summary = (
        f"Amount ₹{txn_data['amount']}, type {txn_data['type']}, "
        f"sender {txn_data['nameOrig']} → receiver {txn_data['nameDest']} "
        f"({risk})."
    )
    match risk:
        case "Fraudulent":
            return f"⚠️ This transaction is FRAUDULENT. {summary}"
        case "High-risk destination":
            return f"🚩 High‑risk destination detected. {summary}"
        case "Shared-risk destination":
            return f"🟡 Shared‑risk destination. {summary}"
        case _:
            return f"✅ Looks clean. {summary}"


# ------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------
def generate_chatbot_response(role: str, user_query: str, txn_data: dict) -> str:
    """
    role        : 'customer', 'fraud analyst', or 'manager'
    user_query  : natural‑language question
    txn_data    : dict / Series containing at least ['amount', 'type', 'nameOrig', 'nameDest', 'riskNote']
    """
    # Always keep a minimal summary handy for prompt or fallback
    txn_summary = (
        f"Amount: ₹{txn_data['amount']}\n"
        f"Type: {txn_data['type']}\n"
        f"Sender: {txn_data['nameOrig']}\n"
        f"Receiver: {txn_data['nameDest']}\n"
        f"Risk Level: {txn_data['riskNote']}"
    )

    # 1️⃣  Hugging Face LLM (primary)
    if USE_HF_LLM and generator is not None:
        template = prompt_templates.get(role.lower(), prompt_templates["customer"])
        full_prompt = template.format(user_query=user_query, txn_summary=txn_summary)
        try:
            raw = generator(full_prompt)[0]["generated_text"]
            # strip the prompt part from generated text if duplicated
            response = raw[len(full_prompt):].strip()
            return response or _rule_based_response(role, user_query, txn_data)
        except Exception as e:
            print(f"[WARN] HF generation failed: {e}")

    # 2️⃣  OpenAI GPT (optional plug‑in)
    if USE_OPENAI:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a fraud‑analysis assistant."},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.3,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"[WARN] OpenAI call failed: {e}")

    # 3️⃣  Fallback: rule‑based
    return _rule_based_response(role, user_query, txn_data)
