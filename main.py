import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from ml_model.model_predict import predict_fraud
from chatbot.chatbot_engine import generate_chatbot_response

st.set_page_config(page_title="FraudGuard AI Chatbot", page_icon="🕵️", layout="centered")

st.title(" FraudGuard AI Chatbot")
st.markdown("Talk to your ML fraud‑detection model in natural language – now powered by a free Hugging Face LLM (Flan‑T5‑base).")

# --------------------------------------------------
#   Load / cache LLM (HF free model)
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading LLM (first time ≈15‑20 s)…")
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=128,
    )

llm = load_llm()

# --------------------------------------------------
# Role & Prompt Helpers
# --------------------------------------------------
role = st.selectbox("Select your role", ["Customer", "Fraud Analyst", "Manager"], index=0)

role_examples = {
    "Customer": [
        "Was my ₹5000 transfer safe?",
        "Show my risky transactions in the last week",
        "Is the receiver C98765 suspicious?",
    ],
    "Fraud Analyst": [
        "Show all transactions flagged as fraudulent",
        "List transfers to suspicious destinations",
        "Which receivers have > 5 risky transfers?",
    ],
    "Manager": [
        "What % of transactions are risky?",
        "Highlight large fraudulent transactions",
        "Breakdown of riskNote counts",
    ],
}

selected_prompt = st.selectbox(" Try a sample question:", [""] + role_examples[role])

# Preserve user input but allow click‑to‑ask templates
def _get_user_query():
    if selected_prompt:
        return selected_prompt
    return st.session_state.get("user_query", "")

user_query = st.text_input(
    "Ask a question about your data:",
    value=_get_user_query(),
    key="user_query",
)

# --------------------------------------------------
# File uploader & prediction
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload a transaction CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    processed_df = predict_fraud(df)

    # Filter only frauds or risky cases
    filtered_df = processed_df[(processed_df["prediction"] == 1) | (processed_df["riskNote"] != "Clean")]
    st.session_state["filtered_df"] = filtered_df

# --------------------------------------------------
# Display results & visualisation
# --------------------------------------------------
if "filtered_df" in st.session_state:
    st.subheader("All Risky or Fraudulent Transactions:")
    st.dataframe(st.session_state["filtered_df"], use_container_width=True)

    csv_bytes = st.session_state["filtered_df"].to_csv(index=False).encode("utf-8")
    st.download_button(" Download Filtered Results", csv_bytes, "risky_transactions.csv", "text/csv")

    # Risk distribution chart
    if not st.session_state["filtered_df"].empty:
        st.subheader(" Risk Distribution")
        fig = px.histogram(
            st.session_state["filtered_df"],
            x="riskNote",
            color="riskNote",
            title="Distribution of Risk Categories",
            labels={"count": "Transaction Count"},
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
#  LLM‑Powered Q&A Section
# --------------------------------------------------
if user_query:
    if "filtered_df" not in st.session_state:
        st.warning(" Please upload a CSV first so I have data to talk about.")
    else:
        df_ctx = st.session_state["filtered_df"]
        # Build a compact context summary for the LLM
        total_risky = len(df_ctx)
        fraud_count = int(df_ctx["prediction"].sum())
        top_types = (
            df_ctx["type"].value_counts().head(3).to_dict()
        )
        ctx_summary = (
            f"You are an AI assistant for {role}. "
            f"There are {total_risky} risky transactions (of which {fraud_count} are predicted fraud). "
            f"Top risky transaction types: {top_types}.\n"
        )

        prompt = (
            ctx_summary
            + "Here is the user's question:\n"
            + user_query
            + "\nAnswer in 2–3 sentences, role‑appropriate, referencing the data."  # instruction to model
        )

        with st.spinner("Thinking…"):
            try:
                llm_response = llm(prompt)[0]["generated_text"].split("\n")[-1].strip()
            except Exception as e:
                llm_response = f"[LLM error: {e}]"

        st.markdown(f"** Chatbot:** {llm_response}")
