import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from ml_model.model_predict import predict_fraud
from chatbot.chatbot_engine import generate_chatbot_response

st.set_page_config(page_title="FraudGuard AI Chatbot", page_icon="🕵️")
st.title("FraudGuard AI Chatbot")
st.markdown(
    "Interact with your fraud‑detection model using natural language. "
    "Rule‑based answers for core questions, LLM fallback for everything else."
)

# --------------------------------------------------
# Load / cache the LLM (Flan‑T5)
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading LLM…")
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=128)

llm = load_llm()

# --------------------------------------------------
# Role & Example Prompts
# --------------------------------------------------
role = st.selectbox("Select your role", ["Customer", "Fraud Analyst", "Bank Manager"], index=0)

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
    "Bank Manager": [
        "What % of transactions are risky?",
        "Highlight large fraudulent transactions",
        "Breakdown of riskNote counts",
    ],
}

selected_prompt = st.selectbox("💬 Try a sample question:", [""] + role_examples[role])

user_query = st.text_input(
    "Ask a question about your data:",
    value=selected_prompt if selected_prompt else st.session_state.get("user_query", ""),
    key="user_query",
)

# --------------------------------------------------
# File upload & prediction
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload a transaction CSV", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    df_pred = predict_fraud(df_raw)

    # Risky view
    risky_df = df_pred[(df_pred["prediction"] == 1) | (df_pred["riskNote"] != "Clean")]
    st.session_state["df_raw"] = df_raw
    st.session_state["df_pred"] = df_pred
    st.session_state["risky_df"] = risky_df

# --------------------------------------------------
# Helper: rule‑based query engine
# --------------------------------------------------
def answer_query(query: str, df_pred: pd.DataFrame, risky_df: pd.DataFrame):
    q = query.lower()

    # 1) Bank‑manager: % risky
    if "% of transactions" in q and "risky" in q:
        pct = round(len(risky_df) / len(df_pred) * 100, 2)
        return f"{pct}% of uploaded transactions are risky.", risky_df

    # 2) Bank‑manager: breakdown counts
    if "breakdown" in q and "risknote" in q:
        counts = risky_df["riskNote"].value_counts().to_dict()
        text = "Breakdown of riskNote counts: " + ", ".join([f"{k}: {v}" for k, v in counts.items()])
        return text, None

    # 3) Fraud‑analyst: show all fraud
    if "show all" in q and "fraudulent" in q:
        fraud_df = risky_df[risky_df["prediction"] == 1]
        return "Showing all transactions flagged as fraudulent.", fraud_df

    # 4) Fraud‑analyst: transfers to suspicious dest
    if "transfers to suspicious" in q:
        suspicious_df = risky_df[(risky_df["type"] == "TRANSFER") & (risky_df["riskNote"] == "High-risk destination")]
        return f"Found {len(suspicious_df)} suspicious transfers.", suspicious_df

    # 5) Fraud‑analyst: receivers > 5 risky
    if "receivers" in q and "> 5" in q:
        counts = risky_df["nameDest"].value_counts()
        high_df = risky_df[risky_df["nameDest"].isin(counts[counts > 5].index)]
        return f"Receivers with more than 5 risky transfers: {high_df['nameDest'].nunique()}", high_df

    # 6) Customer: risky last week (assumes step == hour)
    if "risky transactions" in q and "last week" in q:
        max_step = df_pred["step"].max()
        last_week_df = risky_df[risky_df["step"] >= max_step - 24 * 7]
        return f"You had {len(last_week_df)} risky transactions in the last week.", last_week_df

    # Otherwise fall back to LLM
    return None, None

# --------------------------------------------------
# Main Q&A block
# --------------------------------------------------
if user_query and "df_pred" in st.session_state:
    df_pred = st.session_state["df_pred"]
    risky_df = st.session_state["risky_df"]

    text_answer, df_answer = answer_query(user_query, df_pred, risky_df)

    if text_answer:  # rule‑based answer works
        st.markdown(f"**Chatbot:** {text_answer}")
        if df_answer is not None and not df_answer.empty:
            st.subheader("Query Result")
            st.dataframe(df_answer, use_container_width=True)
            csv_bytes = df_answer.to_csv(index=False).encode("utf-8")
            st.download_button("Download This View", csv_bytes, "query_result.csv", "text/csv")
    else:
        # Fallback to LLM with cleaner context
        total_risky = len(risky_df)
        fraud_cnt = int(risky_df["prediction"].sum())
        context = (
            f"You are an AI assistant for a {role}.\n"
            f"Dataset contains {len(df_pred)} transactions; {total_risky} are risky, {fraud_cnt} predicted as fraud.\n"
            f"Question: {user_query}\n"
            f"Answer clearly in 2–3 sentences."
        )
        with st.spinner("Thinking…"):
            try:
                llm_response = llm(context)[0]["generated_text"].split("\n")[-1].strip()
            except Exception as e:
                llm_response = f"[LLM error: {e}]"
        st.markdown(f"**Chatbot:** {llm_response}")

# Add chart once dataset is known
if "risky_df" in st.session_state:
    st.subheader("Risk Distribution")
    fig = px.histogram(
        st.session_state["risky_df"],
        x="riskNote",
        color="riskNote",
        title="Distribution of Risk Categories",
        labels={"count": "Transaction Count"},
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Display default risky table
# --------------------------------------------------
if "risky_df" in st.session_state:
    st.subheader("All Risky or Fraudulent Transactions")
    st.dataframe(st.session_state["risky_df"], use_container_width=True)

