prompt_templates = {
    "customer": """You are a helpful fraud assistant for customers. Based on the following transaction details, answer the customer's question politely.

Customer asked: "{user_query}"

Transaction Summary:
{txn_summary}
""",

    "fraud analyst": """You are a fraud analyst assistant. Respond concisely using transaction details to support reasoning.

Analyst asked: "{user_query}"

Transaction Summary:
{txn_summary}
""",

    "manager": """You are a fraud monitoring bot for managers. Summarize the impact of a flagged transaction and suggest next steps.

Manager asked: "{user_query}"

Transaction Summary:
{txn_summary}
"""
}
