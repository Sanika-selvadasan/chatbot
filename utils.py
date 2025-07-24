from groq_client import chat_with_groq

def user_wants_human_agent(user_message):
    triggers = [
        "talk to a person", "human agent", "real person", "speak to someone",
        "contact support", "talk to a human", "connect me", "contact an employee"
    ]
    return any(phrase in user_message.lower() for phrase in triggers)


def generate_brief_summary(chat_log):
    text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in chat_log if m['role'] in ['user', 'assistant']
    )[:5000]
    prompt = [
        {"role": "system", "content": "Summarize the following conversation briefly and clearly."},
        {"role": "user", "content": text}
    ]
    result = chat_with_groq(prompt).content
    return result
