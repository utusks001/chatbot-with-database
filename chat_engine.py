# chat_engine.py
from langchain.chat_models import ChatOpenAI, ChatGroq, ChatGooglePalm
from langchain.schema import AIMessage, HumanMessage

def get_llm(model_name="openai", temperature=0.3, top_p=1.0, max_tokens=512):
    if model_name == "groq":
        return ChatGroq(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    elif model_name == "gemini":
        return ChatGooglePalm(temperature=temperature, top_p=top_p, max_output_tokens=max_tokens)
    else:
        return ChatOpenAI(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

def run_chat(llm, history, user_input):
    history.append(HumanMessage(content=user_input))
    response = llm(history)
    history.append(response)
    return response, history
