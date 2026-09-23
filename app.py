import os
import requests

from flask import Flask, render_template, request, jsonify, session

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent


# ============================================
# 1. LOAD API KEYS
# ============================================

load_dotenv()

open_router_api_key = os.getenv("open_router_api_key")
tavily_api_key = os.getenv("tavily_api_key")
openweathermap_api_key = os.getenv("openweathermap_api_key")


# ============================================
# 2. FLASK APP
# ============================================

app = Flask(__name__)

app.secret_key = os.getenv(
    "flask_secret_key",
    "change-this-secret-key"
)


# ============================================
# 3. TAVILY SEARCH TOOL
# ============================================

search_tool = TavilySearchResults(
    max_results=2,
    api_key=tavily_api_key
)


# ============================================
# 4. WEATHER TOOL
# ============================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather of a city."""

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}"
        f"&appid={openweathermap_api_key}"
        "&units=metric"
    )

    try:

        response = requests.get(
            url,
            timeout=10
        )

        data = response.json()

        if data.get("cod") != 200:
            return "Could not find weather for this city."

        weather = data["weather"][0]["description"]
        temperature = data["main"]["temp"]

        return f"{city}: {weather}, {temperature}°C"

    except Exception as e:

        return f"Weather error: {str(e)}"


# ============================================
# 5. TOOLS
# ============================================

tools = [
    search_tool,
    get_weather
]


# ============================================
# 6. LLM
# ============================================

llm = ChatOpenAI(
    model="openrouter/free",
    openai_api_key=open_router_api_key,
    openai_api_base="https://openrouter.ai/api/v1"
)


# ============================================
# 7. AGENT PROMPT
# ============================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant.

            You have access to two tools:

            1. Tavily search
               - Use it when the user asks for current or
                 up-to-date information.

            2. Weather tool
               - Use it when the user asks about weather.

            Give clear and useful answers.
            """
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        (
            "human",
            "{input}"
        ),

        MessagesPlaceholder(
            variable_name="agent_scratchpad"
        )
    ]
)


# ============================================
# 8. CREATE AGENT
# ============================================

agent = create_openai_tools_agent(
    llm,
    tools,
    prompt
)


# ============================================
# 9. AGENT EXECUTOR
# ============================================

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)


# ============================================
# 10. HOME PAGE
# ============================================

@app.route("/")
def home():

    if "messages" not in session:
        session["messages"] = []

    return render_template(
        "index.html",
        messages=session["messages"]
    )


# ============================================
# 11. CHAT API
# ============================================

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json()

        user_input = data.get("message", "").strip()

        if not user_input:

            return jsonify({
                "error": "Please enter a message."
            }), 400


        # ====================================
        # CREATE CHAT HISTORY
        # ====================================

        if "messages" not in session:
            session["messages"] = []

        chat_history = []

        for message in session["messages"]:

            if message["role"] == "user":

                chat_history.append(
                    (
                        "human",
                        message["content"]
                    )
                )

            elif message["role"] == "assistant":

                chat_history.append(
                    (
                        "ai",
                        message["content"]
                    )
                )


        # ====================================
        # RUN AGENT
        # ====================================

        response = agent_executor.invoke(
            {
                "input": user_input,
                "chat_history": chat_history
            }
        )

        answer = response["output"]


        # ====================================
        # SAVE CHAT
        # ====================================

        session["messages"].append(
            {
                "role": "user",
                "content": user_input
            }
        )

        session["messages"].append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        session.modified = True


        return jsonify(
            {
                "answer": answer
            }
        )


    except Exception as e:

        return jsonify(
            {
                "error": str(e)
            }
        ), 500


# ============================================
# 12. CLEAR CHAT
# ============================================

@app.route("/clear", methods=["POST"])
def clear_chat():

    session["messages"] = []

    session.modified = True

    return jsonify(
        {
            "success": True
        }
    )


# ============================================
# 13. RUN FLASK
# ============================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )