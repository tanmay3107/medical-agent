from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

from tools import check_vitals, write_referral_letter

# --- LLM (LM Studio) ---
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="medical-llama-3-8b",
    temperature=0,
)

# --- Tools ---
@tool
def check_vitals_tool(systolic: int, diastolic: int, heart_rate: int) -> str:
    """Analyze vitals and determine risk level."""
    return check_vitals(systolic, diastolic, heart_rate)

@tool
def write_referral_tool(patient_name: str, diagnosis: str, recommendation: str) -> str:
    """Write a referral letter if risk is critical."""
    return write_referral_letter(patient_name, diagnosis, recommendation)

tools = [check_vitals_tool, write_referral_tool]

# --- Create Agent (NEW API) ---
agent = create_agent(
    model=llm,   # 🔴 THIS is what your error was about
    tools=tools,
)

print("🤖 Dr. Llama Medical Agent Online")
print("--------------------------------")

user_input = """
Patient Name: John Doe
Blood Pressure: 190/110
Heart Rate: 95

Analyze vitals and if the condition is critical, generate a referral letter.
"""

result = agent.invoke(
    {"messages": [("human", user_input)]}
)

print("\n✅ Final Output:")
print(result["messages"][-1].content)
