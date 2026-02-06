from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from tools import check_vitals, write_referral_letter

# 1. Setup the Brain (Llama-3 via LM Studio)
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="medical-llama-3-8b",
    temperature=0,
)

# 2. Define Tools
@tool
def check_vitals_tool(systolic: int, diastolic: int, heart_rate: int) -> str:
    """Analyzes vitals. Use this FIRST. Input: systolic, diastolic, heart_rate."""
    return check_vitals(systolic, diastolic, heart_rate)

@tool
def write_referral_tool(patient_name: str, diagnosis: str, recommendation: str) -> str:
    """Writes a referral letter. Use this ONLY if risk is Critical."""
    return write_referral_letter(patient_name, diagnosis, recommendation)

tools = [check_vitals_tool, write_referral_tool]

# 3. The "ReAct" Prompt (The Secret Sauce)
# This forces the model to think step-by-step in a format we can parse.
template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

# 4. Build the Agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,      # This lets us see the "Thought Process"
    handle_parsing_errors=True
)

# --- RUN THE SIMULATION ---

print("🤖 Dr. Llama ReAct Agent Online...")
print("-----------------------------------")

user_input = """
I have a patient named John Doe. 
His Blood Pressure is 190/110 and Heart Rate is 95. 
Please analyze his vitals and if it is critical, draft a referral letter immediately.
"""

print(f"Patient: {user_input}\n")

try:
    response = agent_executor.invoke({"input": user_input})
    print(f"\n✅ Final Result: {response['output']}")
except Exception as e:
    print(f"Error: {e}")