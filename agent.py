# ✅ CORRECTED AGENT.PY
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, List
import operator
from tools import check_vitals, write_referral_letter

# 1. Define the Agent's "State" (Memory)
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]

# 2. Setup the Brain (Llama-3 via LM Studio)
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="medical-llama-3-8b",
    temperature=0,  # Strict mode
)

# 3. Bind Tools
tools = [check_vitals, write_referral_letter]
llm_with_tools = llm.bind_tools(tools)

# 4. Define the Nodes
def reasoner(state: AgentState):
    """The Brain: Decides what to do next."""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    """The Logic: Did the AI ask to use a tool?"""
    last_message = state['messages'][-1]
    
    if last_message.tool_calls:
        print(f"   ⚡ AI decided to use tool: {last_message.tool_calls[0]['name']}")
        return "tools"
    return END

# 5. Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", reasoner)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")
app = workflow.compile()

# --- RUN THE SIMULATION ---

print("🤖 Dr. Llama Triage Agent Online...")
print("-----------------------------------")

user_input = """
I have a patient named John Doe. 
His Blood Pressure is 190/110 and Heart Rate is 95. 
Please analyze his vitals and if it is critical, draft a referral letter immediately.
"""

# ✅ THE FIX: A System Prompt that forces behavior
system_prompt = """
You are an autonomous Medical Triage Agent.
You have access to tools: 'check_vitals' and 'write_referral_letter'.

RULES:
1. NEVER answer a vitals question with your own knowledge. You MUST call 'check_vitals'.
2. If 'check_vitals' returns "Critical", you MUST call 'write_referral_letter'.
3. Do not ask for clarification. ACT immediately.
"""

inputs = {"messages": [
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_input)
]}

# Run the graph
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"🔹 Node '{key}' finished.")
        # Debug: Print what the AI actually said
        if key == "agent":
            msg = value["messages"][-1]
            if not msg.tool_calls:
                print(f"   (AI Thought: {msg.content})")

print("\n-----------------------------------")
print("✅ Workflow Complete.")