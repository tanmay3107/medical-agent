from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
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
    temperature=0  # Temperature 0 is crucial for tools!
)

# 3. Bind Tools to the Brain
tools = [check_vitals, write_referral_letter]
llm_with_tools = llm.bind_tools(tools)

# 4. Define the Nodes (The "Thinking" Steps)

def reasoner(state: AgentState):
    """The Brain: Decides what to do next based on messages."""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    """The Logic: Did the AI ask to use a tool?"""
    last_message = state['messages'][-1]
    
    # If the AI wants to use a tool, go to "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, stop
    return END

# 5. Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", reasoner)
workflow.add_node("tools", ToolNode(tools)) # Helper to run our python functions

workflow.set_entry_point("agent")

# Conditional Edge: Agent -> Tools OR End
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

# Edge: Tools -> Agent (Loop back to let AI see the tool result)
workflow.add_edge("tools", "agent")

# Compile
app = workflow.compile()

# --- RUN THE SIMULATION ---

print("🤖 Dr. Llama Triage Agent Online...")
print("-----------------------------------")

# Scenario: A critical patient
user_input = """
I have a patient named John Doe. 
His Blood Pressure is 190/110 and Heart Rate is 95. 
Please analyze his vitals and if it is critical, draft a referral letter immediately.
"""

print(f"Patient Input: {user_input}\n")

inputs = {"messages": [HumanMessage(content=user_input)]}

# Run the graph and print steps
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"🔹 Node '{key}' finished.")
        
        # Add this to see what the AI actually said!
        if key == "agent":
            msg = value["messages"][-1]
            print(f"   AI Message: {msg.content}")
            print(f"   Tool Calls: {msg.tool_calls}")

print("\n-----------------------------------")