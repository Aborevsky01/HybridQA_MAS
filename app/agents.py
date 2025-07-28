from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from pydantic import BaseModel, Field
import pandas as pd
import json
from .utils import get_retriever_tool
from typing_extensions import List, Optional, TypedDict, List, Dict, Any, Union, Annotated
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Plan(BaseModel):
    context_information_summary: str
    thoughts: str
    task: str
    selected_tool: str

    '''
    subtasks: List[str] = Field(
        description="Ordered list of steps to solve the problem",
        examples=[["Step 1: Find X", "Step 2: Calculate Y"]]
    )
    '''

class AgentState(TypedDict):
    input: str
    table_data: dict
    subtasks: List[Dict[str, Any]]
    current_index: int
    tool_call: Optional[List[Dict[str, Any]]]
    final_answer: Optional[dict]
    trace: List[Dict[str, Any]]


llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert ReAct style planner. You are given user's query, table's headers and previous steps taken if any.
     You should select either to give next task for a table-tool agent or finish exploring the tables with analysis. 
     If you decide to add exploring the table, pass 'analysis' to json.
     You can select one of the following tools: "analysis", "table".

     Always answer in the following JSON format:

     {{
        "context_information_summary": <summary of available context information>,
        "thoughts": <your thoughts about what to do in a next step>,
        "task": <task description for the next step in natural language, including relevant input values>,
        "selected_tool": <the name of the selected tool for the next step. Use one of the available tools>,
    }}
     
    For selecting the next tool or providing a final answer, always use context information. Almost each cell in table is a dict with "value" and "summary" keys.
    "value" contains the value of the cell and "summary" may suggest additional information about this topic.

    The generated task should be a single task. If you did not get response to the previous context, try changing the formulation of subtask, preserving the idea.
    Avoid combining multiple tasks in one step. If you need to perform multiple tasks, do them one after the other.
    Avoid requesting more than one piece of information in a single task. If you need multiple pieces of information request them one after the other."""),
    ("human", "Table Columns: {headers}\n\nUser Question: {input}\n\nPrevious steps: {context}")
])

#      Break down the user question into 1-3 precise subtasks that can be executed using ONLY the provided table.
#      Each task must be self-contained and executable consequently. Output must be JSON with "subtasks" list."""),

class QueryInfo(BaseModel):
    final_answer: str = Field(description="Here should be given final precise answer to the query")

parser = JsonOutputParser(pydantic_object=QueryInfo)
format_instructions = parser.get_format_instructions()

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an analysis expert. Synthesize information from these completed tasks to answer the original input query:

     Completed Tasks:
     {task_results}

     Your answer should be factual and accurate. Respond directly to each of the questions being asked.
     {format_instructions}
    """),
    ("human", "Original Question: {input}")
])

async def plan_node(state: AgentState) -> dict:
    """Generate execution plan based on table structure"""
    human_msg = {
        "headers": state['table_data']['header'],
        "input": state['input'],
        "context": [] if state['subtasks'] == [] else state['subtasks'][-1]
    }
    planner = PLANNER_PROMPT | llm.with_structured_output(Plan)
    plan = await planner.ainvoke(human_msg)
    logger.info(f'Node: Plan.')
    return {
        "subtasks": state['subtasks'] + [{"context_information_summary" : plan.context_information_summary,
                      "task": plan.task, "result": None, "type": plan.selected_tool,
                      "thoughts" : plan.thoughts}], #for t in plan.subtasks[0:1]],
        "current_index": state['current_index'] + 1, #0
        "trace": state["trace"] + [{"node": "plan", "output": plan.model_dump()}]
    }


async def table_node(state: AgentState) -> dict:
    """Execute table-based subtask using pandas agent"""
    current_idx  = state["current_index"]
    current_task = state["subtasks"][current_idx]['task']
    df = state['table_data']['data']

    agent = create_pandas_dataframe_agent(
        llm, df, agent_type='tool-calling', allow_dangerous_code=True,
        verbose=False, agent_executor_kwargs={"handle_parsing_errors": True}, 
        max_iterations=10,  
    )
    logger.info(f'Node: Table {current_idx+1}. Task: {current_task}')

    human_msg = f"Current task: {current_task}"
    if current_idx > 0 :
      human_msg += f'Completed tasks: {state["subtasks"][-1]}\n\n'
    result = await agent.ainvoke(human_msg)
    new_subtasks = state["subtasks"].copy()
    new_subtasks[current_idx]["result"] = result['output']
    return {
        "subtasks": new_subtasks,
        "current_index": current_idx,
        "trace": state["trace"] + [{
            "node": "table", 
            "task": current_task,
            "result": result['output'][:500] + "..." if len(result['output']) > 500 else result['output']
        }]
    }

async def analysis_node(state: AgentState) -> dict:
    """Analyze results and decide if retrieval is needed"""
    task_results = "\n".join(f"- {t['task']}: {t['result']}" for t in state["subtasks"])
    retriever_tool = get_retriever_tool(
        state['table_data']['table_id'],
        state['table_data']['data']
    )
    tools = [retriever_tool] if retriever_tool and state['tool_call'] == [] else []
    if tools != []:
        final_prompt = ANALYSIS_PROMPT + """
        If information is insufficient at least for one question, use retrieval tool to search Wikipedia passages. Prepare a detailed query for the tool, so the retriever will find perfect matching to the answer.
        Otherwise, provide final answer in JSON format with keys: "final_answer".
        """
    else:
        final_prompt = ANALYSIS_PROMPT + """Provide final answer in JSON format with keys: "final_answer"."""

    agent = final_prompt | llm.bind_tools(tools)
    response = await agent.ainvoke({"input": state["input"], "task_results": task_results, 'format_instructions' : format_instructions})
    
    logger.info(f'Node: Analysis.')
    if hasattr(response, "tool_calls") and response.tool_calls and state['tool_call'] == []:
        return {
            "tool_call": [response],
            "trace": state["trace"] + [{"node": "analysis", "action": "tool_call"}]
        }
    try:
        final_json = parser.parse(response.content)
    except Exception as e:
        print(e)
        final_json = {
            "final_answer": response.content
        }
    return {
        "subtasks": state["subtasks"],
        "final_answer": final_json,
        "trace": state["trace"] + [{"node": "analysis", "action": "final_response"}]
    }

async def retrieval_node(state: AgentState) -> dict:
    """Execute retrieval tool and store results"""
    tool_calls = [t["args"]["query"] for t in state["tool_call"][-1].tool_calls]
    retriever_tool = get_retriever_tool(
        state['table_data']['table_id'],
        state['table_data']['data']
    )
    tasks = [retriever_tool.ainvoke(tool_call) for tool_call in tool_calls]
    logger.info(f'Node: Retriever. Query: {tool_calls}')
    result = await asyncio.gather(*tasks) if retriever_tool else "No retriever available"
    retrieval_task = {
        "task": f"Retrieve: {tool_calls}",
        "result": result,
        "type": "retrieval"
    }
    return {
        "subtasks": state["subtasks"] + [retrieval_task],
        "trace": state["trace"] + [{"node": "retrieval", "query": tool_calls}]
    }

def create_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("plan", plan_node)
    workflow.add_node("table", table_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("retrieval", retrieval_node)
    
    workflow.set_entry_point("plan")
    workflow.add_conditional_edges(
        "plan",
        lambda state: "table" if state['subtasks'][-1]['type'] == 'table' else "analysis"
    )
    workflow.add_conditional_edges(
        "table",
        lambda state: "analysis" if state["current_index"] >= 3 else "plan"
    )
    workflow.add_edge("retrieval", "analysis")
    workflow.add_conditional_edges(
        "analysis",
        lambda state: "retrieval" if state.get("tool_call", "") and state["final_answer"] is None else END
    )
    
    return workflow.compile()

workflow_app = create_workflow()