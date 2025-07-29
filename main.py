import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import uuid
from app.agents import workflow_app, AgentState
from app.utils import get_retriever_local
from typing import List, Optional, TypedDict, List, Dict, Any, Union, Annotated
import logging
import os
import uvicorn
import numpy as np
import json


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HybridQA Multi Agent System API",
    description="API for answering questions using both tabular data and Wikipedia passages",
)

class SubtaskResult(BaseModel):
    task: str
    result: Any
    type: str

class FinalResponse(BaseModel):
    final_answer: str
    subtasks: List[SubtaskResult]
    trace: List[Dict[str, Any]]

@app.post("/query", response_model=FinalResponse, tags=["QA"])
async def answer_question(
    table_file: UploadFile = File(..., description="JSON file containing table data")):
    """Answer questions using tabular data and Wikipedia passages"""
    try:
        
        contents = await table_file.read()
        try:
            table_data = json.loads(contents)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid JSON format in table file")
        
        if not all(key in table_data for key in ['question']):
            raise HTTPException(400, "Table file missing required fields")

        if 'table' not in table_data:
            retriever = await get_retriever_local()
            tables = await retriever.aget_relevant_documents(table_data['question'])
            table_data['table'] = tables[0]['table']
            table_data['table_id'] = tables[0]['table_id']
        

        logger.info(f"Processing query: {table_data['question'][:50]}... (table: {table_data['table_id']})")

        table = table_data['table']
        data = np.array(table['data'])
        n_cols = len(table['header'])
        df = pd.DataFrame(data.reshape(-1, n_cols), columns=table['header'])

        #df = pd.DataFrame(np.array(table_data['table']['data']).reshape(len(table_data['table']['data']) // len(table_data['table']['header']), len(table_data['table']['header'])))
        #df.columns = table_data['table']['header']
        table_data['table']['data'] = df
        table_data['table']['table_id'] = table_data['table_id']

        state = {
            "input": table_data['question'],
            'table_data': table_data['table'],
            'metadata': {
                'url': table_data['table'].get('url', ''),
                'title': table_data['table'].get('title', '')
            },
            "subtasks": [],
            "current_index": -1,
            "tool_call": [],
            "final_answer": None,
            "trace": []
        }

        final_state = None
        async for step in workflow_app.astream(state):
            node_name, node_state = next(iter(step.items()))
            final_state = node_state
            if final_state.get("final_answer"):
                break

        if not final_state or not final_state.get("final_answer"):
            raise RuntimeError("Processing failed to produce final answer")

        return FinalResponse(
            final_answer=final_state["final_answer"].get("final_answer", final_state["final_answer"]),
            subtasks=[
                SubtaskResult(
                    task=t["task"],
                    result=t["result"],
                    type=t["type"]
                ) for t in final_state["subtasks"]
            ],
            trace=final_state["trace"]
        )
    except RuntimeError as re:
        logger.error(f"Processing failure: {str(re)}")
        raise HTTPException(500, detail=str(re))
    '''
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(400, detail=str(ve))
    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(500, detail=f"Internal server error: {str(e)}")
    '''

if __name__ == "__main__":
    chroma_db_path ="./app/chroma_data"
    os.makedirs(chroma_db_path, exist_ok=True)
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        timeout_keep_alive=300,
    )
