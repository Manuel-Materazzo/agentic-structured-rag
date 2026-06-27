import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.agents.sql_agent import SQLAgent
from app.agents.vector_store_agent import VectorStoreAgent
from app.orchestrator import Orchestrator
from ingestion.ingestion_manager import IngestionManager
from ingestion.knowledge_manager import KnowledgeManager

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    with IngestionManager.create() as ingestion_manager:
        ingestion_manager.health_check()
        log.info("✅ Data store Health Check Completed")

    knowledge_manager = KnowledgeManager.create().__enter__()
    log.info("✅ Data store initialized and available.")

    sql_agent = SQLAgent(knowledge_manager=knowledge_manager)
    vector_store_agent = VectorStoreAgent(knowledge_manager=knowledge_manager)
    orchestrator = Orchestrator(sql_agent=sql_agent, vector_store_agent=vector_store_agent)

    state["knowledge_manager"] = knowledge_manager
    state["orchestrator"] = orchestrator

    yield  # listen for api calls

    # shutdown
    knowledge_manager.__exit__(None, None, None)
    log.info("🛑 Knowledge manager closed.")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=dict)
async def predict_post(question: str):
    try:
        orchestrator: Orchestrator = state["orchestrator"]
        return orchestrator.execute(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
