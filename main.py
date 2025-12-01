from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

# Import all route modules
from api.routes import index, query, analysis, linting, quality, security, dashboard, duplication, cicd
from utils.logger import get_logger
from core.config import settings
from core.database import initialize_database

load_dotenv()

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting code-analyzer-backend")
    settings.validate()

    # Initialize database
    initialize_database()

    yield
    logger.info("Shutting down code-analyzer-backend")


app = FastAPI(
    title="Code Analyzer Backend",
    description="Analyze GitHub repositories and enable semantic Q&A",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(index.router, prefix="/api", tags=["indexing"])
app.include_router(query.router, prefix="/api", tags=["search"])
app.include_router(analysis.router, prefix="/api", tags=["analysis"])
app.include_router(linting.router, prefix="/api", tags=["linting"])
app.include_router(quality.router, prefix="/api", tags=["quality"])
app.include_router(security.router, prefix="/api", tags=["security"])
app.include_router(duplication.router, prefix="/api", tags=["duplication"])
app.include_router(cicd.router, prefix="/api", tags=["cicd"])
app.include_router(dashboard.router, prefix="/api", tags=["dashboard"])


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "code-analyzer-backend"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
