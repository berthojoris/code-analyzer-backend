from typing import List, Optional
from urllib.parse import urlparse

from openai import OpenAI
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.config import settings
from core.embeddings.embeddings import get_embedding_service
from core.vector_store.pinecone_store import get_pinecone_store
from utils.logger import get_logger

LLM_MODEL = "gpt-4o-mini"
SIMILARITY_SCORE_THRESHOLD = 0.25

logger = get_logger(__name__)
router = APIRouter()


def extract_query_concepts(query: str) -> list[str]:
    """Extract key concepts from user query for better matching."""
    import re
    
    concepts = []
    query_lower = query.lower()
    
    # Map common user intents to code concepts
    concept_mappings = {
        r'framework|library|dependencies': ['imports', 'dependencies', 'package.json', 'requirements'],
        r'api|endpoint|route': ['routing endpoint', 'HTTP request API call', 'router'],
        r'auth|login|sign.?in|sign.?up': ['authentication', 'login', 'auth', 'token'],
        r'database|db|store|persist': ['database operations', 'query', 'model'],
        r'test|spec': ['testing', 'test', 'spec'],
        r'component|ui|render|display': ['UI component rendering', 'component', 'render'],
        r'state|redux|context|store': ['React hooks state management', 'state', 'store'],
        r'async|promise|await': ['asynchronous code', 'async', 'await'],
        r'error|exception|catch': ['error handling', 'try', 'catch'],
        r'websocket|realtime|socket': ['realtime websocket', 'socket'],
        r'config|setting|environment': ['configuration', 'config', 'env'],
        r'style|css|theme': ['styling', 'css', 'theme'],
        r'hook|useEffect|useState': ['React hooks state management', 'hooks'],
    }
    
    for pattern, related_concepts in concept_mappings.items():
        if re.search(pattern, query_lower):
            concepts.extend(related_concepts)
    
    return list(set(concepts))


def enhance_query_for_code_search(query: str) -> str:
    """Enhance user query to match document embedding format.
    
    Aligns query format with how documents are embedded:
    - Documents have: Description, Concepts, Type, Name, Language, Code
    - Query should match these semantic fields
    """
    query = query.strip()
    query_lower = query.lower()
    
    # If query looks like code, search for it directly
    code_indicators = ["def ", "class ", "function ", "import ", "const ", "var ", "let ", "=>", "->"]
    if any(indicator in query_lower for indicator in code_indicators):
        return f"Code: {query}"
    
    # Extract concepts from query
    concepts = extract_query_concepts(query)
    
    # Build enhanced query that matches document format
    parts = [f"Description: {query}"]
    
    if concepts:
        parts.append(f"Concepts: {', '.join(concepts)}")
    
    # Add the original query for direct matching
    parts.append(f"Search: {query}")
    
    return "\n".join(parts)


def extract_repo_name(repo_url: str) -> str:
    """Extract repository name (owner/repo) from GitHub URL."""
    parsed = urlparse(repo_url)
    path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    parts = path.split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return parts[-1] if parts else ""


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language question about the codebase")
    repo_url: Optional[str] = Field(None, description="GitHub repository URL to filter results")
    repo: Optional[str] = Field(None, description="Filter results to specific repository (owner/repo)")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")


class CodeMatch(BaseModel):
    file: str
    snippet: str
    language: str
    repo: str
    chunk_type: str
    name: str
    score: float
    start_line: int
    end_line: int


class SearchResponse(BaseModel):
    query: str
    answer: Optional[str] = None
    matches: List[CodeMatch]
    total_matches: int


def build_code_context(matches: List[CodeMatch]) -> str:
    """Build context string from code matches for LLM."""
    context_parts = []
    for m in matches[:5]:
        context_parts.append(
            f"File: {m.file}\nLines {m.start_line}-{m.end_line}:\n```{m.language}\n{m.snippet}\n```"
        )
    return "\n\n".join(context_parts)


def get_llm_analysis(query: str, code_context: str) -> str:
    """Call LLM to analyze code and answer the question."""
    client = OpenAI(api_key=settings.openai_api_key)
    
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a code analyst. Answer questions about code based on the provided snippets. Be concise and reference specific files/lines when relevant."
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nRelevant code:\n{code_context}"
            }
        ],
        max_tokens=500
    )
    
    return response.choices[0].message.content


@router.post("/search", response_model=SearchResponse)
async def search_code(request: SearchRequest):
    """Search for code snippets using natural language query."""
    query = request.query.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Determine repo filter: prefer repo_url, fall back to repo
    repo_filter = None
    if request.repo_url:
        repo_filter = extract_repo_name(request.repo_url)
        logger.info(f"Extracted repo filter from URL: {repo_filter}")
    elif request.repo:
        repo_filter = request.repo

    try:
        # Enhance query for better code search relevance
        enhanced_query = enhance_query_for_code_search(query)
        logger.info(f"Enhanced query: {enhanced_query}")
        
        # Generate embedding for the enhanced query
        embedding_service = get_embedding_service()
        query_embedding = embedding_service.create_embedding(enhanced_query)

        # Search Pinecone - request more results to account for filtering
        store = get_pinecone_store()
        search_limit = request.top_k * 3 if repo_filter else request.top_k
        results = store.search(
            query_embedding=query_embedding,
            top_k=search_limit,
            repo_filter=repo_filter,
        )

        # Format and filter results
        matches = []
        for result in results:
            score = result.get("score", 0.0)
            
            # Filter out low-relevance results
            if score < SIMILARITY_SCORE_THRESHOLD:
                logger.info(f"Filtering out low-score result: {score:.3f} (threshold: {SIMILARITY_SCORE_THRESHOLD})")
                continue
            
            metadata = result.get("metadata", {})
            result_repo = metadata.get("repo", "")
            
            # Secondary filter: ensure results match requested repo
            if repo_filter and result_repo != repo_filter:
                logger.debug(f"Filtering out result from repo: {result_repo}")
                continue
                
            matches.append(
                CodeMatch(
                    file=metadata.get("file", ""),
                    snippet=metadata.get("snippet", ""),
                    language=metadata.get("language", ""),
                    repo=result_repo,
                    chunk_type=metadata.get("chunk_type", ""),
                    name=metadata.get("name", ""),
                    score=score,
                    start_line=metadata.get("start_line", 0),
                    end_line=metadata.get("end_line", 0),
                )
            )
            
            # Stop once we have enough matches
            if len(matches) >= request.top_k:
                break

        logger.info(f"Returning {len(matches)} matches for repo: {repo_filter or 'all'}")

        # Generate LLM analysis if we have matches
        answer = None
        if matches:
            try:
                code_context = build_code_context(matches)
                answer = get_llm_analysis(query, code_context)
                logger.info("LLM analysis generated successfully")
            except Exception as llm_error:
                logger.error(f"LLM analysis failed: {llm_error}")
                answer = None

        return SearchResponse(
            query=query,
            answer=answer,
            matches=matches,
            total_matches=len(matches),
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}",
        )
