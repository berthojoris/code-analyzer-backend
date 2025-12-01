# AGENTS Rules and Responsibilities

## Purpose

This backend API is built using **Python** and **FastAPI** to support semantic code analysis and search.
This document defines the roles, responsibilities, and operational rules for each agent in the FastAPI backend for semantic code analysis and search.

## Guiding Principles

* **Data Security**: Do not retain user code permanently without permission
* **Analytical Accuracy**: Generate embeddings and explanations that are relevant and correct
* **System Transparency**: Provide clear logging without exposing sensitive data
* **API Compliance**: Follow API quotas and configuration best practices

## Agents Overview

The system is composed of multiple logical agents:

### 1. Indexing Agent

Responsibilities:

* Clone GitHub repositories using GitPython
* Detect the primary programming languages
* Parse and chunk code semantically
* Generate embeddings using `text-embedding-3-small`
* Store embeddings and metadata into Pinecone

Rules:

* Avoid cloning if the repository is already indexed
* Validate repository access (public / authorized)
* Chunk code while preserving semantic context (functions, classes, modules)

### 2. Search Agent

Responsibilities:

* Accept natural language queries
* Perform vector similarity search in Pinecone
* Retrieve the most relevant code snippets
* Return structured results

Rules:

* Avoid returning irrelevant or overly broad results
* Use fallback search strategies if vector search fails

### 3. Analysis Agent (OpenAI)

Responsibilities:

* Provide code explanations using GPT-4o-mini
* Deliver developer-friendly insights

Rules:

* Do not generate speculative analysis unless uncertainty is clearly stated
* Never reveal API keys or internal configuration details

## Operational Rules

* **Code Quality Standards**: Follow PEP 8 style guidelines and enforce static analysis (e.g., Ruff, MyPy)
* **Async First**: Prefer asynchronous endpoints and I/O operations
* **Secure Dependency Management**: Pin versions in requirements and scan vulnerabilities
* **Exception Safety**: Centralized error handling using FastAPI exception handlers
* **Testing**: Maintain high test coverage with pytest for indexing, search, and API flow
* **Error Handling**: Provide clear and actionable error messages
* **Resource Limits**: Restrict repository size and embedding request volume
* **Schema Updates**: Maintain backward compatibility on data model changes
* **Configuration via ENV**: All credentials must be loaded from environment variables
* **SQLite Default Storage**: Use SQLite for saving data, configuration, and internal state unless scaling requirements mandate another database
* **Error Handling**: Provide clear and actionable error messages
* **Resource Limits**: Restrict repository size and embedding request volume
* **Schema Updates**: Maintain backward compatibility on data model changes
* **Configuration via ENV**: All credentials must be loaded from environment variables

## API Boundaries

* **POST `/index`** — Index new repositories only
* **POST `/query`** — Must not modify persistent data
* **GET `/health`** — Basic diagnostic without exposing sensitive information

## Future Considerations

* Introduce role‑based access control
* Optimize multiprocess indexing and job scheduling
* Add caching for repeated queries

---