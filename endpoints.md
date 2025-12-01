# API Endpoints Documentation

## Overview
This document contains all API endpoints for the Code Analyzer Backend service.

**Base URL:** `http://localhost:8000`

---

## Health Check

### GET /health
Health check endpoint to verify the service is running.

**Response:**
```json
{
  "status": "healthy",
  "service": "code-analyzer-backend"
}
```

---

## Indexing Endpoints

### POST /api/index
Index a GitHub repository for semantic search.

**Request Body:**
```json
{
  "repo_url": "string (required)",
  "reindex": "boolean (default: false)"
}
```

**Response:**
```json
{
  "status": "string",
  "repo_name": "string",
  "files_processed": "integer",
  "chunks_indexed": "integer",
  "dominant_language": "string",
  "language_stats": "object",
  "message": "string"
}
```

---

## Search Endpoints

### POST /api/search
Search for code snippets using natural language query.

**Request Body:**
```json
{
  "query": "string (required)",
  "repo_url": "string (optional)",
  "repo": "string (optional)",
  "top_k": "integer (default: 5, range: 1-20)"
}
```

**Response:**
```json
{
  "query": "string",
  "answer": "string (optional)",
  "matches": [
    {
      "file": "string",
      "snippet": "string",
      "language": "string",
      "repo": "string",
      "chunk_type": "string",
      "name": "string",
      "score": "float",
      "start_line": "integer",
      "end_line": "integer"
    }
  ],
  "total_matches": "integer"
}
```

---

## Analysis Endpoints

### POST /api/analyze
Start comprehensive repository analysis (includes linting and quality metrics).

**Request Body:**
```json
{
  "repo_url": "string (required)",
  "reanalyze": "boolean (default: false)",
  "include_linting": "boolean (default: true)",
  "include_quality": "boolean (default: true)",
  "max_files_mb": "integer"
}
```

**Response:**
```json
{
  "repository_id": "integer",
  "status": "string",
  "message": "string"
}
```

### GET /api/{repository_id}/status
Get analysis status for a repository.

**Path Parameters:**
- `repository_id`: integer

**Response:**
```json
{
  "repository_id": "integer",
  "status": "string (pending, running, completed, failed)",
  "files_analyzed": "integer",
  "files_total": "integer",
  "progress_percentage": "float",
  "started_at": "datetime",
  "estimated_completion": "datetime (optional)",
  "error_message": "string (optional)"
}
```

### GET /api/{repository_id}/results
Get complete analysis results for a repository.

**Path Parameters:**
- `repository_id`: integer

**Response:**
```json
{
  "repository_id": "integer",
  "repository_name": "string",
  "analysis_date": "datetime",
  "total_files": "integer",
  "linting_issues": "array",
  "quality_metrics": "array",
  "analysis_summary": "object"
}
```

---

## Linting Endpoints

### GET /api/linting/{owner}/{repo_name}
Get all linting issues for a repository with optional filters.

**Path Parameters:**
- `owner`: string (GitHub username/organization)
- `repo_name`: string (Repository name)

**Query Parameters:**
- `severity`: string (optional) - Filter by severity
- `tool`: string (optional) - Filter by tool
- `file_path`: string (optional) - Filter by file path

**Response:**
```json
{
  "repository_id": "integer",
  "repository_name": "string",
  "total_issues": "integer",
  "error_count": "integer",
  "warning_count": "integer",
  "info_count": "integer",
  "issues": [
    {
      "id": "integer",
      "repository_id": "integer",
      "file_path": "string",
      "line_number": "integer",
      "column_number": "integer",
      "rule_id": "string",
      "severity": "string",
      "message": "string",
      "tool": "string",
      "category": "string",
      "suggestion": "string (optional)"
    }
  ]
}
```

### GET /api/linting/{owner}/{repo_name}/file/{file_path}
Get linting issues for a specific file.

**Path Parameters:**
- `owner`: string
- `repo_name`: string
- `file_path`: string (file path, can include `/`)

**Response:** Array of linting issue objects

### GET /api/linting/{owner}/{repo_name}/severity/{severity}
Get linting issues filtered by severity.

**Path Parameters:**
- `owner`: string
- `repo_name`: string
- `severity`: string (e.g., "error", "warning", "info")

**Response:** Array of linting issue objects

---

## Quality Endpoints

### GET /api/quality/{owner}/{repo_name}
Get quality metrics overview for a repository.

**Path Parameters:**
- `owner`: string
- `repo_name`: string

**Response:**
```json
{
  "repository_id": "integer",
  "repository_name": "string",
  "total_files": "integer",
  "analyzed_files": "integer",
  "avg_cyclomatic_complexity": "float",
  "avg_maintainability_index": "float",
  "max_complexity": "integer",
  "max_maintainability": "float",
  "total_code_lines": "integer",
  "total_comment_lines": "integer",
  "total_blank_lines": "integer",
  "quality_distribution": "object"
}
```

### GET /api/quality/{owner}/{repo_name}/complexity
Get complexity metrics for a repository.

**Path Parameters:**
- `owner`: string
- `repo_name`: string

**Response:** Array of quality metric objects with complexity data

### GET /api/quality/{owner}/{repo_name}/maintainability
Get maintainability metrics for a repository.

**Path Parameters:**
- `owner`: string
- `repo_name`: string

**Response:** Array of quality metric objects with maintainability data

### GET /api/quality/{owner}/{repo_name}/trends
Get quality trends over time for a repository.

**Path Parameters:**
- `owner`: string
- `repo_name`: string

**Response:**
```json
{
  "total_analyses": "integer",
  "recent_complexity_trend": "array",
  "recent_maintainability_trend": "array",
  "analysis_dates": "array"
}
```

---

## Security Endpoints

### POST /api/security/scan
Trigger a security scan for a repository.

**Request Body:**
```json
{
  "repository_id": "string (required)",
  "force_rescan": "boolean (default: false)"
}
```

**Response:**
```json
{
  "status": "string",
  "message": "string",
  "repository_id": "string",
  "scan_id": "string"
}
```

### GET /api/security/{owner}/{repo_name}/issues
Get security issues for a repository.

**Path Parameters:**
- `owner`: string
- `repo_name`: string

**Query Parameters:**
- `severity`: string (optional)
- `tool`: string (optional)
- `limit`: integer (default: 100, range: 1-1000)

**Response:** Array of security issue objects

### GET /api/security/{owner}/{repo_name}/report
Get comprehensive security report for a repository.

**Path Parameters:**
- `owner`: string
- `repo_name`: string

**Response:** Security report object

### GET /api/security/{owner}/{repo_name}/summary
Get security summary for a repository.

**Path Parameters:**
- `owner`: string
- `repo_name`: string

**Response:**
```json
{
  "repo_id": "string",
  "security_score": "integer",
  "total_issues": "integer",
  "critical_issues": "integer",
  "high_issues": "integer",
  "medium_issues": "integer",
  "low_issues": "integer",
  "last_scan": "string",
  "tools_status": "object"
}
```

---

## Dashboard Endpoints

### GET /api/dashboard/{owner}/{repo_name}
Get comprehensive dashboard data for a specific repository.

**Path Parameters:**
- `owner`: string
- `repo_name`: string

**Response:**
```json
{
  "repository_id": "integer",
  "repository_name": "string",
  "owner": "string",
  "repo_name": "string",
  "analysis_status": "string",
  "last_analyzed": "datetime (optional)",
  "total_files": "integer",
  "analyzed_files": "integer",
  "dominant_language": "string",
  "health_score": "float",
  "issue_summary": "object",
  "quality_summary": "object",
  "security_summary": "object",
  "trend_data": "object"
}
```

### GET /api/dashboard/overview
Get overall dashboard statistics and overview.

**Response:**
```json
{
  "total_repositories": "integer",
  "active_analyses": "integer",
  "total_issues": "integer",
  "avg_health_score": "float",
  "language_distribution": "object",
  "recent_activity": "array"
}
```

### GET /api/dashboard/{owner}/{repo_name}/metrics
Get specific metrics for a repository.

**Path Parameters:**
- `owner`: string
- `repo_name`: string

**Query Parameters:**
- `metric_type`: string (default: "all", options: "all", "quality", "linting", "security")

**Response:**
```json
{
  "repository_id": "integer",
  "repository_name": "string",
  "metric_type": "string",
  "metrics": "object"
}
```

---

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request
```json
{
  "detail": "Error description"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Error description"
}
```

---

## Example Usage

### Index a Repository
```bash
curl -X POST http://localhost:8000/api/index \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/username/repo"}'
```

### Search Code
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How is authentication handled?",
    "repo_url": "https://github.com/username/repo",
    "top_k": 5
  }'
```

### Get Repository Dashboard
```bash
curl http://localhost:8000/api/dashboard/owner/repo-name
```

### Get Linting Issues
```bash
curl http://localhost:8000/api/linting/owner/repo-name?severity=error
```

### Get Quality Metrics
```bash
curl http://localhost:8000/api/quality/owner/repo-name
```
