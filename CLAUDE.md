# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

jpgovsummary is a Python tool that automatically summarizes Japanese government meeting documents (from gov.jp). It uses LangChain/LangGraph for workflow orchestration and OpenAI API for AI-powered summarization. The tool extracts content from HTML/PDF documents, discovers related materials, and generates integrated summaries with optional Bluesky posting.

## Common Commands

### Development Setup
```bash
# Install dependencies
poetry install

# Run the tool (development mode)
poetry run jpgovsummary <URL_or_FILE_PATH>
```

### Running the Tool
```bash
# Basic usage with HTML meeting page
poetry run jpgovsummary https://www.kantei.go.jp/jp/singi/example/

# Basic usage with PDF file
poetry run jpgovsummary /path/to/document.pdf

# Batch mode (no human interaction)
poetry run jpgovsummary <URL> --batch

# Overview only (skip related documents)
poetry run jpgovsummary <URL> --overview-only

# Use specific OpenAI model
poetry run jpgovsummary <URL> --model gpt-4o

# Skip Bluesky posting
poetry run jpgovsummary <URL> --skip-bluesky-posting
```

### Code Quality
```bash
# Run linting
poetry run ruff check

# Run formatting
poetry run ruff format

# Auto-fix linting issues
poetry run ruff check --fix
```

## Architecture

### LangGraph Workflow System

The application uses **LangGraph** for state management and agent orchestration. The workflow is defined as a state graph in [jpgovwatcher.py](src/jpgovsummary/jpgovwatcher.py) with different paths depending on input type:

**HTML Meeting Page Flow:**
1. `main_content_extractor` → Extract main content from HTML
2. `overview_generator` → Generate meeting overview
3. **Conditional:** If `overview_only` mode or meeting minutes detected → `summary_finalizer`
4. Otherwise: `report_enumerator` → Discover related documents
5. `report_selector` → Select important documents to summarize
6. `document_summarizer` → Summarize each selected document (loops via `target_report_index`)
7. `summary_integrator` → Integrate all summaries
8. `summary_finalizer` → Human review and finalization
9. `bluesky_poster` → Optional Bluesky posting (if not skipped)

**PDF File Flow:**
1. `document_summarizer` → Summarize the PDF directly
2. `summary_integrator` → Create integrated summary
3. `summary_finalizer` → Human review and finalization
4. `bluesky_poster` → Optional Bluesky posting (if not skipped)

### State Management

The application state is defined in [state.py](src/jpgovsummary/state.py) using TypedDict and Pydantic models:

- **State fields:** Contains all workflow data including messages, summaries, reports, review status
- **Report models:** `Report`, `ScoredReport`, `CandidateReport` with corresponding list types
- **Summary model:** Tracks document summaries with URL, name, content, and document type
- **Review fields:** `review_session`, `review_approved`, `final_review_summary` for human interaction
- **Flags:** `batch`, `skip_bluesky_posting`, `overview_only`, `meeting_minutes_detected`, `is_meeting_page`

### Agent Nodes

Each agent is a separate module in [src/jpgovsummary/agents/](src/jpgovsummary/agents/):

- **main_content_extractor:** Extracts clean meeting content from HTML markdown
- **overview_generator:** Creates initial meeting overview, detects meeting minutes
- **report_enumerator:** Discovers related PDF/document URLs from meeting page
- **report_selector:** Scores and selects top documents to summarize
- **document_summarizer:** Processes individual documents (loops via state.target_report_index)
- **summary_integrator:** Combines overview + document summaries into final summary
- **summary_finalizer:** Handles human review, quality checks, character limits
- **bluesky_poster:** Posts approved summaries to Bluesky (optional)

### Conditional Edges

Two key conditional functions control workflow branching:

- `should_process_additional_files()` - Determines if related documents should be processed
- `should_continue_target_reports()` - Controls document summarization loop

### Model and Configuration

- **Model class** ([model.py](src/jpgovsummary/model.py)): Singleton pattern for OpenAI model initialization, defaults to `OPENAI_MODEL_NAME` env var
- **Config class** ([config.py](src/jpgovsummary/config.py)): Thread-based configuration for LangGraph checkpointing

### Tools

Located in [src/jpgovsummary/tools/](src/jpgovsummary/tools/):

- **html_loader:** Loads HTML pages and converts to markdown using docling
- **pdf_loader:** Loads PDF files and converts to markdown using docling

## Environment Variables

Required:
- `OPENAI_API_KEY` - OpenAI API key for text generation
- `OPENAI_MODEL_NAME` - OpenAI model to use (can be overridden via --model flag)

Optional:
- `SSKY_USER` - Bluesky credentials in format "handle.bsky.social:app-password" for posting

Use `.env` file or export directly. See `.env.local.sample` for template.

## Git Workflow (from .cursor/rules)

**Commit message format:** `{prefix}: {message}` or `{prefix}: {message} (#{issue_number})`

**Prefixes:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `refactor` - Code refactoring
- `chore` - Maintenance tasks
- `test` - Adding or modifying tests

**Constraints:**
- Language: English
- Max 20 words
- Imperative style
- Single sentence (one line only)
- **No author footers** - Do not add "Generated with Claude Code" or "Co-Authored-By" footers

**Commit workflow:**
1. Check status (`git status`)
2. List all changes (new, modified, deleted files)
3. Stage all changes properly (add new/modified, remove deleted)
4. Commit with proper message (without author footers)
5. Push to remote

## Code Standards

- Python 3.12+
- Line length: 100 characters
- Ruff for linting and formatting (see [pyproject.toml](pyproject.toml) configuration)
- Double quotes for strings
- Import order: isort with jpgovsummary as first-party

### Python Compliance
- Follow PEP 8 style guide
- Use PEP 484 type hints and PEP 604 union types (e.g., `str | None`)
- Organize imports properly: no unused imports, no wildcard imports
- Keep code self-documenting with clear naming
- Handle errors appropriately with proper logging

### Code Quality Checks
Before committing, ensure:
- All tests pass
- No debug code or comments left
- No hardcoded values or credentials
- Error handling is appropriate
- Changes are focused and minimal

## Bluesky Integration

The tool includes optional Bluesky posting functionality via the `bluesky_poster` agent.

### Character Limits
Character limits are defined as constants in [bluesky_poster.py](src/jpgovsummary/agents/bluesky_poster.py):
- `MAX_CHARS_INTEGRATED_SUMMARY = 2000` - Maximum for integrated summary (summary + URL + newline)
- `MAX_CHARS_BLUESKY_LONG = 1000` - Maximum for Bluesky posting
- `MAX_CHARS_BLUESKY_SHORT = 1000` - Maximum for Bluesky posting (same as LONG)
- `MIN_CHARS_SUMMARY = 50` - Minimum characters to ensure for summary content
- `MIN_CHARS_INTEGRATED = 200` - Minimum characters to ensure for integrated summary

### Important Safety Rules
When working with Bluesky posting code:
- **Never post without user approval** - Always use dry run first in interactive mode
- Check authentication status before operations
- Never hardcode credentials - use `SSKY_USER` environment variable
- Respect Bluesky's character limits (defined in constants above)
- Handle errors gracefully with clear messages

### Workflow
1. Check authentication status
2. Get user approval for content
3. Handle posting or skipping based on user input
4. Provide clear feedback on success/failure

### Testing
Use `--skip-bluesky-posting` flag to test without posting, or `--batch` mode which automatically skips user confirmation.

## Output Format

The tool outputs a simple two-line format to stdout:
```
{summary_text}
{source_url}
```

Processing logs and human interaction go to stderr.
