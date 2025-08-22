# Prime Context for Claude Code

Use the command `tree` to get an understanding of the project structure.

Start with reading the CLAUDE.md file if it exists to get an understanding of the project.

Read the README.md file to get an understanding of the project.

Read key files in the src/ or root directory

💡

IMPORTANT: Use Serena to search through the codebase. If you get any errors using Serena, retry with different Serena tools.

Explain back to me:
- Project structure
- Project purpose and goals  
- Key files and their purposes
- Any important dependencies
- Any important configuration files

## MLX Trading Pipeline - Quick Command Reference

### Essential Project Understanding
```bash
# Project structure overview
tree -I "__pycache__|*.pyc|.git|node_modules"

# Read project documentation
cat CLAUDE.md
cat README.md

# Check current status
make status
python run_py.py -c "import mlx_trading_pipeline; print('✅ Package works')"
```

### Core Development Commands
```bash
# Environment setup
make install    # Install dependencies
make setup      # Setup dev environment
make rebuild    # Rebuild from scratch

# Universal Python execution (ALWAYS USE THIS)
python run_py.py <command> [args...]
python run_py.py -m pytest tests/ -v --tb=short
python run_py.py main.py

# Alternative via Make
make uv_run ARGS='-m pytest tests/test_complete_pipeline.py -v'
```

### Pipeline Operations  
```bash
# Run trading pipeline
make run
python run_py.py main.py

# Run with mock data
make run-mock-api

# Fetch historical data
make fetch-historical-data

# Train models
make train-models
```

### Testing & Quality
```bash
# Run all tests (PREFERRED)
python run_py.py -m pytest tests/ -v --tb=short

# Specific pipeline tests
python run_py.py -m pytest tests/test_complete_pipeline.py -v --tb=short

# Code quality
make lint       # ruff check . --fix
make format     # black . --line-length 88
make type-check # mypy . --ignore-missing-imports
```

### Serena MCP Tools (Advanced Code Analysis)
```bash
# Use these Serena tools for semantic code exploration:
# - mcp__serena__get_symbols_overview  
# - mcp__serena__find_symbol
# - mcp__serena__search_for_pattern
# - mcp__serena__find_referencing_symbols
```

## Project Context
- **Apple Silicon Trading Pipeline** with MLX GPU acceleration
- **Performance Targets**: >10k samples/sec, <10ms inference, >80% accuracy
- **Binary Classification**: Labels 0 (Hold/Sell), 1 (Buy)
- **37+ Technical Indicators** in real-time feature computation
- **Async Task Architecture** with priority-based execution