# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commandsaa

### Environment Setup
```bash
# Install dependencies
make install

# Setup development environment  
make setup

# Rebuild environment from scratch
make rebuild
```

### Core Pipeline Operations
```bash
# Run main trading pipeline
make run
python main.py

# Run with mock data for testing
make run-mock-api

# Fetch historical data (User Story 1)
make fetch-historical-data

# Train unified models (User Story 3)  
make train-models
```

### Unified Python Execution (ALWAYS USE THIS)
**IMPORTANT**: Always use these commands when running any Python logic, tests, or scripts to ensure consistent environment setup:

```bash
# Universal Python runner - use for ANY Python execution
python run_py.py <command> [args...]

# Examples:
python run_py.py -m pytest tests/test_complete_pipeline.py -v --tb=short
python run_py.py -m pytest tests/ -v --tb=short  
python run_py.py main.py
python run_py.py -c "import mlx_trading_pipeline; print('works')"

# Via Make target (alternative)
make uv_run ARGS='-m pytest tests/test_complete_pipeline.py -v'
make uv_run ARGS='main.py'
make uv_run ARGS='-c "import mlx_trading_pipeline; print(\"works\")"'
```

### Testing
```bash
# Run all tests (PREFERRED METHOD)
python run_py.py -m pytest tests/ -v --tb=short

# Run complete pipeline tests specifically  
python run_py.py -m pytest tests/test_complete_pipeline.py -v --tb=short

# Run integration tests only
make test-integration

# Run performance benchmarks
make test-performance

# Legacy methods (still work)
make test
pytest tests/ -v --tb=short
```

### Code Quality
```bash
# Lint code (run before commits)
make lint
uv run ruff check . --fix

# Format code
make format
uv run black . --line-length 88

# Type checking
make type-check
uv run mypy . --ignore-missing-imports
```

### Pipeline Validation
```bash
# Validate complete pipeline setup
make validate-pipeline

# Check pipeline component status
make status
```

## Architecture Overview

This is an Apple Silicon-optimized real-time trading pipeline with MLX GPU acceleration. The system processes 10k+ samples/second with sub-10ms inference latency.

### Core Architecture Pattern
The pipeline follows an async task-based architecture centered around the `MLXTaskExecutor`:

```
Data Ingestion → Feature Engine → Model Training/Inference → API Service
       ↓              ↓               ↓                    ↓
   MQTT/WS       37+ Technical    GPU-Accelerated      FastAPI
   Streams       Indicators       LightGBM/MLX         Endpoints
```

### Key Components

1. **MLX Task Executor** (`trading_pipeline_executor.py`): Central async task coordinator with Apple Silicon GPU scheduling
2. **Feature Engine** (`trading_pipeline_features.py`): Stateful computation of 37+ technical indicators
3. **Data Ingestion** (`trading_pipeline_ingestion_hot.py`): Real-time MQTT/WebSocket stream processing
4. **Model Training** (`trading_pipeline_training.py`): GPU-accelerated LightGBM with Optuna optimization
5. **Inference Service** (`trading_pipeline_inference.py`): Sub-10ms prediction API with FastAPI
6. **Historical Data Fetcher** (`src/mlx_trading_pipeline/historical_data_fetcher.py`): Async historical data retrieval
7. **Sentiment Service** (`src/mlx_trading_pipeline/sentiment_service.py`): News sentiment analysis integration

### Entry Points
- **Main Pipeline**: `main.py` - Full trading system orchestration
- **Integrated Pipeline**: `src/mlx_trading_pipeline/main_integrated.py` - Alternative entry point
- **Complete Pipeline Test**: `test_complete_pipeline.py` - End-to-end validation

### Data Models
All data structures use Pydantic models in `trading_pipeline_models.py` for type safety:
- `MarketData`: OHLCV market data
- `FeatureVector`: ML model input with 37+ features
- Additional models for API requests/responses

### Technology Stack Specifics

**Apple Silicon Optimization:**
- MLX framework for GPU/ANE acceleration
- Metal backend for LightGBM
- Unified memory architecture utilization
- Performance: 12,500 samples/sec vs 3,200 traditional

**Async Architecture:**
- Priority-based task execution
- Resource-aware scheduling
- Circuit breaker pattern with `purgatory` library
- MQTT client connection management

**ML Pipeline:**
- Stateful feature computation with 200-sample history
- GPU-accelerated training with early stopping
- Real-time inference with model versioning
- Optuna hyperparameter optimization

## Configuration Management

### Environment Variables
Set in development:
```bash
export TRADING_SYMBOLS="AAPL,TSLA,NVDA,MSFT,GOOGL"
export ENABLE_GPU=true
export API_PORT=8000
```

### Time Horizon Configs
Configuration files in `configs/time_horizons/`:
- `horizon_1min.json` - 1-minute trading intervals
- `horizon_10min.json` - 10-minute trading intervals

### Mock Data
Test data available in `workbench/sample_data/`:
- `mock_market_data.parquet` - Historical market data
- `mock_news_data.json` - Sample news/sentiment data
- `mock_api_server.py` - Testing API server

## Development Workflow

### Package Manager
This project uses `uv` as the package manager for ultra-fast dependency resolution. Always use `uv run` or `make` commands rather than direct `python` commands.

### Testing Strategy
- Unit tests for individual components
- Integration tests for full pipeline
- Performance benchmarks with specific Apple Silicon metrics
- Mock data testing with realistic market scenarios

### Apple Silicon Considerations
- Code optimized for M1/M2/M3 processors
- GPU acceleration through MLX and Metal backend
- Unified memory architecture awareness
- Thermal throttling monitoring

### Circuit Breaker Migration
The project has migrated from `pybreaker` to `purgatory` for circuit breaker functionality. See `src/mlx_trading_pipeline/circuit_breaker_migration.py` for implementation details.

## Troubleshooting

### Common Issues
- **MLX not found**: Reinstall with `pip install mlx>=0.19.0`
- **GPU acceleration disabled**: Check Metal availability with MLX
- **Memory issues**: Reduce concurrent tasks via `MAX_CONCURRENT_TASKS=4`
- **LightGBM GPU issues**: Install via conda-forge for Apple Silicon

### Monitoring
- Application logs: `trading_pipeline.log`
- Pipeline metrics via `/metrics` endpoint
- Component health checks in main event loop
- Apple Silicon specific metrics (P-core/E-core utilization, GPU usage)

## Important Execution Instructions

### ALWAYS Use run_py.py for Python Execution
**CRITICAL**: When working with this codebase, ALWAYS use the unified Python runner to avoid import and path issues:

```bash
# ✅ CORRECT - Use this for ALL Python execution
python run_py.py <command> [args...]

# ❌ INCORRECT - Don't use these directly
python main.py
pytest tests/
uv run python main.py
```

The `run_py.py` script automatically:
- Sets up correct PYTHONPATH including src/ directory  
- Uses appropriate Python environment (prefers /Users/bprzybysz/nc-src/.venv)
- Ensures all imports work correctly for the package structure
- Provides consistent execution environment

### Examples of Correct Usage:
```bash
# Run tests
python run_py.py -m pytest tests/test_complete_pipeline.py -v --tb=short

# Run main pipeline
python run_py.py main.py

# Quick import test
python run_py.py -c "import mlx_trading_pipeline; print('✅ Package imports work')"

# Alternative via Make
make uv_run ARGS='-m pytest tests/test_complete_pipeline.py -v'
```

This ensures all tests pass and imports work correctly across the modular package structure.

## Context Engineering Framework

### Advanced AI Development Support
This project includes a comprehensive **Context Engineering** framework in the `context-engineering/` directory that provides:

#### Key Resources:
- **`context-engineering/README.md`**: Complete framework overview and methodology
- **`context-engineering/examples/`**: Implementation patterns for MLX Trading Pipeline
- **`context-engineering/use-cases/`**: Apple Silicon optimization scenarios  
- **`context-engineering/PRPs/`**: Product Requirements Prompts (detailed implementation blueprints)

#### Context Engineering Benefits:
- **10x Better than Prompt Engineering**: Complete implementation "screenplays" vs simple directions
- **Consistent Quality**: All implementations follow established Apple Silicon patterns
- **Performance Built-in**: MLX optimizations and <10ms targets integrated into every feature
- **Comprehensive Validation**: Built-in testing and benchmark frameworks

#### When to Use Context Engineering:
1. **New Feature Development**: Use `context-engineering/INITIAL.md` template
2. **Performance Optimization**: Reference Apple Silicon use cases
3. **Architecture Changes**: Update patterns and examples in framework
4. **AI-Assisted Development**: Provide PRPs as complete context for implementation

#### Usage Pattern:
```bash
# 1. Review relevant examples and use cases
cat context-engineering/examples/feature-implementation.md
cat context-engineering/use-cases/apple-silicon-optimization.md

# 2. Create structured feature request
cp context-engineering/INITIAL.md my-feature-request.md

# 3. Generate PRP (Product Requirements Prompt) 
# 4. Provide PRP to AI for implementation
# 5. Validate with integrated test framework
```

This context engineering approach ensures all implementations meet Apple Silicon performance targets and follow established architectural patterns.
- rememeber to call make run_py <cmd/script> for all py logic tests included