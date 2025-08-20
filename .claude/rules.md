# Claude Rules for MLX Trading Pipeline

## Project-Specific Guidelines

### Apple Silicon Optimization Requirements
- All code must leverage Apple Silicon architecture (M1/M2/M3)
- Use MLX framework for GPU/ANE acceleration where applicable
- Ensure Metal backend compatibility for LightGBM
- Target unified memory architecture utilization
- Performance benchmarks must exceed traditional x86 setups

### Trading Pipeline Architecture Patterns
- Follow async task-based design patterns
- Use MLXTaskExecutor for central coordination
- Implement circuit breaker patterns with purgatory library
- Ensure real-time processing capabilities (<10ms latency)
- Maintain stateful feature computation with history buffers

### Data Handling Standards
- Use Pydantic models for all data structures
- Implement proper type hints throughout
- Follow lazy evaluation patterns with Polars
- Ensure proper OHLCV data validation
- Handle real-time streaming data efficiently

### Model Development Guidelines
- GPU-accelerated training is mandatory
- Use Optuna for hyperparameter optimization
- Implement early stopping mechanisms
- Ensure model versioning and artifact management
- Target >95% test coverage for ML components

### API Design Patterns
- Use FastAPI with async endpoints
- Implement proper dependency injection
- Use background tasks for post-response processing
- Ensure WebSocket support for real-time updates
- Follow OpenAPI documentation standards

### Testing Requirements
- Integration tests for full pipeline
- Performance benchmarks on Apple Silicon
- Mock data testing with realistic scenarios
- Circuit breaker failure testing
- GPU acceleration validation tests

### Documentation Standards
- Follow Context7 documentation format
- Include performance metrics in docstrings
- Document Apple Silicon specific optimizations
- Provide troubleshooting guides
- Maintain architecture decision records

### Code Quality Standards
- Use black formatting (88 character line length)
- Run ruff linting with project configuration
- Ensure mypy type checking passes
- Use pre-commit hooks for quality gates
- Maintain project-specific import organization

### Dependencies and Tooling
- Use uv as package manager
- Prefer MLX over standard numpy operations
- Use polars over pandas for data processing
- Implement prometheus metrics collection
- Use uvloop for async event loops

### Performance Targets
- Feature computation: <5ms average
- Model inference: <10ms average  
- Throughput: >10k samples/second
- Memory usage: <16GB peak
- GPU utilization: >70% during training

### Security and Compliance
- Never commit API keys or secrets
- Use environment variables for configuration
- Implement proper error handling without data leaks
- Follow financial data privacy requirements
- Use secure WebSocket connections

### Development Workflow
- Create feature branches for all changes
- Use make commands for common operations
- Validate pipeline setup before development
- Run full test suite before commits
- Document breaking changes in PRPs