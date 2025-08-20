# MLX Trading Pipeline - Project Context

## Project Overview
Apple Silicon-optimized real-time intraday trading pipeline achieving 10k+ samples/second processing with sub-10ms inference latency using MLX GPU acceleration.

## Core Value Proposition
- **Performance**: 4x faster than traditional x86 setups
- **Efficiency**: 50% lower memory usage through unified architecture  
- **Real-time**: <10ms inference with >12k samples/second throughput
- **Scalability**: Async task-based architecture with resource management

## Technology Stack

### Core Frameworks
- **MLX**: Apple Silicon GPU/ANE acceleration framework
- **LightGBM**: GPU-accelerated ML training with Metal backend
- **FastAPI**: High-performance async web framework  
- **Polars**: Lazy evaluation data processing
- **Pydantic**: Type-safe data modeling
- **MQTT/WebSocket**: Real-time data streaming

### Development Tooling
- **uv**: Ultra-fast Python package manager
- **pytest**: Testing framework with async support
- **black/ruff**: Code formatting and linting
- **mypy**: Static type checking
- **make**: Build automation and workflow management

## Architecture Patterns

### Central Coordination Pattern
```python
MLXTaskExecutor (Central Hub)
    ├── Data Ingestion (MQTT/WS streams)
    ├── Feature Engine (37+ technical indicators)  
    ├── Model Training (GPU-accelerated)
    └── Inference Service (API endpoints)
```

### Data Flow Architecture
```
Market Data → Feature Engineering → ML Pipeline → Trading Signals
     ↓              ↓                  ↓              ↓
  MQTT/WS        Stateful          GPU Training    FastAPI
  Streams        Indicators        with MLX        WebSocket
```

### Component Responsibilities

**MLX Task Executor**: 
- Resource-aware async task scheduling
- GPU memory management
- Priority-based execution queues
- Health monitoring and circuit breaking

**Feature Engine**:
- Real-time technical indicator computation
- Stateful 200-sample history buffers
- <5ms computation targets per symbol
- 37+ indicators (SMA, EMA, RSI, MACD, etc.)

**Model Training**:
- GPU-accelerated LightGBM with Metal backend
- Optuna hyperparameter optimization
- Early stopping and cross-validation
- Model artifact versioning and persistence

**Data Ingestion**:
- Multi-protocol support (MQTT, WebSocket)
- Circuit breaker pattern with purgatory
- Real-time stream processing
- Data validation and error handling

**Inference Service**:
- <10ms prediction latency
- Batch and single prediction APIs
- WebSocket broadcasting for real-time updates
- Prometheus metrics collection

## Performance Characteristics

### Benchmarks (Apple M2 Pro)
- Feature computation: 2.3ms average (vs 12.5ms traditional)
- Model inference: 6.7ms average (vs 28.3ms traditional)  
- Memory usage: 8.2GB peak (vs 16.4GB traditional)
- Throughput: 12,500 samples/sec (vs 3,200 traditional)

### Resource Utilization
- GPU utilization: 70-85% during training
- Unified memory efficiency: 50% improvement
- P-core/E-core balanced workload distribution
- Thermal management: <85°C sustained operation

## File Structure and Key Locations

### Entry Points
- `main.py` - Primary pipeline orchestrator
- `src/mlx_trading_pipeline/main_integrated.py` - Alternative entry point
- `test_complete_pipeline.py` - End-to-end validation

### Core Components  
- `trading_pipeline_executor.py` - Central async coordinator
- `trading_pipeline_features.py` - Feature engineering engine
- `trading_pipeline_training.py` - ML model training
- `trading_pipeline_inference.py` - Prediction service
- `trading_pipeline_models.py` - Pydantic data models

### Configuration and Data
- `configs/time_horizons/` - Trading interval configurations
- `workbench/sample_data/` - Mock data for testing
- `models/` - Trained model artifacts
- `pyproject.toml` - Project dependencies and tooling

### Development Workflow
- `Makefile` - Build and development commands
- `BEST_PRACTICES.md` - Technology-specific guidelines
- `context-engineering/` - AI context and PRPs

## Development Patterns

### Async Architecture
All components use async/await patterns with proper resource management:
```python
async with MLXTaskExecutor(enable_gpu=True) as executor:
    await executor.schedule_task(priority="high", task=feature_computation)
```

### Type Safety
Comprehensive Pydantic modeling for all data structures:
```python  
class FeatureVector(BaseModel):
    symbol: str
    timestamp: float
    features: Dict[str, float]
    price: float
    returns: float = 0.0
```

### Circuit Breaker Pattern
Fault tolerance with purgatory library:
```python
@circuit_breaker(failure_threshold=5, timeout=30)
async def fetch_market_data(symbol: str) -> MarketData:
    # Implementation with automatic failure handling
```

### Testing Strategy
- Unit tests: Individual component validation
- Integration tests: Full pipeline workflows  
- Performance tests: Apple Silicon benchmarks
- Mock data tests: Realistic market scenarios

## Common Development Tasks

### Environment Setup
```bash
make setup          # Initialize development environment
make install        # Install all dependencies
make validate-pipeline  # Verify component setup
```

### Development Workflow
```bash
make run            # Start full pipeline
make test           # Run complete test suite
make lint           # Code quality checks
make format         # Apply code formatting
```

### Pipeline Operations
```bash
make fetch-historical-data  # Download market data
make train-models           # GPU-accelerated training
make run-mock-api          # Testing with mock data
```

## Troubleshooting Common Issues

### MLX Setup Issues
- Verify Apple Silicon compatibility
- Check Metal framework availability
- Ensure MLX version >= 0.19.0

### Performance Degradation
- Monitor GPU utilization metrics
- Check thermal throttling events
- Validate memory usage patterns
- Review async task queue depths

### Data Pipeline Issues
- Verify MQTT broker connectivity
- Check WebSocket connection stability
- Validate market data formats
- Monitor circuit breaker states