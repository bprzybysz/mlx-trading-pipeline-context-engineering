# PRP: Binary Classification Training Demo for MLX Trading Pipeline

## 1. Feature Overview

### Objective
Implement a comprehensive binary classification training demo that showcases the MLX Trading Pipeline's ability to train models using the existing two-label system (0: Hold/Sell, 1: Buy) with real feature vectors and demonstrate Apple Silicon optimization benefits.

### Business Value
- **Proof of Concept**: Validate the complete ML training pipeline with realistic data
- **Performance Demonstration**: Showcase Apple Silicon MLX acceleration vs traditional CPU training
- **Training Validation**: Ensure the binary classification approach works with computed technical indicators
- **Demo Material**: Create compelling demonstration of sub-10ms inference with >85% accuracy targets

### Success Criteria
- Model achieves >80% accuracy on test data with binary classification
- MLX-accelerated training shows >2x speedup vs CPU-only training
- End-to-end training demo completes in <60 seconds with 1000+ samples
- Inference latency remains <10ms per prediction
- Demo generates realistic feature vectors using actual technical indicators

## 2. Apple Silicon Requirements

### MLX Framework Integration
- [x] GPU acceleration for LightGBM training with Metal backend
- [x] MLX array operations for feature preprocessing and batch inference
- [x] Unified memory architecture optimization for large feature matrices
- [x] Metal backend compatibility for gradient boosting operations

### Performance Targets
- **Training Latency**: <60 seconds for 1000+ samples with hyperparameter optimization
- **Inference Latency**: <10ms per prediction (individual feature vector)
- **Batch Inference**: >500 predictions/second for batch processing
- **Memory Usage**: <4GB total memory footprint during training
- **GPU Utilization**: >70% during training phases, >50% during inference

### Thermal and Power Considerations
- [x] Sustained performance under thermal load during extended training
- [x] P-core/E-core workload distribution (training on P-cores, inference on E-cores)
- [x] Power efficiency optimization during batch processing
- [x] Thermal throttling prevention with adaptive batch sizing

## 3. Technical Architecture

### Component Integration
- [x] **MLXTaskExecutor**: Schedule training tasks with HIGH priority, inference with MEDIUM priority
- [x] **Feature Engine**: Generate realistic feature vectors with 37+ technical indicators
- [x] **Model Training**: Binary classification with LightGBM + MLX acceleration
- [x] **Demo Interface**: Interactive demo script showing training → inference pipeline
- [x] **Performance Monitoring**: Real-time metrics during training and inference

### Data Models
```python
# Enhanced data models for binary classification demo
class BinaryClassificationLabel(Enum):
    HOLD_SELL = 0  # Hold or sell signal
    BUY = 1        # Buy signal

class TrainingDataPoint(BaseModel):
    feature_vector: FeatureVector
    label: BinaryClassificationLabel
    timestamp: float
    symbol: str
    actual_return: Optional[float] = None  # For validation

class TrainingDataset(BaseModel):
    training_points: List[TrainingDataPoint]
    validation_points: List[TrainingDataPoint]
    symbols: List[str]
    time_range: Tuple[float, float]
    feature_stats: Dict[str, float]

class BinaryClassificationResult(BaseModel):
    prediction: BinaryClassificationLabel
    confidence: float
    feature_importance: Dict[str, float]
    processing_time_ms: float

class TrainingMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time_seconds: float
    mlx_speedup_factor: float
    samples_processed: int
```

### API Changes
- [x] New demo endpoint: `/demo/binary-classification/train`
- [x] New demo endpoint: `/demo/binary-classification/predict`
- [x] New demo endpoint: `/demo/binary-classification/metrics`
- [x] WebSocket integration: Real-time training progress updates
- [x] Demo dashboard: Interactive visualization of training progress and results

## 4. Implementation Blueprint

### Phase 1: Core Implementation

1. **Setup Training Data Generation**
   - [x] Create synthetic but realistic market data generator
   - [x] Generate feature vectors using existing FeatureEngine
   - [x] Implement intelligent labeling logic based on future returns
   - [x] Create balanced dataset with ~50% buy/sell distribution
   - [x] Add data validation and quality checks

```python
class BinaryClassificationDemo:
    """
    CONTEXT: Comprehensive demo of binary classification training pipeline
    INPUT: Market symbols, training duration, model parameters
    OUTPUT: Trained model, performance metrics, live predictions
    PERFORMANCE: <60s training, <10ms inference, >80% accuracy
    """
    
    def __init__(
        self,
        symbols: List[str] = ["AAPL", "TSLA", "NVDA"],
        training_samples: int = 1000,
        enable_mlx: bool = True
    ):
        self.symbols = symbols
        self.training_samples = training_samples
        self.enable_mlx = enable_mlx
        self.feature_engine = FeatureEngine(max_history=200)
        self.model_trainer = ModelTrainer(enable_gpu=enable_mlx)
        self.task_executor = MLXTaskExecutor(enable_metrics=True)
```

2. **Implement Intelligent Labeling Strategy**
```python
async def generate_training_labels(
    self, 
    feature_vectors: List[FeatureVector],
    lookforward_periods: int = 5
) -> List[BinaryClassificationLabel]:
    """
    Generate labels based on future price movements.
    
    Strategy:
    - BUY (1): If price increases >2% in next 5 periods
    - HOLD/SELL (0): Otherwise
    """
    labels = []
    for i, fv in enumerate(feature_vectors[:-lookforward_periods]):
        current_price = fv.price
        future_prices = [fv_future.price for fv_future in 
                        feature_vectors[i+1:i+1+lookforward_periods]]
        
        max_future_return = max(
            (future_price - current_price) / current_price 
            for future_price in future_prices
        )
        
        # Label as BUY if returns exceed 2% threshold
        label = BinaryClassificationLabel.BUY if max_future_return > 0.02 else BinaryClassificationLabel.HOLD_SELL
        labels.append(label)
    
    return labels
```

3. **Integration with Pipeline**
   - [x] Integrate with MLXTaskExecutor for async training
   - [x] Add to unified training workflow
   - [x] Create real-time inference service
   - [x] Add comprehensive performance monitoring

### Phase 2: MLX Optimization and Testing

1. **Apple Silicon Optimization**
```python
class MLXAcceleratedTraining:
    """Apple Silicon optimized training with MLX acceleration."""
    
    async def train_with_mlx_acceleration(
        self, 
        features: List[FeatureVector],
        labels: List[int]
    ) -> Dict[str, Any]:
        """
        Train binary classifier with MLX GPU acceleration.
        
        PERFORMANCE TARGETS:
        - 2x speedup vs CPU training
        - <4GB memory usage
        - >70% GPU utilization
        """
        import mlx.core as mx
        import time
        
        # Convert to MLX arrays for GPU processing
        start_time = time.time()
        feature_matrix = self._convert_features_to_mlx_array(features)
        label_vector = mx.array(labels)
        
        # MLX-accelerated preprocessing
        normalized_features = self._mlx_normalize_features(feature_matrix)
        
        # Train with LightGBM Metal backend
        model_params = {
            'objective': 'binary',
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
        }
        
        model = await self._train_lightgbm_metal(
            normalized_features, 
            label_vector, 
            model_params
        )
        
        training_time = time.time() - start_time
        return {
            'model': model,
            'training_time_seconds': training_time,
            'mlx_acceleration_used': True,
            'gpu_utilization_percent': self._get_gpu_utilization()
        }
```

2. **Performance Validation Framework**
```python
@pytest.mark.benchmark
async def test_binary_classification_performance():
    """
    Comprehensive performance validation for binary classification demo.
    """
    demo = BinaryClassificationDemo(
        symbols=["AAPL", "TSLA", "NVDA"],
        training_samples=1000,
        enable_mlx=True
    )
    
    # Performance benchmarking
    cpu_metrics = await demo.run_training_benchmark(use_mlx=False)
    mlx_metrics = await demo.run_training_benchmark(use_mlx=True)
    
    # Validate performance targets
    assert mlx_metrics.training_time_seconds < 60.0
    assert mlx_metrics.inference_latency_ms < 10.0
    assert mlx_metrics.accuracy > 0.80
    assert mlx_metrics.training_time_seconds < cpu_metrics.training_time_seconds * 0.5
    assert mlx_metrics.gpu_utilization_percent > 70.0
    
    # Validate model quality
    assert mlx_metrics.precision > 0.75
    assert mlx_metrics.recall > 0.75
    assert mlx_metrics.f1_score > 0.75
```

### Phase 3: Demo Interface and Integration

1. **Interactive Demo Script**
```python
class InteractiveBinaryClassificationDemo:
    """
    Interactive demonstration of binary classification training.
    
    Features:
    - Real-time training progress
    - Live performance metrics
    - Interactive prediction testing
    - Apple Silicon optimization showcase
    """
    
    async def run_complete_demo(self):
        """Run the complete binary classification demo."""
        print("🚀 MLX Trading Pipeline Binary Classification Demo")
        print("=" * 60)
        
        # Phase 1: Data Generation
        print("📊 Generating training data...")
        training_data = await self.generate_realistic_training_data()
        
        # Phase 2: Model Training
        print("🧠 Training binary classification model...")
        training_results = await self.train_model_with_mlx(training_data)
        
        # Phase 3: Performance Validation
        print("⚡ Validating performance...")
        performance_metrics = await self.validate_model_performance()
        
        # Phase 4: Live Inference Demo
        print("🔮 Running live inference demo...")
        await self.demonstrate_live_inference()
        
        # Phase 5: Results Summary
        self.display_comprehensive_results()
```

2. **FastAPI Demo Endpoints**
```python
@app.post("/demo/binary-classification/train")
async def start_training_demo(
    request: BinaryClassificationDemoRequest
) -> BinaryClassificationDemoResponse:
    """Start binary classification training demo."""
    demo = BinaryClassificationDemo(
        symbols=request.symbols,
        training_samples=request.training_samples,
        enable_mlx=request.enable_mlx
    )
    
    results = await demo.run_complete_training()
    
    return BinaryClassificationDemoResponse(
        training_metrics=results.metrics,
        model_performance=results.performance,
        apple_silicon_optimization=results.mlx_metrics
    )

@app.websocket("/demo/binary-classification/live-training")
async def live_training_updates(websocket: WebSocket):
    """WebSocket endpoint for real-time training progress."""
    await websocket.accept()
    
    async for progress_update in training_progress_stream():
        await websocket.send_json({
            "phase": progress_update.phase,
            "progress_percent": progress_update.progress,
            "current_metric": progress_update.current_metric,
            "apple_silicon_utilization": progress_update.gpu_utilization
        })
```

## 5. Validation Framework

### Unit Testing Requirements
```python
@pytest.mark.unit
class TestBinaryClassificationDemo:
    async def test_data_generation(self):
        """Test realistic training data generation."""
        demo = BinaryClassificationDemo()
        training_data = await demo.generate_training_data()
        
        # Validate data quality
        assert len(training_data.training_points) >= 800  # 80% training
        assert len(training_data.validation_points) >= 200  # 20% validation
        assert all(len(tp.feature_vector.features) >= 37 for tp in training_data.training_points)
        
        # Validate label distribution (should be roughly balanced)
        buy_labels = sum(1 for tp in training_data.training_points if tp.label == BinaryClassificationLabel.BUY)
        label_ratio = buy_labels / len(training_data.training_points)
        assert 0.3 <= label_ratio <= 0.7  # Reasonable balance
    
    async def test_mlx_acceleration(self):
        """Test MLX acceleration provides performance benefits."""
        demo = BinaryClassificationDemo(training_samples=100)
        
        # Train with CPU
        cpu_start = time.time()
        cpu_results = await demo.train_model(use_mlx=False)
        cpu_time = time.time() - cpu_start
        
        # Train with MLX
        mlx_start = time.time()
        mlx_results = await demo.train_model(use_mlx=True)
        mlx_time = time.time() - mlx_start
        
        # Validate speedup
        assert mlx_time < cpu_time * 0.8  # At least 20% improvement
        assert mlx_results.accuracy >= cpu_results.accuracy - 0.05  # Similar accuracy
    
    async def test_inference_latency(self):
        """Test inference meets latency requirements."""
        demo = BinaryClassificationDemo()
        await demo.train_model()
        
        # Test single prediction latency
        feature_vector = await demo.generate_sample_feature_vector()
        
        latencies = []
        for _ in range(100):
            start_time = time.time()
            prediction = await demo.predict(feature_vector)
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[95]
        
        assert avg_latency < 10.0  # Average under 10ms
        assert p95_latency < 15.0   # 95th percentile under 15ms
```

### Integration Testing Requirements
```python
@pytest.mark.integration
class TestBinaryClassificationIntegration:
    async def test_end_to_end_demo(self):
        """Test complete end-to-end demo pipeline."""
        demo = InteractiveBinaryClassificationDemo()
        
        # Run complete demo
        results = await demo.run_complete_demo()
        
        # Validate all phases completed successfully
        assert results.data_generation_success
        assert results.training_success
        assert results.performance_validation_success
        assert results.inference_demo_success
        
        # Validate performance targets met
        assert results.training_time_seconds < 60.0
        assert results.model_accuracy > 0.80
        assert results.inference_latency_ms < 10.0
    
    async def test_websocket_live_updates(self):
        """Test WebSocket live training updates."""
        # Connect to WebSocket endpoint
        uri = "ws://localhost:8000/demo/binary-classification/live-training"
        
        progress_updates = []
        async with websockets.connect(uri) as websocket:
            # Start training in background
            asyncio.create_task(start_demo_training())
            
            # Collect progress updates
            async for message in websocket:
                update = json.loads(message)
                progress_updates.append(update)
                
                if update["progress_percent"] >= 100:
                    break
        
        # Validate we received comprehensive progress updates
        assert len(progress_updates) > 10
        assert any(update["phase"] == "data_generation" for update in progress_updates)
        assert any(update["phase"] == "model_training" for update in progress_updates)
        assert any(update["phase"] == "performance_validation" for update in progress_updates)
```

## 6. Error Handling Strategy

### Expected Error Scenarios
- [x] **Insufficient Training Data**: Handle cases with <100 samples gracefully
- [x] **MLX Hardware Unavailable**: Automatic fallback to CPU-based training
- [x] **Memory Constraints**: Adaptive batch sizing and memory monitoring
- [x] **Model Convergence Issues**: Alternative hyperparameter strategies
- [x] **Feature Computation Failures**: Robust error recovery and data validation

### Error Response Patterns
```python
class BinaryClassificationDemoError(Exception):
    """Base exception for demo errors."""
    pass

class InsufficientDataError(BinaryClassificationDemoError):
    """Raised when not enough training data is available."""
    pass

class MLXUnavailableError(BinaryClassificationDemoError):
    """Raised when MLX acceleration is requested but unavailable."""
    pass

async def robust_training_execution(self) -> TrainingResults:
    """Execute training with comprehensive error handling."""
    try:
        # Attempt MLX-accelerated training
        results = await self.train_with_mlx()
        return results
        
    except MLXUnavailableError:
        logger.warning("MLX unavailable, falling back to CPU training")
        results = await self.train_with_cpu()
        results.mlx_acceleration_used = False
        return results
        
    except InsufficientDataError as e:
        logger.error(f"Insufficient training data: {e}")
        # Generate additional synthetic data
        await self.augment_training_data()
        return await self.train_with_cpu()
        
    except MemoryError:
        logger.warning("Memory constraints detected, reducing batch size")
        self.reduce_batch_size()
        return await self.train_with_reduced_memory()
```

## 7. Configuration Management

### Required Settings
```python
class BinaryClassificationDemoConfig(BaseSettings):
    # Demo-specific configuration
    demo_enable_binary_classification: bool = Field(default=True)
    demo_training_samples: int = Field(default=1000, ge=100, le=10000)
    demo_validation_split: float = Field(default=0.2, gt=0.0, lt=1.0)
    demo_symbols: List[str] = Field(default=["AAPL", "TSLA", "NVDA"])
    
    # Model training parameters
    demo_accuracy_threshold: float = Field(default=0.80, ge=0.5, le=1.0)
    demo_max_training_time_seconds: float = Field(default=60.0, gt=0)
    demo_inference_timeout_ms: float = Field(default=10.0, gt=0)
    
    # Apple Silicon specific settings
    demo_enable_mlx_acceleration: bool = Field(default=True)
    demo_mlx_device_id: int = Field(default=0, ge=0)
    demo_target_gpu_utilization: float = Field(default=0.70, ge=0.1, le=1.0)
    
    # Label generation strategy
    demo_buy_threshold_percent: float = Field(default=0.02, ge=0.001, le=0.1)
    demo_lookforward_periods: int = Field(default=5, ge=1, le=20)
    
    class Config:
        env_prefix = "BINARY_DEMO_"
```

### Environment Variables
```bash
# Demo configuration
export BINARY_DEMO_ENABLE_BINARY_CLASSIFICATION=true
export BINARY_DEMO_TRAINING_SAMPLES=1000
export BINARY_DEMO_SYMBOLS="AAPL,TSLA,NVDA,MSFT"

# Performance targets
export BINARY_DEMO_ACCURACY_THRESHOLD=0.80
export BINARY_DEMO_MAX_TRAINING_TIME_SECONDS=60.0
export BINARY_DEMO_INFERENCE_TIMEOUT_MS=10.0

# Apple Silicon optimization
export BINARY_DEMO_ENABLE_MLX_ACCELERATION=true
export BINARY_DEMO_TARGET_GPU_UTILIZATION=0.70
```

## 8. Monitoring and Metrics

### Prometheus Metrics
```python
# Demo-specific metrics
BINARY_DEMO_TRAINING_REQUESTS_TOTAL = Counter(
    'binary_demo_training_requests_total',
    'Total binary classification demo training requests'
)

BINARY_DEMO_TRAINING_DURATION = Histogram(
    'binary_demo_training_duration_seconds',
    'Binary classification training duration'
)

BINARY_DEMO_MODEL_ACCURACY = Gauge(
    'binary_demo_model_accuracy',
    'Current model accuracy'
)

BINARY_DEMO_INFERENCE_LATENCY = Histogram(
    'binary_demo_inference_latency_milliseconds',
    'Inference latency for binary classification'
)

BINARY_DEMO_GPU_UTILIZATION = Gauge(
    'binary_demo_gpu_utilization_percent',
    'GPU utilization during demo operations'
)

BINARY_DEMO_PREDICTION_CONFIDENCE = Histogram(
    'binary_demo_prediction_confidence',
    'Distribution of prediction confidence scores'
)
```

### Health Checks
- [x] Demo service availability endpoint
- [x] MLX hardware status and performance check
- [x] Model accuracy and performance metrics validation
- [x] Memory usage and thermal monitoring

## 9. Documentation Requirements

### Code Documentation
- [x] Context7 format docstrings for all demo functions
- [x] Performance characteristics documented for each component
- [x] Apple Silicon specific optimizations clearly documented
- [x] Binary classification strategy and rationale documented

### Demo Documentation
```python
"""
Binary Classification Demo for MLX Trading Pipeline

CONTEXT: End-to-end demonstration of binary classification capabilities
SCOPE: Training, inference, and performance validation on Apple Silicon
PERFORMANCE: <60s training, <10ms inference, >80% accuracy
OPTIMIZATION: MLX GPU acceleration, unified memory utilization

This demo showcases:
1. Realistic feature generation using 37+ technical indicators
2. Intelligent binary labeling based on future price movements  
3. MLX-accelerated model training with LightGBM Metal backend
4. Sub-10ms inference with confidence scoring
5. Real-time performance monitoring and validation

Usage:
    demo = BinaryClassificationDemo(
        symbols=["AAPL", "TSLA", "NVDA"],
        training_samples=1000,
        enable_mlx=True
    )
    results = await demo.run_complete_demo()
"""
```

### User Documentation
- [x] Interactive demo usage guide with examples
- [x] Binary classification strategy explanation
- [x] Apple Silicon optimization benefits documentation
- [x] Performance tuning and troubleshooting guide

## 10. Acceptance Criteria

### Functional Requirements
- [x] Generates realistic training data with balanced binary labels
- [x] Trains binary classification model with >80% accuracy
- [x] Provides sub-10ms inference with confidence scoring
- [x] Demonstrates MLX acceleration benefits with >2x speedup
- [x] Includes interactive demo interface with real-time progress

### Performance Requirements
- [x] Complete training demo finishes in <60 seconds
- [x] MLX acceleration shows measurable performance improvement
- [x] Memory usage stays within 4GB during training
- [x] GPU utilization exceeds 70% during training phases
- [x] Inference latency consistently under 10ms

### Quality Requirements
- [x] >95% test coverage for all demo components
- [x] All integration tests passing including WebSocket functionality
- [x] No performance regressions in existing pipeline components
- [x] Code follows established MLX Trading Pipeline patterns

### Demo Requirements
- [x] Interactive demo script provides clear progress feedback
- [x] WebSocket endpoint enables real-time training monitoring
- [x] FastAPI endpoints support programmatic demo execution
- [x] Comprehensive results display with performance metrics

## 11. Risk Mitigation

### Technical Risks
- [x] **Risk**: Binary classification may not achieve target accuracy with synthetic data
  - **Mitigation**: Implement multiple labeling strategies and validate with historical data
  
- [x] **Risk**: MLX acceleration may not provide expected 2x speedup
  - **Mitigation**: Optimize MLX array operations and compare with CPU baseline
  
- [x] **Risk**: Demo may consume excessive memory on Apple Silicon devices
  - **Mitigation**: Implement adaptive memory management and monitoring

### Performance Risks
- [x] **Risk**: Training time may exceed 60-second target with large datasets
  - **Mitigation**: Implement progressive training and early stopping
  
- [x] **Risk**: Inference latency may degrade under sustained load
  - **Mitigation**: Implement inference batching and thermal monitoring

## 12. Success Validation

### Implementation Validation Checklist
- [ ] Binary classification demo generates realistic training data
- [ ] Model training achieves >80% accuracy consistently  
- [ ] MLX acceleration provides >2x speedup vs CPU training
- [ ] Inference latency stays <10ms for individual predictions
- [ ] Interactive demo provides comprehensive progress feedback
- [ ] WebSocket updates work for real-time training monitoring
- [ ] All performance targets met on Apple Silicon hardware
- [ ] Memory usage stays within 4GB limits during operation
- [ ] GPU utilization exceeds 70% during training phases
- [ ] Demo documentation is complete and user-friendly

### Post-Implementation Validation
- [ ] Demo successfully showcases MLX Trading Pipeline capabilities
- [ ] Performance benchmarks validate Apple Silicon optimization benefits
- [ ] User feedback confirms demo effectively demonstrates system capabilities
- [ ] Integration with existing pipeline components works seamlessly

This PRP provides a comprehensive blueprint for implementing a binary classification training demo that showcases the MLX Trading Pipeline's capabilities while meeting all Apple Silicon performance targets and maintaining code quality standards.