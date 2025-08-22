# PRP: Optuna Hyperparameter Optimization Implementation

## Context Engineering Classification

- **Type**: Implementation PRP
- **Complexity**: High
- **Domain**: MLX Trading Pipeline Hyperparameter Optimization
- **Framework**: Apple Silicon-optimized machine learning with real-time constraints

## Problem Statement

**CONTEXT**: MLX Trading Pipeline achieves 67.6% recall with hardcoded LightGBM parameters and mock Optuna integration
**INPUT**: 30k samples of 1-minute NVDA trading data with 38 technical indicators
**OUTPUT**: Production-ready Optuna hyperparameter optimization achieving >80% recall while maintaining <10ms inference
**PERFORMANCE**: Complete optimization in <30 minutes with systematic parameter search

The current binary classification demo uses hardcoded parameters and a mock `optimize_hyperparameters()` method. Documentation analysis reveals no real hyperparameter tuning specifications exist. We need comprehensive Optuna integration to improve the 67.6% recall performance through systematic parameter optimization.

## Success Criteria

### Primary Objectives

1. **Real Optuna Integration**: Replace mock implementation with production-ready hyperparameter optimization
2. **Recall Improvement**: Achieve >80% recall (vs current 67.6%) through optimized parameters
3. **Performance Maintenance**: Maintain <10ms inference latency and Apple Silicon acceleration
4. **Search Space Definition**: Comprehensive LightGBM parameter space with MLX-specific constraints

### Performance Targets

- **Recall Improvement**: 67.6% → >80% (minimum 15% improvement)
- **Optimization Time**: Complete search in <30 minutes for 100 trials
- **Inference Latency**: Maintain <10ms requirement (currently 0.1ms)
- **Training Efficiency**: MLX acceleration preserved through optimization process

## Technical Requirements

### Core Optuna Implementation

1. **Study Creation and Management**:
   - SQLite storage for trial persistence
   - TPE sampler for efficient parameter search
   - Pruning for early termination of poor trials
   - Multi-objective optimization (recall + latency)

2. **Search Space Definition**:
   - LightGBM parameters with Apple Silicon constraints
   - MLX-specific optimization parameters
   - Time-series validation strategy
   - Feature engineering hyperparameters

3. **Objective Function Design**:
   - Primary metric: Recall optimization
   - Secondary constraints: Inference latency, training time
   - Cross-validation strategy for time-series data
   - Performance degradation detection

### Apple Silicon Integration

1. **MLX Framework Optimization**:
   - GPU memory usage parameters
   - Metal backend configuration
   - Unified memory constraints
   - Parallel trial execution

2. **Resource Management**:
   - CPU vs GPU task scheduling
   - Memory usage monitoring
   - Thermal throttling considerations
   - Power efficiency optimization

## Implementation Strategy

### Phase 1: Core Optuna Infrastructure

```python
# Replace mock implementation with real Optuna study
class OptunaBinaryClassificationOptimizer:
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )
    
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define comprehensive LightGBM search space"""
        return {
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 0, 7)
        }
```

### Phase 2: Multi-Objective Optimization

```python
# Optimize for both recall and inference latency
def objective_function(trial: optuna.Trial) -> float:
    params = self.define_search_space(trial)
    
    # Train model with suggested parameters
    model, metrics, latency = await self.train_and_evaluate(params)
    
    # Primary objective: Recall
    recall = metrics.recall
    
    # Constraint: Inference latency must be <10ms
    if latency > 10.0:  # milliseconds
        trial.set_user_attr('latency_violation', True)
        return 0.0  # Penalize high latency
    
    # Secondary objectives as attributes
    trial.set_user_attr('auc', metrics.auc)
    trial.set_user_attr('f1_score', metrics.f1_score)
    trial.set_user_attr('inference_latency', latency)
    
    return recall
```

### Phase 3: Time-Series Cross-Validation

```python
# Proper time-series validation for trading data
class TimeSeriesOptimization:
    def __init__(self, n_splits: int = 5):
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
    
    async def cross_validate_params(self, params: Dict, feature_vectors: List[FeatureVector]) -> Dict[str, float]:
        """Time-series cross-validation with temporal ordering"""
        recalls = []
        aucs = []
        latencies = []
        
        for train_idx, val_idx in self.tscv.split(feature_vectors):
            train_data = [feature_vectors[i] for i in train_idx]
            val_data = [feature_vectors[i] for i in val_idx]
            
            # Train with suggested parameters
            model = await self.train_model(train_data, params)
            
            # Evaluate on validation set
            metrics, latency = await self.evaluate_model(model, val_data)
            
            recalls.append(metrics.recall)
            aucs.append(metrics.auc)
            latencies.append(latency)
        
        return {
            'recall_mean': np.mean(recalls),
            'recall_std': np.std(recalls),
            'auc_mean': np.mean(aucs),
            'latency_mean': np.mean(latencies)
        }
```

### Phase 4: MLX-Specific Optimization

```python
# Apple Silicon-specific parameters
class MLXOptimizationConfig:
    def __init__(self):
        self.mlx_params = {
            'use_gpu': True,
            'gpu_memory_limit': 8.0,  # GB
            'batch_size_optimization': True,
            'metal_backend': True
        }
    
    def get_mlx_constraints(self, trial: optuna.Trial) -> Dict[str, Any]:
        """MLX-specific parameter constraints"""
        # Optimize batch processing for Apple Silicon
        batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048])
        
        # Memory usage optimization
        max_memory_gb = trial.suggest_float('max_memory_gb', 1.0, 8.0)
        
        # GPU utilization strategy
        gpu_strategy = trial.suggest_categorical('gpu_strategy', ['aggressive', 'balanced', 'conservative'])
        
        return {
            'batch_size': batch_size,
            'max_memory_gb': max_memory_gb,
            'gpu_strategy': gpu_strategy
        }
```

## Expected Implementation Results

### Performance Improvements

1. **Recall Enhancement**:
   - Target: 67.6% → >80% recall
   - Method: Systematic parameter optimization
   - Validation: Time-series cross-validation

2. **Latency Maintenance**:
   - Constraint: <10ms inference (currently 0.1ms)
   - Strategy: Latency-aware parameter selection
   - Monitoring: Real-time performance tracking

3. **Training Efficiency**:
   - Target: <30 minutes for 100 trials
   - Method: Pruning and parallel execution
   - Acceleration: Apple Silicon optimization

### Integration Points

1. **Binary Classification Demo**:
   - Replace hardcoded parameters in `binary_classification_demo.py`
   - Add optimization config loading
   - Integrate with existing MLX pipeline

2. **Trading Pipeline Training**:
   - Update `trading_pipeline_training.py` with real implementation
   - Add Optuna study management
   - Preserve existing model versioning

3. **Configuration Management**:
   - Add optimization configs to `configs/` directory
   - Support multiple optimization strategies
   - Environment-specific parameter ranges

## Validation Framework

### Optimization Quality Metrics

1. **Search Effectiveness**:
   - Parameter space coverage
   - Convergence rate analysis
   - Best parameter stability

2. **Performance Validation**:
   - Recall improvement verification
   - Latency constraint compliance
   - MLX acceleration preservation

3. **Production Readiness**:
   - Model robustness testing
   - Parameter sensitivity analysis
   - Deployment compatibility

### Apple Silicon Benchmarks

1. **MLX Performance**:
   - GPU utilization metrics
   - Memory usage optimization
   - Thermal efficiency

2. **Comparative Analysis**:
   - MLX vs CPU performance
   - Parameter optimization impact
   - Resource usage patterns

## Configuration Specifications

### Optimization Config Structure

```python
@dataclass
class OptimizationConfig:
    # Study configuration
    study_name: str = "lightgbm_recall_optimization"
    n_trials: int = 100
    timeout_minutes: int = 30
    
    # Objective configuration
    primary_metric: str = "recall"
    target_recall: float = 0.80
    max_inference_latency_ms: float = 10.0
    
    # Search space configuration
    param_ranges: Dict[str, Tuple] = field(default_factory=lambda: {
        'num_leaves': (10, 300),
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 15),
        'feature_fraction': (0.4, 1.0),
        'bagging_fraction': (0.4, 1.0)
    })
    
    # MLX-specific configuration
    mlx_optimization: bool = True
    apple_silicon_constraints: bool = True
    gpu_memory_limit_gb: float = 8.0
    
    # Validation configuration
    cv_splits: int = 5
    validation_method: str = "time_series"
    early_stopping_patience: int = 10
```

### Usage Integration

```python
# Integration with existing demo
async def optimize_and_train():
    config = OptimizationConfig(
        n_trials=50,
        target_recall=0.80,
        max_inference_latency_ms=10.0
    )
    
    optimizer = OptunaBinaryClassificationOptimizer(config)
    
    # Run optimization
    best_params = await optimizer.optimize(feature_vectors, labels)
    
    # Train final model with best parameters
    final_model = await demo.train_model_with_params(best_params)
    
    return final_model, best_params
```

## Success Metrics

### Quantitative Targets

- **Recall**: 67.6% → >80% (minimum 18% relative improvement)
- **AUC**: Maintain >95% (currently 96.3%)
- **F1 Score**: >75% (currently 69.7%)
- **Inference Latency**: <10ms (currently 0.1ms)
- **Optimization Time**: <30 minutes for 100 trials

### Qualitative Outcomes

- **Production Ready**: Real Optuna integration replacing mock
- **Apple Silicon Optimized**: MLX-specific parameter optimization
- **Maintainable**: Configuration-driven optimization strategy
- **Extensible**: Framework for future optimization enhancements

---

**Context Engineering Framework**: This PRP provides comprehensive specifications for production-ready Optuna hyperparameter optimization that maintains Apple Silicon performance advantages while achieving significant recall improvements through systematic parameter search.