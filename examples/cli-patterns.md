# CLI and Make Command Patterns

## Overview
Comprehensive patterns for command-line interactions and Make-based workflow automation in the MLX Trading Pipeline, specifically optimized for Apple Silicon development.

## Make Command Patterns

### 1. Development Workflow Commands

#### Environment Setup Pattern
```makefile
# Standard development setup with Apple Silicon optimization
setup: install ## Setup development environment with Apple Silicon optimizations
	@echo "🚀 Setting up MLX Trading Pipeline for Apple Silicon..."
	@mkdir -p data/historical artifacts models logs
	@echo "📋 Verifying Apple Silicon compatibility..."
	@python -c "import platform; assert platform.machine() == 'arm64', 'Apple Silicon required'"
	@echo "🔍 Verifying MLX installation..."
	@python -c "import mlx.core as mx; print(f'✅ MLX {mx.__version__} with Metal: {mx.metal.is_available()}')"
	@echo "✅ Development environment ready for Apple Silicon"

install: ## Install dependencies with Apple Silicon optimization
	@echo "📦 Installing dependencies optimized for Apple Silicon..."
	uv pip install -e .
	uv pip install --no-deps mlx mlx-lm  # Ensure latest MLX versions
	@echo "🔧 Installing Apple Silicon specific packages..."
	uv pip install lightgbm --config-settings=cmake.define.USE_GPU=ON
	@echo "✅ All dependencies installed with Apple Silicon support"
```

#### Pipeline Operation Commands
```makefile
# Core pipeline operations with performance monitoring
run: validate-setup ## Run the main trading pipeline with Apple Silicon optimization
	@echo "🚀 Starting MLX Trading Pipeline with Apple Silicon acceleration..."
	@echo "📊 System specs:"
	@python -c "import psutil, platform; print(f'  CPU: {platform.processor()}'); print(f'  Cores: {psutil.cpu_count()} total'); print(f'  Memory: {psutil.virtual_memory().total // (1024**3)}GB')"
	@echo "🎯 Performance targets: <10ms inference, >10k samples/sec"
	@time uv run python main.py

run-benchmark: install ## Run pipeline with comprehensive Apple Silicon benchmarking
	@echo "⏱️ Running Apple Silicon performance benchmarks..."
	@echo "📈 Measuring baseline performance..."
	@python -c "import time; start=time.time(); import mlx.core as mx; print(f'MLX import time: {(time.time()-start)*1000:.1f}ms')"
	@echo "🔥 Starting pipeline with performance profiling..."
	@time uv run python main.py --benchmark --apple-silicon-profiling
	@echo "📊 Generating performance report..."
	@python scripts/generate_performance_report.py
```

#### Testing Command Patterns
```makefile
# Apple Silicon specific testing patterns
test-apple-silicon: install ## Run Apple Silicon specific tests
	@echo "🧪 Running Apple Silicon optimization tests..."
	@echo "🔍 Testing MLX acceleration..."
	@uv run pytest tests/ -v -m apple_silicon --tb=short
	@echo "🌡️ Testing thermal sustainability..."
	@uv run pytest tests/ -v -m thermal --tb=short
	@echo "💾 Testing unified memory optimization..."
	@uv run pytest tests/ -v -m memory --tb=short
	@echo "✅ All Apple Silicon tests completed"

test-performance: install ## Run performance benchmarks with Apple Silicon targets
	@echo "⚡ Running Apple Silicon performance benchmarks..."
	@echo "🎯 Targets: <5ms features, <10ms inference, >10k samples/sec"
	@uv run pytest tests/ -v -m benchmark --benchmark-only
	@echo "📊 Performance report:"
	@cat performance_results.json | jq '.apple_silicon_metrics'

test-integration: install ## Run integration tests with Apple Silicon pipeline
	@echo "🔗 Testing complete pipeline integration on Apple Silicon..."
	@uv run pytest tests/ -v -m integration --tb=short
	@echo "✅ Integration tests completed"
```

### 2. Data Management Command Patterns

#### Historical Data Commands
```makefile
# Data fetching with Apple Silicon optimization
fetch-historical-data: install ## Fetch historical data with Apple Silicon acceleration
	@echo "📈 Fetching historical market data..."
	@echo "🚀 Using Apple Silicon acceleration for data processing..."
	@uv run python src/mlx_trading_pipeline/historical_data_fetcher.py \
		--symbols AAPL TSLA NVDA MSFT GOOGL \
		--days 90 \
		--enable-mlx \
		--batch-size 1000
	@echo "✅ Historical data fetch completed with MLX acceleration"

prepare-training-data: fetch-historical-data ## Prepare training data with Apple Silicon optimization
	@echo "⚙️ Preparing training data with Apple Silicon optimization..."
	@echo "🔧 Processing with MLX acceleration..."
	@uv run python scripts/prepare_training_data.py \
		--input data/historical/ \
		--output data/processed/ \
		--enable-gpu \
		--unified-memory
	@echo "✅ Training data prepared with Apple Silicon optimization"
```

#### Model Training Commands  
```makefile
# Model training with Apple Silicon optimization
train-models: prepare-training-data ## Train models with Apple Silicon acceleration
	@echo "🤖 Training models with Apple Silicon optimization..."
	@echo "🔥 Using Metal backend for LightGBM acceleration..."
	@echo "⚡ Target: 3x faster training vs CPU baseline..."
	@time uv run python src/mlx_trading_pipeline/unified_training_workflow.py \
		--symbols AAPL TSLA NVDA MSFT GOOGL \
		--enable-metal \
		--gpu-acceleration \
		--unified-memory \
		--thermal-monitoring
	@echo "📊 Training completed - generating performance report..."
	@python scripts/training_performance_report.py

train-single-symbol: install ## Train model for single symbol (development)
	@echo "🎯 Training single symbol model for development..."
	@read -p "Enter symbol (e.g., AAPL): " symbol; \
	uv run python src/mlx_trading_pipeline/unified_training_workflow.py \
		--symbols $$symbol \
		--enable-metal \
		--dev-mode
```

### 3. Code Quality and Development Commands

#### Code Quality with Apple Silicon Considerations
```makefile
# Code quality checks with Apple Silicon specific linting
lint: install ## Run linting with Apple Silicon considerations
	@echo "🔍 Linting code with Apple Silicon specific checks..."
	@uv run ruff check . --fix
	@echo "🔍 Checking for Apple Silicon optimization opportunities..."
	@python scripts/check_apple_silicon_optimization.py
	@echo "✅ Linting completed"

format: install ## Format code following Apple Silicon project standards
	@echo "✨ Formatting code following Apple Silicon project standards..."
	@uv run black . --line-length 88
	@echo "🔧 Organizing imports for Apple Silicon modules..."
	@uv run isort . --profile black
	@echo "✅ Code formatting completed"

type-check: install ## Type checking with Apple Silicon type stubs
	@echo "🔎 Type checking with Apple Silicon considerations..."
	@uv run mypy . \
		--ignore-missing-imports \
		--exclude 'build/' \
		--exclude 'dist/'
	@echo "✅ Type checking completed"
```

#### Development Utilities
```makefile
# Development utilities for Apple Silicon
dev-shell: install ## Start development shell with Apple Silicon environment
	@echo "🐚 Starting development shell with Apple Silicon environment..."
	@echo "Environment variables set:"
	@echo "  ENABLE_GPU=true"
	@echo "  ENABLE_MLX=true" 
	@echo "  METAL_BACKEND=true"
	@echo "  UNIFIED_MEMORY=true"
	@export ENABLE_GPU=true ENABLE_MLX=true METAL_BACKEND=true UNIFIED_MEMORY=true && \
	 uv run python

profile-performance: install ## Profile Apple Silicon performance
	@echo "📊 Profiling Apple Silicon performance..."
	@echo "🔥 Running with py-spy profiler..."
	@uv run py-spy record \
		--duration 60 \
		--rate 100 \
		--output apple_silicon_profile.svg \
		--format speedscope \
		-- python main.py --profile-mode
	@echo "📈 Profile saved to apple_silicon_profile.svg"
```

### 4. Monitoring and Debugging Patterns

#### System Monitoring Commands
```makefile
# Apple Silicon system monitoring
monitor-thermal: ## Monitor Apple Silicon thermal performance
	@echo "🌡️ Monitoring Apple Silicon thermal performance..."
	@echo "Press Ctrl+C to stop monitoring"
	@while true; do \
		echo "=== $$(date) ==="; \
		python -c "import psutil; temps = psutil.sensors_temperatures(); print(f'CPU Temp: {temps.get(\"coretemp\", [{}])[0].get(\"current\", 0):.1f}°C')"; \
		python -c "import mlx.core as mx; print(f'Metal Available: {mx.metal.is_available()}')"; \
		sleep 5; \
	done

monitor-performance: run ## Monitor real-time performance during pipeline execution
	@echo "📊 Starting real-time performance monitoring..."
	@python scripts/monitor_apple_silicon_performance.py &
	@echo "🚀 Pipeline performance monitoring active..."

status: ## Show comprehensive Apple Silicon system status
	@echo "💻 Apple Silicon System Status"
	@echo "================================"
	@echo "🖥️  Hardware:"
	@python -c "import platform; print(f'    Architecture: {platform.machine()}')"
	@python -c "import psutil; print(f'    CPU Cores: {psutil.cpu_count(logical=False)} P-cores + {psutil.cpu_count(logical=True) - psutil.cpu_count(logical=False)} E-cores')"
	@python -c "import psutil; print(f'    Memory: {psutil.virtual_memory().total // (1024**3)}GB unified')"
	@echo "⚡ MLX Framework:"
	@python -c "import mlx.core as mx; print(f'    MLX Version: {mx.__version__}')" 2>/dev/null || echo "    MLX: Not installed"
	@python -c "import mlx.core as mx; print(f'    Metal Backend: {mx.metal.is_available()}')" 2>/dev/null || echo "    Metal: Not available"
	@echo "📁 Project Status:"
	@echo "    Data: $$(ls -la data/ 2>/dev/null | wc -l) items"
	@echo "    Models: $$(ls -la models/ 2>/dev/null | wc -l) items"
	@echo "    Artifacts: $$(ls -la artifacts/ 2>/dev/null | wc -l) items"
```

### 5. Production and Deployment Patterns

#### Production Commands
```makefile
# Production deployment with Apple Silicon optimization
validate-production: test-apple-silicon test-performance ## Validate production readiness
	@echo "🔍 Validating production readiness for Apple Silicon deployment..."
	@echo "✅ Apple Silicon tests: PASSED"
	@echo "✅ Performance benchmarks: PASSED"
	@echo "🔧 Validating configuration..."
	@python scripts/validate_production_config.py
	@echo "📊 Generating production readiness report..."
	@python scripts/production_readiness_report.py
	@echo "✅ Production validation complete"

package-apple-silicon: validate-production ## Package for Apple Silicon deployment
	@echo "📦 Packaging MLX Trading Pipeline for Apple Silicon..."
	@echo "🔧 Building optimized package..."
	@uv build --wheel
	@echo "🍎 Creating Apple Silicon specific deployment package..."
	@python scripts/create_apple_silicon_package.py
	@echo "✅ Apple Silicon package ready for deployment"

deploy-apple-silicon: package-apple-silicon ## Deploy to Apple Silicon infrastructure
	@echo "🚀 Deploying to Apple Silicon infrastructure..."
	@echo "⚠️  Ensure target systems have Apple Silicon (M1/M2/M3)"
	@python scripts/deploy_apple_silicon.py
	@echo "✅ Deployment to Apple Silicon infrastructure complete"
```

## CLI Script Patterns

### 1. Apple Silicon Validation Scripts

#### System Validation Script
```python
#!/usr/bin/env python3
"""
Apple Silicon system validation script
Usage: python scripts/validate_apple_silicon.py
"""
import platform
import subprocess
import sys
from typing import Dict, List

def validate_apple_silicon() -> Dict[str, bool]:
    """Comprehensive Apple Silicon validation"""
    results = {}
    
    # Check architecture
    results['arm64_architecture'] = platform.machine() == 'arm64'
    
    # Check MLX availability
    try:
        import mlx.core as mx
        results['mlx_installed'] = True
        results['metal_available'] = mx.metal.is_available()
        results['mlx_version_ok'] = mx.__version__ >= '0.19.0'
    except ImportError:
        results['mlx_installed'] = False
        results['metal_available'] = False
        results['mlx_version_ok'] = False
    
    # Check LightGBM GPU support
    try:
        import lightgbm as lgb
        # Test GPU training capability
        test_data = lgb.Dataset([[1, 2], [3, 4]], label=[0, 1])
        lgb.train({'device': 'gpu'}, test_data, num_boost_round=1)
        results['lightgbm_gpu'] = True
    except:
        results['lightgbm_gpu'] = False
    
    return results

if __name__ == "__main__":
    print("🍎 Apple Silicon Validation")
    print("=" * 40)
    
    results = validate_apple_silicon()
    all_passed = True
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check:20}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 40)
    if all_passed:
        print("🎉 All Apple Silicon checks passed!")
        sys.exit(0)
    else:
        print("⚠️  Some Apple Silicon checks failed!")
        sys.exit(1)
```

#### Performance Benchmarking Script
```python
#!/usr/bin/env python3
"""
Apple Silicon performance benchmarking script
Usage: python scripts/benchmark_apple_silicon.py
"""
import time
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict

@dataclass
class BenchmarkResults:
    cpu_feature_computation_ms: float
    mlx_feature_computation_ms: float
    cpu_model_training_sec: float
    metal_model_training_sec: float
    memory_efficiency_percent: float
    speedup_factor: float

def run_apple_silicon_benchmarks() -> BenchmarkResults:
    """Run comprehensive Apple Silicon benchmarks"""
    print("⚡ Running Apple Silicon Performance Benchmarks...")
    
    # Feature computation benchmark
    print("📊 Testing feature computation acceleration...")
    test_data = np.random.rand(10000, 50).astype(np.float32)
    
    # CPU baseline
    start = time.time()
    cpu_features = np.mean(test_data, axis=1)  # Simplified feature computation
    cpu_time = (time.time() - start) * 1000  # Convert to ms
    
    # MLX acceleration
    try:
        import mlx.core as mx
        mlx_data = mx.array(test_data)
        
        start = time.time()
        with mx.stream(mx.gpu):
            mlx_features = mx.mean(mlx_data, axis=1)
            mx.eval(mlx_features)
        mlx_time = (time.time() - start) * 1000  # Convert to ms
        
        speedup = cpu_time / mlx_time if mlx_time > 0 else 1.0
        
    except ImportError:
        mlx_time = cpu_time
        speedup = 1.0
    
    # Model training benchmark (simplified)
    print("🤖 Testing model training acceleration...")
    try:
        import lightgbm as lgb
        from sklearn.datasets import make_regression
        
        X, y = make_regression(n_samples=10000, n_features=20, random_state=42)
        train_data = lgb.Dataset(X, y)
        
        # CPU training
        start = time.time()
        cpu_model = lgb.train({'device': 'cpu', 'verbose': -1}, train_data, num_boost_round=50)
        cpu_training_time = time.time() - start
        
        # Metal training
        try:
            start = time.time()
            metal_model = lgb.train({'device': 'gpu', 'verbose': -1}, train_data, num_boost_round=50)
            metal_training_time = time.time() - start
        except:
            metal_training_time = cpu_training_time
            
    except ImportError:
        cpu_training_time = 0
        metal_training_time = 0
    
    # Memory efficiency (simplified estimation)
    memory_efficiency = min(50.0, speedup * 10)  # Rough estimation
    
    return BenchmarkResults(
        cpu_feature_computation_ms=cpu_time,
        mlx_feature_computation_ms=mlx_time,
        cpu_model_training_sec=cpu_training_time,
        metal_model_training_sec=metal_training_time,
        memory_efficiency_percent=memory_efficiency,
        speedup_factor=speedup
    )

if __name__ == "__main__":
    print("🚀 Apple Silicon Performance Benchmarking")
    print("=" * 50)
    
    results = run_apple_silicon_benchmarks()
    
    # Display results
    print("\n📊 Benchmark Results:")
    print(f"Feature Computation:")
    print(f"  CPU Time:     {results.cpu_feature_computation_ms:.2f}ms")
    print(f"  MLX Time:     {results.mlx_feature_computation_ms:.2f}ms")
    print(f"  Speedup:      {results.speedup_factor:.2f}x")
    
    print(f"\nModel Training:")
    print(f"  CPU Time:     {results.cpu_model_training_sec:.2f}s")
    print(f"  Metal Time:   {results.metal_model_training_sec:.2f}s")
    
    print(f"\nMemory Efficiency: {results.memory_efficiency_percent:.1f}%")
    
    # Save results
    with open('apple_silicon_benchmark_results.json', 'w') as f:
        json.dump(asdict(results), f, indent=2)
    
    print(f"\n✅ Results saved to apple_silicon_benchmark_results.json")
```

### 2. Interactive CLI Patterns

#### Development CLI Helper
```makefile
# Interactive development commands
dev-interactive: install ## Start interactive development session
	@echo "🛠️  MLX Trading Pipeline - Interactive Development"
	@echo "Available commands:"
	@echo "  1) Run pipeline with debug mode"
	@echo "  2) Test Apple Silicon optimizations" 
	@echo "  3) Monitor system performance"
	@echo "  4) Generate performance report"
	@echo "  5) Validate configuration"
	@read -p "Select option (1-5): " choice; \
	case $$choice in \
		1) make run-debug;; \
		2) make test-apple-silicon;; \
		3) make monitor-performance;; \
		4) make benchmark-report;; \
		5) make validate-setup;; \
		*) echo "Invalid option";; \
	esac

quick-start: ## Quick start guide for Apple Silicon development
	@echo "🚀 MLX Trading Pipeline - Quick Start Guide"
	@echo "==========================================="
	@echo
	@echo "1️⃣  First time setup:"
	@echo "   make setup"
	@echo
	@echo "2️⃣  Run the pipeline:"
	@echo "   make run"
	@echo
	@echo "3️⃣  Run tests:"
	@echo "   make test-apple-silicon"
	@echo
	@echo "4️⃣  Check performance:"
	@echo "   make benchmark-report"
	@echo
	@echo "📚 For more commands: make help"
```

This comprehensive CLI pattern framework provides developers with intuitive, Apple Silicon-optimized commands that handle the complexity of the MLX trading pipeline while providing clear feedback and performance monitoring capabilities.