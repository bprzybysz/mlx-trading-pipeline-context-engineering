# Execute PRP Command

## Purpose
Execute a Product Requirements Prompt (PRP) with comprehensive validation and Apple Silicon optimization for the MLX Trading Pipeline.

## Usage
This command implements a feature following a detailed PRP blueprint with built-in validation and performance monitoring.

## Command Structure
```
/execute-prp [prp-file] [validation-level] [benchmark-mode]
```

## Parameters
- **prp-file**: Path to PRP file (e.g., "PRPs/risk-management-prp.md")
- **validation-level**: Testing rigor (basic, standard, comprehensive)
- **benchmark-mode**: Performance validation (development, production, stress)

## Execution Process

### Phase 1: PRP Validation
- Parse PRP for completeness and consistency
- Validate Apple Silicon optimization requirements
- Check performance targets are achievable
- Confirm integration points are valid

### Phase 2: Implementation Planning
- Break down PRP into implementable tasks
- Identify dependencies and prerequisites
- Plan Apple Silicon optimization integration
- Schedule performance validation checkpoints

### Phase 3: Iterative Implementation
- Implement following PRP specifications exactly
- Apply Apple Silicon patterns from examples/
- Validate each step against PRP criteria
- Measure performance at each checkpoint

### Phase 4: Comprehensive Validation
- Execute complete test suite per PRP requirements
- Run Apple Silicon performance benchmarks
- Validate integration with existing pipeline
- Confirm all acceptance criteria are met

## Implementation Strategies

### Apple Silicon Optimization
```python
# Follow established MLX patterns
async def implement_with_mlx_optimization(self, data: MarketData):
    """
    CONTEXT: Implement feature with Apple Silicon optimization
    INPUT: Market data requiring processing
    OUTPUT: Optimized results meeting PRP performance targets
    PERFORMANCE: Must meet PRP-specified latency requirements
    """
    # Use MLX for GPU acceleration
    with mx.stream(mx.gpu):
        result = await self._mlx_accelerated_computation(data)
        mx.eval(result)  # Force evaluation for timing
    
    return result
```

### Performance Validation
```python
@pytest.mark.benchmark
async def test_prp_performance_requirements(self):
    """Validate implementation meets all PRP performance targets"""
    # Execute with realistic data load
    start_time = time.time()
    result = await feature.process_batch(test_data_10k_samples)
    execution_time = time.time() - start_time
    
    # Validate PRP performance requirements
    assert execution_time < PRP_LATENCY_TARGET
    assert result.throughput > PRP_THROUGHPUT_TARGET
    assert get_gpu_utilization() > PRP_GPU_UTILIZATION_TARGET
```

### Integration Validation  
```python
@pytest.mark.integration
async def test_prp_integration_requirements(self):
    """Validate feature integrates per PRP specifications"""
    # Test with existing pipeline components
    pipeline = TradingPipeline(enable_new_feature=True)
    
    # Validate integration points specified in PRP
    assert pipeline.feature_engine.supports_new_feature()
    assert pipeline.model_trainer.can_use_new_features()
    assert pipeline.inference_service.includes_new_predictions()
```

## Execution Modes

### Development Mode
- Basic validation with mock data
- Performance estimates rather than full benchmarks
- Abbreviated test suite for rapid iteration
- Development-grade Apple Silicon optimization

**Usage**: `/execute-prp risk-management-prp.md basic development`

### Standard Mode  
- Complete test suite execution
- Full performance validation on Apple Silicon
- Integration testing with realistic data
- Production-grade optimization validation

**Usage**: `/execute-prp risk-management-prp.md standard production`

### Comprehensive Mode
- Exhaustive testing including edge cases
- Stress testing under sustained load
- Thermal and power management validation
- Complete documentation generation

**Usage**: `/execute-prp risk-management-prp.md comprehensive stress`

## Validation Framework

### Unit Testing Requirements
```python
# Ensure >95% code coverage per PRP
@pytest.mark.unit
class TestPRPImplementation:
    async def test_core_functionality(self):
        # Test all PRP-specified functionality
    
    async def test_apple_silicon_optimization(self):
        # Validate MLX acceleration works correctly
    
    async def test_error_handling(self):
        # Test all PRP-specified error scenarios
    
    async def test_performance_characteristics(self):
        # Validate performance meets PRP targets
```

### Integration Testing Framework
```python
# Test with complete pipeline as per PRP
@pytest.mark.integration  
class TestPRPIntegration:
    async def test_end_to_end_workflow(self):
        # Complete pipeline with new feature
    
    async def test_data_flow_integration(self):
        # Validate data flows as specified in PRP
    
    async def test_api_integration(self):
        # Test API endpoints work with new feature
```

### Performance Benchmarking
```python
# Apple Silicon specific benchmarks per PRP
@pytest.mark.benchmark
class TestAppleSiliconPerformance:
    def test_mlx_acceleration_improvement(self):
        # Compare CPU vs MLX performance
    
    def test_unified_memory_efficiency(self):
        # Validate memory usage optimization
    
    def test_thermal_sustainability(self):
        # Ensure sustained performance under load
```

## Quality Gates

### Implementation Quality Gates
1. **Code Coverage**: >95% as specified in PRP
2. **Type Safety**: All Pydantic models properly defined
3. **Apple Silicon**: MLX optimization implemented correctly
4. **Performance**: All PRP targets met or exceeded

### Integration Quality Gates  
1. **Pipeline Compatibility**: No breaking changes to existing functionality
2. **API Consistency**: All endpoints follow established patterns
3. **Data Flow**: Proper integration with feature engine and model trainer
4. **Monitoring**: Comprehensive metrics collection implemented

### Performance Quality Gates
1. **Latency**: Must meet PRP-specified latency targets
2. **Throughput**: Must achieve PRP-specified throughput requirements
3. **Resource Usage**: Memory and GPU utilization within PRP limits
4. **Sustainability**: Performance maintained under sustained load

## Error Handling and Recovery

### Implementation Failures
- Detailed error reporting with PRP requirement mapping
- Automatic rollback to previous stable state
- Comprehensive failure analysis and recommendations
- Alternative implementation strategies if needed

### Performance Failures
- Detailed benchmarking results vs PRP targets
- Performance optimization recommendations
- Apple Silicon tuning suggestions
- Resource usage analysis and optimization paths

### Integration Failures
- Component compatibility analysis
- Data flow validation results
- API integration test results
- Recommended integration pattern adjustments

## Success Criteria

### Feature Implementation Success
- [ ] All PRP functional requirements implemented
- [ ] Apple Silicon optimization patterns applied
- [ ] Performance targets met or exceeded
- [ ] Integration with existing pipeline validated
- [ ] Comprehensive test suite passing
- [ ] Documentation updated per PRP requirements

### Quality Validation Success
- [ ] >95% test coverage achieved
- [ ] All Apple Silicon benchmarks passed
- [ ] No regression in existing functionality
- [ ] All error scenarios properly handled
- [ ] Monitoring and metrics properly implemented
- [ ] User documentation completed

## Post-Execution Validation

### Implementation Review
1. Validate all PRP requirements are met
2. Confirm Apple Silicon optimizations are effective
3. Review performance benchmarks vs targets
4. Validate integration points work correctly

### Documentation Update
1. Update architecture documentation with changes
2. Add new feature documentation to user guides
3. Update troubleshooting guides with new scenarios
4. Record lessons learned for future PRP improvements

### Performance Monitoring
1. Establish baseline metrics for new feature
2. Set up alerts for performance degradation
3. Monitor Apple Silicon resource utilization
4. Track thermal and power management impact