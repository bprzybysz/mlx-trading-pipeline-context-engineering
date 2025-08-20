# Context Engineering for MLX Trading Pipeline

## Overview
This directory contains comprehensive context engineering artifacts that enable advanced AI-driven development for the Apple Silicon-optimized trading pipeline. Context engineering goes beyond prompt engineering by providing complete implementation frameworks that enable consistent, high-quality AI assistance.

## Philosophy
"Context Engineering is 10x better than prompt engineering" - by treating AI implementation like writing a complete screenplay rather than giving simple directions, we achieve:

- **Consistency**: All implementations follow established patterns
- **Quality**: Comprehensive validation ensures robust code
- **Performance**: Apple Silicon optimizations are built into every feature
- **Maintainability**: Standardized approaches reduce technical debt

## Directory Structure

```
context-engineering/
├── .claude/                    # AI assistant configuration
│   ├── rules.md               # Project-specific development rules
│   └── project-context.md     # Comprehensive project overview
├── examples/                   # Implementation patterns and examples
│   └── feature-implementation.md
├── use-cases/                  # Specific Apple Silicon optimization scenarios
│   └── apple-silicon-optimization.md
├── PRPs/                       # Product Requirements Prompts
│   └── README.md              # PRP methodology and templates
├── CLAUDE.md                   # Global AI guidelines (symlinked to root)
├── INITIAL.md                  # Feature request template
└── README.md                   # This file
```

## Core Components

### 1. AI Configuration (.claude/)
Contains project-specific rules and context that configure AI assistants for optimal performance:

- **rules.md**: Development standards, Apple Silicon requirements, performance targets
- **project-context.md**: Architecture overview, technology stack, common patterns

### 2. Implementation Examples (examples/)
Comprehensive code patterns demonstrating:
- MLX Task Executor usage patterns
- Feature Engine integration techniques
- Model Training with Apple Silicon optimization
- FastAPI endpoint implementations
- Circuit breaker and error handling patterns
- Performance monitoring and metrics collection

### 3. Use Cases (use-cases/)
Specific scenarios for Apple Silicon optimization:
- MLX GPU acceleration patterns
- Unified memory architecture utilization
- P-core/E-core workload distribution
- Metal backend integration for LightGBM
- Thermal and power management strategies

### 4. Product Requirements Prompts (PRPs/)
Detailed implementation blueprints that serve as "screenplays" for AI development:
- Feature specifications with validation criteria
- Performance benchmarks and targets
- Testing requirements and edge cases
- Documentation standards and examples

## Context Engineering Workflow

### Phase 1: Setup Global Context
1. Configure AI rules and project context in `.claude/`
2. Establish development patterns in `examples/`
3. Define optimization strategies in `use-cases/`

### Phase 2: Feature Request
1. Use `INITIAL.md` template for structured feature requests
2. Include specific Apple Silicon requirements
3. Define performance targets and validation criteria
4. Reference relevant examples and use cases

### Phase 3: Create PRP (Product Requirements Prompt)
1. Generate comprehensive implementation blueprint
2. Include step-by-step implementation plan
3. Define validation framework with specific tests
4. Specify Apple Silicon optimization requirements

### Phase 4: AI Implementation
1. Provide complete PRP as context to AI assistant
2. AI follows detailed blueprint for implementation
3. AI validates against specified criteria
4. AI iterates based on test results and performance benchmarks

### Phase 5: Validation and Integration
1. Run comprehensive test suite
2. Validate performance targets are met
3. Ensure Apple Silicon optimizations are effective
4. Document implementation and update patterns

## Apple Silicon Focus Areas

### Performance Optimization
- **Target Metrics**: <10ms inference, >10k samples/sec throughput
- **MLX Integration**: GPU acceleration for all compute-intensive operations
- **Memory Efficiency**: Unified memory architecture utilization
- **Thermal Management**: Sustained performance under load

### Architecture Patterns
- **Async Coordination**: MLXTaskExecutor for resource management
- **Stateful Processing**: Feature engines with history buffers
- **Circuit Breaking**: Fault tolerance with purgatory library
- **Real-time Streaming**: MQTT/WebSocket with <100ms updates

### Technology Integration
- **MLX Framework**: Apple Silicon GPU/ANE acceleration
- **LightGBM Metal**: GPU-accelerated ML training
- **Polars Lazy**: Efficient data processing patterns
- **FastAPI Async**: High-performance web framework

## Quality Standards

### Code Quality
- **Type Safety**: Pydantic models for all data structures
- **Performance**: Apple Silicon optimization mandatory
- **Testing**: >95% coverage with integration tests
- **Documentation**: Context7 format with performance metrics

### Implementation Consistency
- **Patterns**: Follow established examples exactly
- **Error Handling**: Comprehensive circuit breaker usage
- **Monitoring**: Prometheus metrics for all components
- **Configuration**: Type-safe settings with Pydantic

### Validation Rigor
- **Unit Tests**: Component-level validation
- **Integration Tests**: End-to-end pipeline scenarios
- **Performance Tests**: Apple Silicon benchmark validation
- **Stress Tests**: Sustained load and thermal management

## Getting Started

### For New Features
1. Review existing patterns in `examples/`
2. Check relevant use cases in `use-cases/`
3. Create feature request using `INITIAL.md` template
4. Generate PRP with detailed implementation plan
5. Implement using AI assistant with PRP context

### For Performance Optimization
1. Review Apple Silicon use cases
2. Identify optimization opportunities
3. Create PRP with specific performance targets
4. Implement with MLX acceleration patterns
5. Validate with Apple Silicon benchmarks

### For Architecture Changes
1. Update project context and rules
2. Create new patterns in examples
3. Document use cases and optimization strategies  
4. Update PRPs for consistency
5. Validate across existing implementations

## Benefits of Context Engineering

### Development Velocity
- **Faster Implementation**: AI follows detailed blueprints
- **Reduced Iteration**: Comprehensive validation prevents rework
- **Consistent Quality**: Established patterns ensure reliability
- **Knowledge Transfer**: Context artifacts preserve implementation knowledge

### Apple Silicon Optimization
- **Built-in Performance**: Optimization patterns are standard
- **Resource Efficiency**: Unified memory and GPU utilization maximized
- **Thermal Awareness**: Sustained performance under load
- **Architecture Alignment**: P-core/E-core workload distribution

### Maintainability
- **Documentation Quality**: Implementation decisions preserved
- **Pattern Consistency**: Standardized approaches reduce complexity
- **Testing Coverage**: Comprehensive validation built into process
- **Knowledge Sharing**: Context artifacts enable team collaboration

## Success Metrics

### Implementation Quality
- **Test Coverage**: >95% for all new features
- **Performance Targets**: All Apple Silicon benchmarks met
- **Code Consistency**: All implementations follow established patterns
- **Documentation Completeness**: All features fully documented

### Development Efficiency
- **Implementation Speed**: 3x faster with comprehensive context
- **Bug Reduction**: 70% fewer issues through detailed validation
- **Knowledge Transfer**: 90% faster onboarding with context artifacts
- **Technical Debt**: Minimal accumulation through consistent patterns

This context engineering framework ensures that AI-assisted development consistently produces high-quality, Apple Silicon-optimized code that meets performance targets and follows established architectural patterns.