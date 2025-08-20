# Generate PRP Command

## Purpose
Generate a comprehensive Product Requirements Prompt (PRP) for feature implementation in the MLX Trading Pipeline.

## Usage
This command creates a detailed implementation blueprint that serves as a "screenplay" for AI-driven development.

## Command Structure
```
/generate-prp [feature-name] [priority] [apple-silicon-optimization]
```

## Parameters
- **feature-name**: Brief name for the feature (e.g., "risk-management", "correlation-engine")
- **priority**: Implementation priority (high, medium, low)  
- **apple-silicon-optimization**: Required optimization level (performance, memory, thermal)

## PRP Generation Process

### 1. Analyze Feature Request
- Review INITIAL.md for feature details
- Identify Apple Silicon optimization opportunities
- Assess performance impact and requirements
- Determine integration complexity

### 2. Generate Implementation Blueprint
- Create detailed step-by-step implementation plan
- Include specific Apple Silicon optimization patterns
- Define comprehensive validation criteria
- Specify testing requirements and benchmarks

### 3. Include Reference Materials
- Link to relevant examples from examples/ directory
- Reference appropriate use-cases from use-cases/
- Include external documentation links
- Provide similar implementation patterns

### 4. Define Validation Framework
- Specify unit test requirements (>95% coverage)
- Define integration test scenarios
- Set Apple Silicon performance benchmarks
- Include error handling test cases

## Example Usage

### Generate PRP for Risk Management Feature
```
Input: /generate-prp risk-management high performance
```

**Generated PRP Includes**:
- Feature overview with business justification
- Apple Silicon optimization requirements (MLX acceleration)
- Step-by-step implementation plan with code patterns
- Performance targets (<5ms risk calculation latency)
- Comprehensive test suite with benchmarks
- Documentation requirements with examples

### Generate PRP for API Enhancement
```
Input: /generate-prp websocket-streaming medium memory
```

**Generated PRP Includes**:
- WebSocket implementation blueprint
- Memory-efficient streaming patterns for Apple Silicon
- Integration with existing FastAPI infrastructure  
- Real-time performance validation requirements
- Load testing specifications

## PRP Template Structure
Each generated PRP follows this structure:

```markdown
# PRP: [Feature Name]

## 1. Feature Overview
- Objective
- Business Value  
- Success Criteria

## 2. Apple Silicon Requirements
- MLX framework usage
- Performance targets
- Memory optimization
- Thermal considerations

## 3. Implementation Blueprint
- Step-by-step plan
- Code patterns to follow
- Integration points
- Configuration updates

## 4. Validation Framework
- Unit test specifications
- Integration test scenarios
- Performance benchmarks
- Error handling requirements

## 5. Documentation Requirements
- Code documentation standards
- API specification updates
- Architecture documentation changes
- User guide updates
```

## Quality Standards

### Technical Requirements
- All implementations must use Apple Silicon optimization patterns
- Performance targets must be specific and measurable
- Integration with MLXTaskExecutor is mandatory
- Type safety with Pydantic models required

### Validation Requirements
- Comprehensive test coverage (>95%)
- Apple Silicon benchmark validation
- Integration testing with realistic data
- Error handling and edge case coverage

### Documentation Standards
- Context7 format docstrings required
- Performance metrics in all documentation
- Troubleshooting guides for Apple Silicon issues
- Architecture impact documentation

## Best Practices

### PRP Creation
- Be extremely specific in all requirements
- Include concrete examples and code patterns
- Define measurable success criteria
- Consider Apple Silicon thermal and power constraints

### Implementation Guidance
- Reference established patterns from examples/
- Follow Apple Silicon optimization use cases
- Implement comprehensive monitoring
- Ensure backwards compatibility where possible

### Validation Thoroughness
- Test all specified scenarios completely
- Measure all performance targets
- Validate Apple Silicon optimizations
- Document any deviations from PRP specifications