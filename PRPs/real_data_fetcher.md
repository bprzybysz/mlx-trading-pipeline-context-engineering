# PRP: Real Historical Data Fetcher

## 1. Feature Overview

### Objective
Implement a comprehensive real-time historical data fetcher that replaces mock data with high-quality market data from free/low-cost providers. The fetcher must support multiple data sources, handle rate limiting, cache data efficiently, and provide sufficient historical depth for technical indicator computation in the MLX Trading Pipeline.

### Business Value
- **Data Quality**: Replace 5-record mock data with years of real historical data
- **Feature Engine Enablement**: Provide sufficient history for 37+ technical indicators computation
- **Demo Viability**: Enable binary classification demo with realistic market scenarios
- **Production Readiness**: Create reusable data infrastructure for live trading
- **Cost Efficiency**: Utilize free/low-cost data providers with high quotas

### Success Criteria
- **Data Volume**: Fetch ≥2 years of daily OHLCV data for target symbols (NVDA, AAPL, TSLA, MSFT, GOOGL)
- **Data Quality**: <0.1% missing data points, validated OHLCV integrity
- **Performance**: <30s to fetch 2 years of data for 5 symbols
- **Reliability**: 99.9% success rate with automatic retry and failover
- **Cost**: Zero-cost operation within free tier quotas
- **Integration**: Seamless replacement of existing mock data infrastructure

## 2. Apple Silicon Requirements

### MLX Framework Integration
- [ ] Async data processing with MLX acceleration for large datasets
- [ ] GPU-accelerated data validation and cleaning operations
- [ ] Unified memory architecture for efficient data caching
- [ ] Metal backend compatibility for numerical computations

### Performance Targets
- **Latency**: <30s for 2 years × 5 symbols daily data
- **Throughput**: >1000 data points/second processing
- **Memory Usage**: <1GB additional overhead for caching
- **GPU Utilization**: >50% during data processing and validation

### Thermal and Power Considerations
- [ ] Sustained performance during large data fetches
- [ ] P-core/E-core workload distribution for concurrent API calls
- [ ] Power efficiency during idle periods with intelligent caching
- [ ] Thermal throttling prevention during bulk operations

## 3. Technical Architecture

### Component Integration
- [ ] **MLXTaskExecutor**: Async task scheduling for concurrent API calls and data processing
- [ ] **Feature Engine**: Provide sufficient historical data for technical indicators
- [ ] **Model Training**: Supply training datasets with real market conditions
- [ ] **Inference Service**: Historical context for prediction accuracy
- [ ] **Data Ingestion**: Replace mock data pipeline with real data streams

### Data Models
```python
class DataProviderConfig(BaseModel):
    """Configuration for external data providers."""
    provider_name: str = Field(..., description="Provider identifier")
    api_key: Optional[str] = Field(None, description="API key if required")
    base_url: str = Field(..., description="Base API URL")
    rate_limit_per_minute: int = Field(60, description="API calls per minute")
    free_tier_daily_limit: int = Field(1000, description="Daily request limit")

class HistoricalDataRequest(BaseModel):
    """Request for historical market data."""
    symbols: List[str] = Field(..., min_items=1, max_items=50)
    start_date: datetime = Field(..., description="Start date for historical data")
    end_date: datetime = Field(..., description="End date for historical data")
    interval: str = Field("1d", regex="^(1m|5m|15m|30m|1h|1d)$")
    include_dividends: bool = Field(False, description="Include dividend adjustments")
    include_splits: bool = Field(True, description="Include stock split adjustments")

class MarketDataPoint(BaseModel):
    """Enhanced market data point with quality metrics."""
    symbol: str = Field(..., min_length=1, max_length=10)
    timestamp: datetime = Field(..., description="Data point timestamp")
    open: Decimal = Field(..., gt=0, description="Opening price")
    high: Decimal = Field(..., gt=0, description="High price")
    low: Decimal = Field(..., gt=0, description="Low price") 
    close: Decimal = Field(..., gt=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Trading volume")
    adjusted_close: Optional[Decimal] = Field(None, description="Split/dividend adjusted close")
    data_quality_score: float = Field(1.0, ge=0, le=1.0, description="Data quality metric")
    source_provider: str = Field(..., description="Data source identifier")

class DataProviderResponse(BaseModel):
    """Standardized response from data providers."""
    success: bool = Field(..., description="Request success status")
    data: List[MarketDataPoint] = Field(default_factory=list)
    error_message: Optional[str] = Field(None, description="Error details if failed")
    rate_limit_remaining: int = Field(0, description="API calls remaining")
    cache_hit: bool = Field(False, description="Data served from cache")
    processing_time_ms: float = Field(..., gt=0, description="Processing time")

class RealDataFetcherConfig(BaseSettings):
    """Configuration for real data fetcher."""
    primary_provider: str = Field("yfinance", description="Primary data provider")
    fallback_providers: List[str] = Field(["alpha_vantage", "polygon"], description="Fallback providers")
    cache_directory: str = Field("./data/historical_cache", description="Data cache directory")
    enable_compression: bool = Field(True, description="Enable data compression")
    max_concurrent_requests: int = Field(10, description="Max concurrent API requests")
    request_timeout_seconds: int = Field(30, description="Request timeout")
    
    # Provider-specific settings
    alpha_vantage_api_key: Optional[str] = Field(None, env="ALPHA_VANTAGE_API_KEY")
    polygon_api_key: Optional[str] = Field(None, env="POLYGON_API_KEY")
    
    class Config:
        env_prefix = "FETCHER_"
```

### API Changes
- [ ] New endpoints: `/api/v1/data/fetch-historical`, `/api/v1/data/providers/status`
- [ ] Modified endpoints: Update existing data endpoints to use real data
- [ ] WebSocket integration: Real-time data quality and fetch progress updates
- [ ] Response format changes: Enhanced error reporting and provider attribution

## 4. Implementation Blueprint

### Phase 1: Core Implementation
1. **Setup Infrastructure**
   - [ ] Create Pydantic models for provider abstraction and data validation
   - [ ] Add configuration settings with multi-provider support
   - [ ] Setup async HTTP client pool with rate limiting
   - [ ] Initialize performance monitoring and caching layer

2. **Implement Provider Abstraction**
   ```python
   class BaseDataProvider(ABC):
       """Abstract base for data providers."""
       
       async def fetch_historical_data(
           self, 
           request: HistoricalDataRequest
       ) -> DataProviderResponse:
           """
           CONTEXT: Fetch historical market data from external provider
           INPUT: HistoricalDataRequest with symbols, dates, and parameters
           OUTPUT: DataProviderResponse with standardized market data points
           PERFORMANCE: <30s for 2 years × 5 symbols, >1000 points/second
           """
           
       async def validate_api_status(self) -> Dict[str, Any]:
           """Check provider API health and rate limits."""
           
       async def get_rate_limit_status(self) -> Dict[str, int]:
           """Get current rate limit status."""
   
   class YFinanceProvider(BaseDataProvider):
       """Yahoo Finance data provider implementation."""
       
   class AlphaVantageProvider(BaseDataProvider):
       """Alpha Vantage API provider implementation."""
       
   class PolygonProvider(BaseDataProvider):
       """Polygon.io API provider implementation."""
   ```

3. **Core Fetcher Implementation**
   ```python
   class RealHistoricalDataFetcher:
       """
       CONTEXT: Production-ready historical data fetcher with provider failover
       INPUT: Multi-symbol requests with date ranges and quality requirements
       OUTPUT: Validated, cached historical market data with quality metrics
       PERFORMANCE: <30s fetch time, 99.9% reliability, zero-cost operation
       """
       
       async def fetch_with_failover(
           self, 
           request: HistoricalDataRequest
       ) -> DataProviderResponse:
           """Fetch data with automatic provider failover."""
           
       async def validate_data_quality(
           self, 
           data: List[MarketDataPoint]
       ) -> float:
           """GPU-accelerated data quality validation."""
           
       async def cache_data_efficiently(
           self, 
           data: List[MarketDataPoint],
           request: HistoricalDataRequest
       ) -> str:
           """Cache data with compression and content addressing."""
   ```

### Phase 2: Provider Implementations
1. **Yahoo Finance Provider (Primary)**
   - [ ] Implement yfinance wrapper with rate limiting
   - [ ] Handle bulk symbol requests efficiently
   - [ ] Parse and validate Yahoo Finance data format
   - [ ] Implement retry logic for transient failures

2. **Alpha Vantage Provider (Fallback #1)**
   - [ ] Implement Alpha Vantage API client
   - [ ] Handle free tier limitations (500 requests/day)
   - [ ] Parse Alpha Vantage JSON response format
   - [ ] Implement smart request batching

3. **Polygon.io Provider (Fallback #2)**
   - [ ] Implement Polygon API client
   - [ ] Handle free tier limitations (5 requests/minute)
   - [ ] Parse Polygon REST API response format
   - [ ] Implement request queuing and backoff

### Phase 3: Integration and Optimization
1. **MLX Acceleration Integration**
   - [ ] GPU-accelerated data validation using MLX arrays
   - [ ] Parallel data processing for large datasets
   - [ ] Unified memory optimization for data caching
   - [ ] Metal backend for numerical quality computations

2. **Advanced Caching Strategy**
   ```python
   class IntelligentDataCache:
       """MLX-accelerated data caching with content addressing."""
       
       async def store_with_mlx_compression(
           self, 
           data: List[MarketDataPoint]
       ) -> str:
           """GPU-accelerated data compression and storage."""
           
       async def retrieve_with_validation(
           self, 
           cache_key: str
       ) -> Optional[List[MarketDataPoint]]:
           """Retrieve and validate cached data."""
   ```

## 5. Data Provider Research and Selection

### Primary Provider: Yahoo Finance (yfinance)
**Advantages:**
- **Free**: Unlimited requests, no API key required
- **Comprehensive**: Global markets, multiple asset classes
- **Reliable**: Backed by Yahoo Finance infrastructure
- **Fast**: Direct access to Yahoo's data feeds
- **Python Library**: Well-maintained yfinance package

**Limitations:**
- **Terms of Service**: Not officially sanctioned API
- **Rate Limiting**: Informal limits, may block aggressive usage
- **Data Lag**: 15-minute delay for real-time data

**Usage Pattern:**
```python
import yfinance as yf

# Efficient bulk download
symbols_str = " ".join(["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL"])
data = yf.download(symbols_str, start="2022-01-01", end="2024-01-01", 
                   interval="1d", group_by="ticker", threads=True)
```

### Fallback Provider #1: Alpha Vantage
**Advantages:**
- **Official API**: Proper terms of service and support
- **Free Tier**: 500 requests/day, 5 requests/minute
- **High Quality**: Professional-grade financial data
- **Comprehensive**: Stocks, forex, crypto, commodities

**Limitations:**
- **Rate Limits**: Strict 500/day limit
- **API Key Required**: Registration needed
- **Cost**: Premium tiers required for higher usage

### Fallback Provider #2: Polygon.io
**Advantages:**
- **Professional Grade**: Institutional-quality data
- **Free Tier**: 5 requests/minute
- **Real-time**: Sub-second data availability
- **Comprehensive**: US equities, options, forex, crypto

**Limitations:**
- **Rate Limits**: Very restrictive free tier (5/minute)
- **Cost**: Expensive premium tiers ($99+/month)
- **US Focus**: Limited international coverage

### Provider Selection Strategy
1. **Primary**: Yahoo Finance for bulk historical data (2+ years)
2. **Quality Validation**: Cross-reference with Alpha Vantage samples
3. **Real-time**: Polygon.io for live data validation
4. **Failover**: Automatic provider switching on rate limits/failures

## 6. Validation Framework

### Data Quality Validation
```python
@pytest.mark.unit
class TestDataQuality:
    async def test_ohlcv_integrity(self, sample_data):
        """Validate OHLCV data integrity rules."""
        for point in sample_data:
            assert point.low <= point.open <= point.high
            assert point.low <= point.close <= point.high
            assert point.volume >= 0
            
    async def test_temporal_consistency(self, sample_data):
        """Validate data timestamps are properly ordered."""
        timestamps = [point.timestamp for point in sample_data]
        assert timestamps == sorted(timestamps)
        
    async def test_mlx_accelerated_validation(self, large_dataset):
        """Test GPU-accelerated data validation performance."""
        start_time = time.time()
        quality_score = await self.fetcher.validate_data_quality(large_dataset)
        processing_time = time.time() - start_time
        
        assert quality_score > 0.95
        assert processing_time < 5.0  # <5s for 10k+ data points
```

### Provider Integration Testing
```python
@pytest.mark.integration
class TestProviderIntegration:
    async def test_yfinance_integration(self):
        """Test Yahoo Finance provider integration."""
        request = HistoricalDataRequest(
            symbols=["NVDA", "AAPL"], 
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 1, 1)
        )
        response = await self.yfinance_provider.fetch_historical_data(request)
        
        assert response.success
        assert len(response.data) > 500  # ~2 years × 2 symbols × 252 trading days
        
    async def test_provider_failover(self):
        """Test automatic provider failover functionality."""
        # Simulate primary provider failure
        self.mock_yfinance_failure()
        
        response = await self.fetcher.fetch_with_failover(self.test_request)
        
        assert response.success
        assert response.data  # Should succeed with fallback provider
        assert "fallback" in response.error_message.lower()
```

### Performance Benchmarking
```python
@pytest.mark.benchmark
class TestRealDataFetcherPerformance:
    def test_bulk_fetch_performance(self):
        """Test bulk historical data fetch performance."""
        symbols = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL"]
        request = HistoricalDataRequest(
            symbols=symbols,
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2024, 1, 1)
        )
        
        start_time = time.time()
        response = asyncio.run(self.fetcher.fetch_with_failover(request))
        fetch_time = time.time() - start_time
        
        # Performance targets
        assert fetch_time < 30.0  # <30s for 2 years × 5 symbols
        assert len(response.data) > 2500  # Expected data volume
        assert response.success
        
    def test_apple_silicon_acceleration(self):
        """Compare CPU vs MLX performance for data processing."""
        large_dataset = self.generate_test_dataset(10000)
        
        # CPU implementation
        cpu_start = time.time()
        cpu_quality = self.validate_data_quality_cpu(large_dataset)
        cpu_time = time.time() - cpu_start
        
        # MLX implementation
        mlx_start = time.time()
        mlx_quality = asyncio.run(self.fetcher.validate_data_quality(large_dataset))
        mlx_time = time.time() - mlx_start
        
        assert abs(cpu_quality - mlx_quality) < 0.01  # Same accuracy
        assert mlx_time < cpu_time * 0.5  # 2x improvement minimum
```

## 7. Error Handling Strategy

### Expected Error Scenarios
- [ ] **Provider Rate Limits**: Automatic backoff and provider switching
- [ ] **Network Failures**: Retry with exponential backoff (up to 3 attempts)
- [ ] **Invalid Symbols**: Graceful handling with partial success responses
- [ ] **Date Range Issues**: Validation and adjustment of date ranges
- [ ] **Data Quality Issues**: Automatic cleaning and quality scoring
- [ ] **Cache Corruption**: Automatic cache invalidation and refresh
- [ ] **API Key Issues**: Clear error messages and fallback options

### Error Response Patterns
```python
class DataFetchError(Exception):
    """Base exception for data fetching errors."""
    def __init__(self, provider: str, message: str, retry_after: Optional[int] = None):
        self.provider = provider
        self.message = message
        self.retry_after = retry_after
        super().__init__(f"{provider}: {message}")

class RateLimitError(DataFetchError):
    """Rate limit exceeded error."""
    pass

class DataQualityError(DataFetchError):
    """Data quality below threshold error."""
    pass

# Error handling implementation
try:
    response = await self.primary_provider.fetch_historical_data(request)
    if not response.success:
        raise DataFetchError(self.primary_provider.name, response.error_message)
        
except RateLimitError as e:
    logger.warning(f"Rate limit hit: {e}, switching to fallback provider")
    response = await self.fallback_provider.fetch_historical_data(request)
    
except DataQualityError as e:
    logger.error(f"Data quality issue: {e}, attempting data cleaning")
    cleaned_data = await self.clean_data_with_mlx(response.data)
    response.data = cleaned_data
    
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    return DataProviderResponse(
        success=False, 
        error_message=f"Failed to fetch data: {str(e)}",
        data=[]
    )
```

## 8. Configuration Management

### Required Settings
```python
class RealDataFetcherConfig(BaseSettings):
    """Comprehensive configuration for real data fetcher."""
    
    # Provider settings
    primary_provider: str = Field("yfinance", description="Primary data provider")
    fallback_providers: List[str] = Field(["alpha_vantage", "polygon"])
    
    # Performance settings
    max_concurrent_requests: int = Field(10, ge=1, le=50)
    request_timeout_seconds: int = Field(30, ge=5, le=300)
    retry_attempts: int = Field(3, ge=1, le=10)
    retry_backoff_seconds: int = Field(5, ge=1, le=60)
    
    # Caching settings
    cache_directory: str = Field("./data/historical_cache")
    enable_compression: bool = Field(True)
    cache_ttl_days: int = Field(7, ge=1, le=365)
    max_cache_size_gb: float = Field(10.0, gt=0, le=100)
    
    # Data quality settings
    min_data_quality_score: float = Field(0.95, ge=0.5, le=1.0)
    enable_data_cleaning: bool = Field(True)
    max_missing_data_percent: float = Field(1.0, ge=0, le=10)
    
    # Apple Silicon settings
    enable_mlx_acceleration: bool = Field(True)
    mlx_device_id: int = Field(0, ge=0)
    unified_memory_limit_gb: float = Field(8.0, gt=0)
    
    # Provider API keys (optional)
    alpha_vantage_api_key: Optional[str] = Field(None, env="ALPHA_VANTAGE_API_KEY")
    polygon_api_key: Optional[str] = Field(None, env="POLYGON_API_KEY")
    
    class Config:
        env_prefix = "FETCHER_"
```

### Environment Variables
```bash
# Core settings
export FETCHER_PRIMARY_PROVIDER=yfinance
export FETCHER_MAX_CONCURRENT_REQUESTS=10
export FETCHER_ENABLE_MLX_ACCELERATION=true

# API keys (optional for fallback providers)
export ALPHA_VANTAGE_API_KEY=your_key_here
export POLYGON_API_KEY=your_key_here

# Performance tuning
export FETCHER_UNIFIED_MEMORY_LIMIT_GB=16.0
export FETCHER_MAX_CACHE_SIZE_GB=20.0
```

## 9. Integration Requirements

### Binary Classification Demo Integration
- [ ] Replace mock data loader in `demo_with_real_data.py`
- [ ] Provide sufficient historical data for feature engine (37+ indicators)
- [ ] Ensure data quality meets model training requirements
- [ ] Add progress reporting for data fetching during demo

### Existing Pipeline Integration
- [ ] Update `historical_data_fetcher.py` with real provider implementations
- [ ] Maintain backward compatibility with existing mock data interface
- [ ] Add configuration for switching between mock and real data
- [ ] Update tests to use real data where appropriate

### API Service Integration
```python
@app.post("/api/v1/data/fetch-historical")
async def fetch_historical_endpoint(request: HistoricalDataRequest):
    """
    Fetch historical data via REST API.
    
    PERFORMANCE: <30s response time for 2 years × 5 symbols
    RELIABILITY: 99.9% success rate with automatic failover
    """
    try:
        response = await real_data_fetcher.fetch_with_failover(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/api/v1/data/fetch-progress")
async def fetch_progress_websocket(websocket: WebSocket):
    """Real-time progress updates for large data fetches."""
    await websocket.accept()
    # Stream progress updates during bulk data fetch
```

## 10. Acceptance Criteria

### Functional Requirements
- [ ] Successfully fetch 2+ years of daily data for target symbols (NVDA, AAPL, TSLA, MSFT, GOOGL)
- [ ] Provider failover works automatically on rate limits or errors
- [ ] Data quality validation achieves >95% quality scores
- [ ] Cache system reduces redundant API calls by >90%
- [ ] Integration with binary classification demo works seamlessly

### Performance Requirements
- [ ] <30s to fetch 2 years × 5 symbols of daily data
- [ ] >1000 data points/second processing with MLX acceleration
- [ ] <1GB additional memory usage for caching
- [ ] 99.9% reliability with automatic retry and failover
- [ ] Zero-cost operation within free tier quotas

### Quality Requirements
- [ ] >95% test coverage for all provider implementations
- [ ] All integration tests passing with real data
- [ ] No security vulnerabilities (API keys properly secured)
- [ ] Code follows project patterns and Apple Silicon optimization

### Documentation Requirements
- [ ] Provider comparison and selection rationale documented
- [ ] API documentation with real data examples
- [ ] Configuration guide for different deployment scenarios
- [ ] Troubleshooting guide for common provider issues

## 11. Risk Mitigation

### Technical Risks
- [ ] **Risk**: Yahoo Finance may block aggressive usage or change API
  - **Mitigation**: Implement intelligent rate limiting and multiple fallback providers
  
- [ ] **Risk**: Provider APIs may return inconsistent data formats
  - **Mitigation**: Comprehensive data validation and normalization layer
  
- [ ] **Risk**: Large datasets may exceed Apple Silicon memory limits
  - **Mitigation**: Streaming data processing and intelligent memory management

### Operational Risks
- [ ] **Risk**: Free tier quotas may be insufficient for production usage
  - **Mitigation**: Smart caching strategy and quota monitoring with alerts
  
- [ ] **Risk**: Provider Terms of Service changes may restrict usage
  - **Mitigation**: Legal review of ToS and diversified provider strategy

## 12. Success Validation

### Implementation Validation
- [ ] All provider integrations working with test symbols
- [ ] Binary classification demo runs successfully with real NVDA data
- [ ] Performance targets achieved (fetch time, data quality, reliability)
- [ ] MLX acceleration provides measurable improvements

### Integration Validation
- [ ] No regressions in existing pipeline functionality
- [ ] Seamless replacement of mock data infrastructure
- [ ] Real-time progress reporting working for large fetches
- [ ] Cache hit rates >90% for repeated requests

### Production Readiness
- [ ] Monitoring and alerting configured for provider health
- [ ] Error handling graceful for all failure scenarios
- [ ] Configuration flexible for different deployment environments
- [ ] Documentation complete for operational teams