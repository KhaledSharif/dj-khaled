# AudioCraft Test Suite

Comprehensive test suite for the AudioCraft music generation system, including unit tests, integration tests, and edge case validation.

## Test Structure

### test_main.py

Core functionality tests covering:

- Model loading and caching
- Audio generation and processing
- Layer generation with conditioning
- Mix operations and normalization
- Cache operations
- Complete production workflow

### test_coverage.py

Additional tests for complete code coverage:

- Error handling scenarios
- Edge cases in audio processing
- Alternative code paths
- Boundary conditions

### test_corner_cases.py

Advanced corner case testing:

- Numerical edge cases (NaN, Inf, extreme values)
- Configuration validation
- Audio processing boundaries
- Model output validation
- Resource and filesystem handling
- Real-world integration scenarios

## Running Tests

### Run All Tests

```bash
# From repository root
python run_tests.py

# Or from tests directory
cd tests
python -m unittest discover
```

### Run Specific Test Files

```bash
cd tests
python test_main.py
python test_coverage.py
python test_corner_cases.py
```

### Run Specific Test Classes

```bash
python -m unittest test_main.TestAudioGeneration
python -m unittest test_corner_cases.TestNumericalEdgeCases
```

### Run Specific Test Methods

```bash
python -m unittest test_main.TestAudioGeneration.test_generate_section_with_conditioning
```

## Test Coverage

To measure code coverage:

```bash
pip install coverage
coverage run -m unittest discover tests
coverage report
coverage html  # Generate HTML report
```

## Test Categories

### Unit Tests

- Individual method testing
- Mock external dependencies
- Isolated functionality verification

### Integration Tests

- Multi-component interaction
- Full production pipeline
- Conditioning chains
- Cache integration

### Edge Case Tests

- Boundary value testing
- Error conditions
- Resource limits
- Malformed inputs

## Common Test Patterns

### Mocking MusicGen Models

```python
with patch("main.MusicGen") as mock_musicgen:
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_musicgen.get_pretrained.return_value = mock_model
```

### Testing with Temporary Files

```python
def setUp(self):
    self.temp_dir = tempfile.mkdtemp()
    
def tearDown(self):
    shutil.rmtree(self.temp_dir)
```

### Asserting Tensor Properties

```python
# Check tensor shape
self.assertEqual(result.shape, (expected_samples,))

# Check value ranges
self.assertLessEqual(torch.abs(result).max().item(), 1.0)

# Check tensor equality
self.assertTrue(torch.allclose(expected, actual))
```

## Test Fixtures

### Sample Configurations

Tests use minimal YAML configurations for speed:

```python
config = {
    "song": {"name": "Test", "sample_rate": 32000},
    "sections": {"intro": {"start": 0, "duration": 10}},
    "layers": [...]
}
```

### Mock Audio Data

```python
# Generate mock audio tensors
mock_audio = torch.randn(32000 * duration)  # Random audio
mock_audio = torch.ones(32000 * duration)   # Constant audio
mock_audio = torch.zeros(32000 * duration)  # Silent audio
```

## Known Test Issues

1. **Mock Tensor Operations**: Some PyTorch operations don't work with MagicMock objects
2. **File System Tests**: Some tests require actual file I/O and can't be fully mocked
3. **Model Loading**: Real model loading is skipped in tests for speed

## Writing New Tests

### Test Structure Template

```python
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        # Initialize test objects
        
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir)
        
    def test_feature_normal_case(self):
        """Test normal operation"""
        # Arrange
        # Act
        # Assert
        
    def test_feature_edge_case(self):
        """Test edge cases"""
        # Test boundary conditions
```

### Best Practices

1. **Isolation**: Each test should be independent
2. **Clear Names**: Test names should describe what they test
3. **Single Responsibility**: Test one thing per test method
4. **Fast Execution**: Mock expensive operations
5. **Deterministic**: Tests should not depend on random values

## Debugging Failed Tests

### Verbose Output

```bash
python -m unittest test_main -v
```

### Run with Python Debugger

```bash
python -m pdb test_main.py
```

### Print Debug Information

Add print statements or use logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Continuous Integration

Tests should be run:

- Before committing changes
- In CI/CD pipelines
- After dependency updates
- When changing Python versions

## Test Maintenance

### Adding Tests for New Features

1. Write tests before implementation (TDD)
2. Cover happy path and error cases
3. Test boundary conditions
4. Add integration tests for complex features

### Updating Tests for Changes

1. Update mocks when signatures change
2. Adjust assertions for new behavior
3. Add tests for bug fixes
4. Remove obsolete tests

## Performance Testing

For performance-critical code:

```python
import time

def test_performance(self):
    start = time.time()
    # Run operation
    result = expensive_operation()
    elapsed = time.time() - start
    self.assertLess(elapsed, 1.0)  # Should complete in < 1 second
```
