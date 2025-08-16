# Contributing to Hybrid RAO-Genetic Algorithm

Thank you for your interest in contributing to the Hybrid RAO-Genetic Algorithm project! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

- Use the GitHub issue tracker
- Include detailed reproduction steps
- Provide system information (OS, CUDA version, etc.)
- Include error messages and stack traces

### Suggesting Enhancements

- Describe the enhancement clearly
- Explain why it would be useful
- Provide examples if applicable
- Consider implementation complexity

### Code Contributions

- Algorithm improvements
- Performance optimizations
- Bug fixes
- Documentation updates
- Test coverage improvements

## Development Setup

### Prerequisites

1. **CUDA Toolkit** (11.0 or higher)
2. **NVIDIA GPU** with compute capability 6.0+
3. **C++ Compiler** (GCC 7.0+ or Clang 6.0+)
4. **Python** (3.8+ for development tools)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hybrid-rao-genetic-fungal.git
cd hybrid-rao-genetic-fungal

# Install development dependencies
pip install -r requirements.txt

# Build the project
make all

# Run tests
make test
```

### Development Environment

```bash
# Set up pre-commit hooks
pre-commit install

# Activate virtual environment (if using)
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

## Coding Standards

### C++/CUDA Code

- **Naming Convention**: snake_case for variables and functions
- **Constants**: UPPER_CASE for macros and constants
- **Indentation**: 4 spaces (no tabs)
- **Line Length**: Maximum 100 characters
- **Documentation**: Doxygen-style comments for all functions

#### Example

```cpp
/**
 * @brief Calculate fungal growth score
 * @param temperature Temperature in Celsius
 * @param ph pH level
 * @param nutrients Nutrient concentration in mg/L
 * @return Growth score (0-100)
 */
float calculate_growth_score(float temperature, float ph, float nutrients) {
    const float OPTIMAL_TEMP = 37.0f;
    const float OPTIMAL_PH = 7.0f;
    const float OPTIMAL_NUTRIENTS = 60.0f;
    
    float temp_score = -(temperature - OPTIMAL_TEMP) * (temperature - OPTIMAL_TEMP);
    float ph_score = -(ph - OPTIMAL_PH) * (ph - OPTIMAL_PH);
    float nutrient_score = -(nutrients - OPTIMAL_NUTRIENTS) * (nutrients - OPTIMAL_NUTRIENTS);
    
    return temp_score + ph_score + nutrient_score + 100.0f;
}
```

### Python Code

- **Style**: Follow PEP 8 guidelines
- **Type Hints**: Use type hints for function parameters and return values
- **Docstrings**: Use Google-style docstrings

#### Example

```python
from typing import List, Tuple, Optional
import numpy as np

def analyze_performance(
    serial_times: List[float], 
    parallel_times: List[float]
) -> Tuple[float, float]:
    """
    Analyze performance improvement between serial and parallel implementations.
    
    Args:
        serial_times: List of execution times for serial version
        parallel_times: List of execution times for parallel version
        
    Returns:
        Tuple of (mean_speedup, std_speedup)
        
    Raises:
        ValueError: If input lists have different lengths
    """
    if len(serial_times) != len(parallel_times):
        raise ValueError("Input lists must have the same length")
    
    speedups = [s / p for s, p in zip(serial_times, parallel_times)]
    return np.mean(speedups), np.std(speedups)
```

## Testing Guidelines

### C++/CUDA Testing

- **Unit Tests**: Test individual functions
- **Integration Tests**: Test algorithm components
- **Performance Tests**: Benchmark execution times
- **Memory Tests**: Check for memory leaks

#### Test Structure

```cpp
#include <gtest/gtest.h>
#include "growth_function.h"

class GrowthFunctionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup test data
    }
    
    void TearDown() override {
        // Cleanup test data
    }
};

TEST_F(GrowthFunctionTest, OptimalConditions) {
    float score = calculate_growth_score(37.0f, 7.0f, 60.0f);
    EXPECT_NEAR(score, 100.0f, 0.01f);
}

TEST_F(GrowthFunctionTest, BoundaryConditions) {
    float score = calculate_growth_score(20.0f, 3.0f, 5.0f);
    EXPECT_GT(score, 0.0f);
    EXPECT_LT(score, 100.0f);
}
```

### Python Testing

- **Unit Tests**: Test utility functions
- **Integration Tests**: Test data processing
- **Performance Tests**: Benchmark analysis functions

#### Test Structure

```python
import pytest
import numpy as np
from src.analysis import analyze_performance

class TestPerformanceAnalysis:
    def test_speedup_calculation(self):
        """Test speedup calculation with known values."""
        serial_times = [10.0, 20.0, 30.0]
        parallel_times = [5.0, 10.0, 15.0]
        
        mean_speedup, std_speedup = analyze_performance(serial_times, parallel_times)
        
        assert mean_speedup == 2.0
        assert std_speedup == 0.0
    
    def test_different_length_lists(self):
        """Test error handling for mismatched list lengths."""
        with pytest.raises(ValueError):
            analyze_performance([1.0, 2.0], [1.0])
```

## Documentation

### Code Documentation

- **Function Documentation**: Explain purpose, parameters, and return values
- **Algorithm Documentation**: Describe mathematical foundations
- **Example Usage**: Provide practical examples
- **Performance Notes**: Document complexity and optimization tips

### User Documentation

- **Installation Guide**: Step-by-step setup instructions
- **Usage Examples**: Common use cases and workflows
- **Troubleshooting**: Common issues and solutions
- **API Reference**: Complete function documentation

### Research Documentation

- **Algorithm Theory**: Mathematical foundations and proofs
- **Performance Analysis**: Benchmark results and analysis
- **Biological Model**: Scientific background and validation
- **Future Work**: Research directions and extensions

## Pull Request Process

### Before Submitting

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Ensure all tests pass**: `make test`
7. **Check code style**: `make lint`

### Pull Request Guidelines

- **Clear Title**: Descriptive title for the PR
- **Detailed Description**: Explain what and why, not how
- **Related Issues**: Link to relevant issues
- **Screenshots**: Include visual changes if applicable
- **Test Results**: Show that tests pass

### Review Process

1. **Automated Checks**: CI/CD pipeline validation
2. **Code Review**: At least one maintainer approval
3. **Testing**: Ensure all tests pass
4. **Documentation**: Verify documentation is updated
5. **Merge**: Approved PRs are merged

## Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Changelog is updated
- [ ] Version numbers are updated
- [ ] Release notes are written
- [ ] GitHub release is created

### Release Notes Template

```markdown
## [Version] - [Date]

### Added
- New features and functionality

### Changed
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Security-related changes
```

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: [your.email@example.com]

### Resources

- **Documentation**: [docs/](docs/) directory
- **Examples**: [examples/](examples/) directory
- **Papers**: Research publications and references
- **Community**: Related projects and collaborations

## Recognition

Contributors will be recognized in:

- **README.md**: Contributor list
- **Changelog**: Individual contributions
- **Research Papers**: Co-authorship for significant contributions
- **Conference Presentations**: Acknowledgment in presentations

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Hybrid RAO-Genetic Algorithm project! Your contributions help advance research in computational biology and optimization algorithms.
