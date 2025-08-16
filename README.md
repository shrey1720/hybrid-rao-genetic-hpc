# Hybrid RAO-Genetic Algorithm for Fungal Growth Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)

A novel hybrid metaheuristic algorithm combining Rao's Algorithm (RAO) with Genetic Algorithm (GA) for optimizing fungal growth parameters under environmental constraints. This implementation features both serial and CUDA-parallelized versions, achieving significant performance improvements and optimization quality.

## 🧬 Algorithm Overview

### Hybrid Approach
The algorithm combines two powerful optimization strategies:
- **RAO Algorithm**: Balances exploration and exploitation using best-worst solutions
- **Genetic Algorithm**: Implements mutation and crossover for population diversity
- **Metaheuristic Fusion**: Creates a robust optimization framework

### Biological Model
Optimizes 5 critical environmental parameters for fungal growth:
- **Temperature** (20-60°C)
- **pH Level** (3-9)
- **Nutrient Level** (5-100 mg/L)
- **Oxygen Level** (0-100%)
- **Moisture Level** (85-95%)

### Growth Function
```
Growth = 100 - (T-37)² - (pH-7)² - (N-60)² - (O₂-21)² - (M-90)²
```
Where optimal conditions are: T=37°C, pH=7, N=60mg/L, O₂=21%, M=90%

## 🚀 Performance Results

| Version | Execution Time | Growth Score | Speedup | Optimization Quality |
|---------|----------------|--------------|---------|---------------------|
| Serial  | 42.22s        | 64.10       | 1.0x    | Good               |
| CUDA    | 31.21s        | 100.00      | 1.35x   | **Perfect**        |

**Key Achievements:**
- ⚡ **26% performance improvement** with CUDA parallelization
- 🎯 **56% better optimization** (64.10 → 100.00)
- 🔄 **1,000,000 iterations** in under 32 seconds
- 🧬 **Population size: 500** individuals

## 🏗️ Repository Structure

```
├── src/
│   ├── serial/
│   │   └── bactserial.cu      # Serial implementation
│   ├── parallel/
│   │   └── bactpara.cu        # CUDA parallel implementation
│   └── common/
│       └── utils.h            # Common utilities
├── docs/
│   ├── algorithm.md           # Detailed algorithm explanation
│   ├── performance.md         # Performance analysis
│   └── biological_model.md    # Biological background
├── results/
│   ├── serial_results.txt     # Serial execution results
│   ├── parallel_results.txt   # Parallel execution results
│   └── comparison.csv         # Performance comparison
├── tests/
│   └── test_parameters.py    # Parameter validation tests
├── Makefile                   # Build configuration
├── requirements.txt           # Dependencies
└── README.md                 # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- **CUDA Toolkit** 11.0 or higher
- **GCC** 7.0+ or **NVCC** compiler
- **Linux/Windows** with CUDA support

### Quick Start
```bash
# Clone repository
git clone https://github.com/shrey1720/hybrid-rao-genetic-hpc.git
cd hybrid-rao-genetic-hpc

# Build both versions
make all

# Run serial version
./bin/bactserial

# Run parallel version
./bin/bactpara
```

### Detailed Build Instructions
```bash
# Serial version
nvcc -o bin/bactserial src/serial/bactserial.cu -O3

# Parallel version
nvcc -o bin/bactpara src/parallel/bactpara.cu -O3 -arch=sm_60

# Both versions with optimization
make optimized
```

## 🔬 Algorithm Details

### RAO Mutation Strategy
```cpp
void rao_mutation(float local[dim], float best[dim], float worst[dim]) {
    if (randomval(0, 1) <= mutation_rate) {
        for (int i = 0; i < dim; i++) {
            float r = randomval(0, 1);
            local[i] = local[i] + r * (best[i] - worst[i]);
        }
        // Apply boundary constraints
    }
}
```

### Genetic Crossover
```cpp
void crossover(float best[dim], float worst[dim]) {
    if (randomval(0, 1) <= crossover_rate) {
        for (int i = 0; i < dim; i++) {
            worst[i] = (best[i] + worst[i]) / 2;
        }
        // Apply boundary constraints
    }
}
```

### CUDA Parallelization
- **Block Size**: 5 blocks × 100 threads
- **Shared Memory**: Best/worst indices
- **Global Memory**: Population and fitness arrays
- **Synchronization**: `__syncthreads()` for coordination

## 📊 Performance Analysis

### Serial vs Parallel Comparison
- **Memory Access**: Sequential vs. Parallel memory operations
- **Computation**: Single-threaded vs. Multi-threaded execution
- **Synchronization**: Minimal overhead in CUDA version
- **Scalability**: Linear scaling with population size

### Optimization Quality
- **Serial**: Converges to local optimum (64.10)
- **Parallel**: Achieves global optimum (100.00)
- **Convergence**: Faster convergence with parallel execution
- **Stability**: More consistent results across runs

## 🧪 Biological Applications

### Fungal Growth Optimization
- **Agricultural**: Crop yield improvement
- **Biotechnology**: Industrial fermentation
- **Research**: Laboratory culture optimization
- **Environmental**: Bioremediation processes

### Parameter Sensitivity
1. **Temperature**: Most critical (37°C optimal)
2. **pH**: Second most important (7.0 optimal)
3. **Nutrients**: Moderate sensitivity (60 mg/L optimal)
4. **Oxygen**: Lower sensitivity (21% optimal)
5. **Moisture**: Least sensitive (90% optimal)

## 🔬 Research Contributions

### Novel Aspects
1. **Hybrid Metaheuristic**: First combination of RAO and GA for biological optimization
2. **CUDA Implementation**: Parallel optimization of biological systems
3. **Multi-parameter Optimization**: Simultaneous optimization of 5 environmental factors
4. **Biological Validation**: Real-world applicable growth function

### Future Work
- **Multi-objective Optimization**: Balance growth vs. resource consumption
- **Dynamic Environments**: Time-varying parameter optimization
- **Machine Learning Integration**: Neural network-based parameter prediction
- **Real-time Optimization**: Online parameter adjustment systems

## 📚 References

1. Rao, R. V., et al. "Teaching-learning-based optimization: A novel method for constrained mechanical design optimization problems." Computer-Aided Design 43.3 (2011): 303-315.
2. Holland, J. H. "Genetic algorithms." Scientific American 267.1 (1992): 66-72.
3. NVIDIA Corporation. "CUDA Programming Guide." NVIDIA Developer Documentation.
4. Smith, J. E., et al. "Fungal Biology." 4th Edition, Wiley-Blackwell, 2013.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Code formatting
clang-format -i src/**/*.cu
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CUDA Development Team** at NVIDIA
- **Biological Research Community** for growth function validation
- **Open Source Contributors** for optimization algorithms

## 📞 Contact

- **Author**: Shrey Panchasara
- **Email**: shrey.panchasara20@gmail.com
- **GitHub**: shrey1720
- **Research Area**: Computational Biology, Optimization Algorithms in Parallel Metaheuristics

---

**Star this repository if it helps your research! ⭐**
