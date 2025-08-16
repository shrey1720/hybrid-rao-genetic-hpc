# Makefile for Hybrid RAO-Genetic Algorithm for Fungal Growth Optimization
# Author: Shrey Panchasara
# Date: 17/08/2025

# Compiler and flags
NVCC = nvcc
CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c99
NVCCFLAGS = -O3 -arch=sm_60 -Xcompiler -O3

# Directories
SRC_DIR = src
BIN_DIR = bin
SERIAL_DIR = $(SRC_DIR)/serial
PARALLEL_DIR = $(SRC_DIR)/parallel
COMMON_DIR = $(SRC_DIR)/common

# Source files
SERIAL_SRC = $(SERIAL_DIR)/bactserial.cu
PARALLEL_SRC = $(PARALLEL_DIR)/bactpara.cu

# Executables
SERIAL_EXEC = $(BIN_DIR)/bactserial
PARALLEL_EXEC = $(BIN_DIR)/bactpara

# Default target
all: $(SERIAL_EXEC) $(PARALLEL_EXEC)

# Create bin directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build serial version
$(SERIAL_EXEC): $(SERIAL_SRC) | $(BIN_DIR)
	@echo "Building serial version..."
	$(NVCC) $(NVCCFLAGS) -o $@ $<
	@echo "Serial version built successfully: $@"

# Build parallel version
$(PARALLEL_EXEC): $(PARALLEL_SRC) | $(BIN_DIR)
	@echo "Building parallel version..."
	$(NVCC) $(NVCCFLAGS) -o $@ $<
	@echo "Parallel version built successfully: $@"

# Build with debug information
debug: NVCCFLAGS += -g -G
debug: CFLAGS += -g
debug: all

# Build with maximum optimization
optimized: NVCCFLAGS += -O3 -use_fast_math
optimized: CFLAGS += -O3 -march=native
optimized: all

# Build for specific CUDA architecture
sm70: NVCCFLAGS = -O3 -arch=sm_70 -Xcompiler -O3
sm70: all

sm80: NVCCFLAGS = -O3 -arch=sm_80 -Xcompiler -O3
sm80: all

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BIN_DIR)
	rm -f *.o *.out
	@echo "Clean complete"

# Install dependencies (Ubuntu/Debian)
install-deps:
	@echo "Installing dependencies..."
	sudo apt-get update
	sudo apt-get install -y build-essential nvidia-cuda-toolkit
	@echo "Dependencies installed"

# Install dependencies (CentOS/RHEL)
install-deps-rhel:
	@echo "Installing dependencies for RHEL/CentOS..."
	sudo yum groupinstall -y "Development Tools"
	sudo yum install -y cuda-toolkit
	@echo "Dependencies installed"

# Run tests
test: all
	@echo "Running serial version test..."
	@time $(SERIAL_EXEC) > results/serial_results.txt 2>&1
	@echo "Running parallel version test..."
	@time $(PARALLEL_EXEC) > results/parallel_results.txt 2>&1
	@echo "Tests completed. Check results/ directory"

# Performance comparison
benchmark: all
	@echo "Running performance benchmark..."
	@echo "=== SERIAL VERSION ===" > results/benchmark.txt
	@time $(SERIAL_EXEC) >> results/benchmark.txt 2>&1
	@echo -e "\n=== PARALLEL VERSION ===" >> results/benchmark.txt
	@time $(PARALLEL_EXEC) >> results/benchmark.txt 2>&1
	@echo "Benchmark completed. Check results/benchmark.txt"

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version || echo "CUDA not found. Please install CUDA toolkit."
	@nvidia-smi || echo "NVIDIA driver not found or GPU not available."

# Show help
help:
	@echo "Available targets:"
	@echo "  all          - Build both serial and parallel versions (default)"
	@echo "  debug        - Build with debug information"
	@echo "  optimized    - Build with maximum optimization"
	@echo "  sm70         - Build for CUDA compute capability 7.0"
	@echo "  sm80         - Build for CUDA compute capability 8.0"
	@echo "  clean        - Remove build artifacts"
	@echo "  install-deps - Install dependencies (Ubuntu/Debian)"
	@echo "  install-deps-rhel - Install dependencies (RHEL/CentOS)"
	@echo "  test         - Run both versions and save results"
	@echo "  benchmark    - Run performance comparison"
	@echo "  check-cuda   - Check CUDA installation"
	@echo "  help         - Show this help message"

# Create results directory
results:
	mkdir -p results

# Run with results directory creation
run: results test

# Phony targets
.PHONY: all debug optimized sm70 sm80 clean install-deps install-deps-rhel test benchmark check-cuda help results run
