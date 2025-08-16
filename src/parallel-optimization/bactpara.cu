/**
 * @file bactpara.cu
 * @brief CUDA parallel implementation of Hybrid RAO-Genetic Algorithm for Fungal Growth Optimization
 * @author Shrey Panchasara
 * @date 17/08/2025
 * 
 * This file implements the hybrid RAO-Genetic Algorithm using CUDA parallelization for
 * optimizing fungal growth parameters. The parallel implementation significantly improves
 * performance while maintaining optimization quality.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// Algorithm Parameters
#define POPULATION_SIZE 500      ///< Number of individuals in population
#define DIMENSIONS 5             ///< Number of environmental parameters
#define ITERATIONS 1000000       ///< Maximum number of optimization iterations

// CUDA Configuration
#define BLOCKS 5                 ///< Number of CUDA blocks
#define THREADS_PER_BLOCK 100    ///< Threads per CUDA block

// Environmental Parameter Bounds
#define TEMP_MIN 20.0           ///< Minimum temperature (°C)
#define TEMP_MAX 45.0           ///< Maximum temperature (°C)
#define PH_MIN 4.0              ///< Minimum pH level
#define PH_MAX 9.0              ///< Maximum pH level
#define NUTRIENT_MIN 10.0       ///< Minimum nutrient level (mg/L)
#define NUTRIENT_MAX 100.0      ///< Maximum nutrient level (mg/L)
#define OXYGEN_MIN 0.0          ///< Minimum oxygen level (%)
#define OXYGEN_MAX 100.0        ///< Maximum oxygen level (%)
#define MOISTURE_MIN 85.0       ///< Minimum moisture level (%)
#define MOISTURE_MAX 95.0       ///< Maximum moisture level (%)

// Genetic Algorithm Parameters
#define MUTATION_RATE 0.8       ///< Probability of mutation (higher for parallel)
#define CROSSOVER_RATE 0.3      ///< Probability of crossover

/**
 * @brief GPU random number generator (device function)
 * @param min Minimum value
 * @param max Maximum value
 * @param seed Seed for pseudo-random generation
 * @return Random float value
 */
__device__ float randomGPU(float min, float max, int seed) {
    return min + ((float)(seed % 10000) / 10000.0f) * (max - min);
}

/**
 * @brief Calculate fungal growth score (host and device function)
 * 
 * The growth function uses a negative squared difference approach where:
 * - Optimal temperature: 37°C
 * - Optimal pH: 7.0
 * - Optimal nutrients: 60 mg/L
 * - Optimal oxygen: 21%
 * - Optimal moisture: 90%
 * 
 * @param t Temperature (°C)
 * @param p pH level
 * @param n Nutrient level (mg/L)
 * @param o Oxygen level (%)
 * @param m Moisture level (%)
 * @return Growth score (0-100, higher is better)
 */
__host__ __device__ float growth(float t, float p, float n, float o, float m) {
    return -(t - 37) * (t - 37) - (p - 7) * (p - 7) - 
           (n - 60) * (n - 60) - (o - 21) * (o - 21) - 
           (m - 90) * (m - 90) + 100;
}

/**
 * @brief Apply boundary constraints to parameter values (device function)
 * @param val Pointer to value to constrain
 * @param min Minimum allowed value
 * @param max Maximum allowed value
 */
__device__ void boundary(float *val, float min, float max) {
    if (*val < min) *val = min;
    if (*val > max) *val = max;
}

/**
 * @brief Apply RAO mutation to individual parameters (device function)
 * 
 * RAO mutation uses the difference between best and worst solutions
 * to guide the search direction, balancing exploration and exploitation.
 * 
 * @param local Current individual parameters
 * @param best Best solution parameters
 * @param worst Worst solution parameters
 * @param seed Seed for random number generation
 */
__device__ void rao_mutation(float local[DIMENSIONS], float best[DIMENSIONS], 
                             float worst[DIMENSIONS], int seed) {
    float r_check = randomGPU(0, 1, seed + 7);
    if (r_check <= MUTATION_RATE) {
        for (int j = 0; j < DIMENSIONS; j++) {
            float r = randomGPU(0, 1, seed + j);
            local[j] += r * (best[j] - worst[j]);
        }
        
        // Apply boundary constraints after mutation
        boundary(&local[0], TEMP_MIN, TEMP_MAX);      // Temperature
        boundary(&local[1], PH_MIN, PH_MAX);          // pH
        boundary(&local[2], NUTRIENT_MIN, NUTRIENT_MAX); // Nutrients
        boundary(&local[3], OXYGEN_MIN, OXYGEN_MAX);  // Oxygen
        boundary(&local[4], MOISTURE_MIN, MOISTURE_MAX); // Moisture
    }
}

/**
 * @brief Apply genetic crossover between best and worst solutions (device function)
 * 
 * Crossover creates diversity by combining genetic material from
 * different individuals, helping escape local optima.
 * 
 * @param best Best solution parameters
 * @param worst Worst solution parameters
 */
__device__ void crossover(float best[DIMENSIONS], float worst[DIMENSIONS]) {
    if (randomGPU(0, 1, threadIdx.x) <= CROSSOVER_RATE) {
        for (int j = 0; j < DIMENSIONS; j++) {
            worst[j] = (best[j] + worst[j]) / 2.0f;
        }
        
        // Apply boundary constraints after crossover
        boundary(&worst[0], TEMP_MIN, TEMP_MAX);
        boundary(&worst[1], PH_MIN, PH_MAX);
        boundary(&worst[2], NUTRIENT_MIN, NUTRIENT_MAX);
        boundary(&worst[3], OXYGEN_MIN, OXYGEN_MAX);
        boundary(&worst[4], MOISTURE_MIN, MOISTURE_MAX);
    }
}

/**
 * @brief CUDA kernel for parallel optimization
 * 
 * This kernel implements the hybrid RAO-Genetic Algorithm in parallel:
 * 1. Each thread handles one individual
 * 2. Shared memory stores best/worst indices
 * 3. Synchronization ensures proper coordination
 * 4. Parallel mutation and crossover operations
 * 
 * @param population Array of population parameters (device memory)
 * @param fit Array of fitness values (device memory)
 */
__global__ void optimize_kernel(float population[POPULATION_SIZE][DIMENSIONS], 
                               float fit[POPULATION_SIZE]) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= POPULATION_SIZE) return;
    
    // Shared memory for best/worst indices
    __shared__ int best_idx;
    __shared__ int worst_idx;
    
    // Thread 0 finds best and worst solutions
    if (threadIdx.x == 0) {
        best_idx = 0;
        worst_idx = 0;
        for (int i = 1; i < POPULATION_SIZE; i++) {
            if (fit[i] > fit[best_idx]) best_idx = i;
            if (fit[i] < fit[worst_idx]) worst_idx = i;
        }
    }
    __syncthreads();  // Ensure all threads see the best/worst indices
    
    // Local copy of individual parameters
    float local[DIMENSIONS];
    for (int j = 0; j < DIMENSIONS; j++) {
        local[j] = population[tid][j];
    }
    
    // Apply RAO mutation
    rao_mutation(local, population[best_idx], population[worst_idx], tid);
    
    // Update population with mutated values
    for (int j = 0; j < DIMENSIONS; j++) {
        population[tid][j] = local[j];
    }
    
    // Calculate new fitness
    fit[tid] = growth(local[0], local[1], local[2], local[3], local[4]);
    
    __syncthreads();  // Ensure all fitness values are updated
    
    // Apply crossover (only worst individual)
    if (tid == worst_idx) {
        crossover(population[best_idx], population[worst_idx]);
        fit[worst_idx] = growth(population[worst_idx][0], population[worst_idx][1], 
                               population[worst_idx][2], population[worst_idx][3], 
                               population[worst_idx][4]);
    }
}

/**
 * @brief CPU random number generator for host initialization
 * @param min Minimum value
 * @param max Maximum value
 * @return Random float value
 */
float randomCPU(float min, float max) {
    return min + ((float)rand() / RAND_MAX) * (max - min);
}

/**
 * @brief Main function - Entry point of the program
 * @return Exit status
 */
int main() {
    // Start timing
    clock_t start = clock();
    srand(time(NULL));
    
    // Host memory allocation
    float h_population[POPULATION_SIZE][DIMENSIONS];
    float h_fit[POPULATION_SIZE];
    
    // Initialize host population with random values
    printf("Initializing population with %d individuals...\n", POPULATION_SIZE);
    for (int i = 0; i < POPULATION_SIZE; i++) {
        h_population[i][0] = randomCPU(TEMP_MIN, TEMP_MAX);      // Temperature
        h_population[i][1] = randomCPU(PH_MIN, PH_MAX);          // pH
        h_population[i][2] = randomCPU(NUTRIENT_MIN, NUTRIENT_MAX); // Nutrients
        h_population[i][3] = randomCPU(OXYGEN_MIN, OXYGEN_MAX);  // Oxygen
        h_population[i][4] = randomCPU(MOISTURE_MIN, MOISTURE_MAX); // Moisture
        
        // Calculate initial fitness
        h_fit[i] = growth(h_population[i][0], h_population[i][1], h_population[i][2], 
                          h_population[i][3], h_population[i][4]);
    }
    
    // Device memory allocation
    float (*d_population)[DIMENSIONS];
    float *d_fit;
    
    cudaMalloc(&d_population, sizeof(float) * POPULATION_SIZE * DIMENSIONS);
    cudaMalloc(&d_fit, sizeof(float) * POPULATION_SIZE);
    
    // Copy data from host to device
    cudaMemcpy(d_population, h_population, sizeof(float) * POPULATION_SIZE * DIMENSIONS, 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_fit, h_fit, sizeof(float) * POPULATION_SIZE, cudaMemcpyHostToDevice);
    
    printf("Starting CUDA parallel optimization with %d iterations...\n", ITERATIONS);
    printf("CUDA Configuration: %d blocks × %d threads = %d total threads\n", 
           BLOCKS, THREADS_PER_BLOCK, BLOCKS * THREADS_PER_BLOCK);
    
    // Run optimization kernel for specified iterations
    for (int i = 0; i < ITERATIONS; i++) {
        optimize_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_population, d_fit);
        
        // Check for CUDA errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(error));
            return -1;
        }
    }
    
    // Copy results from device to host
    cudaMemcpy(h_population, d_population, sizeof(float) * POPULATION_SIZE * DIMENSIONS, 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fit, d_fit, sizeof(float) * POPULATION_SIZE, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_population);
    cudaFree(d_fit);
    
    // Find best solution
    int best_idx = 0;
    for (int i = 1; i < POPULATION_SIZE; i++) {
        if (h_fit[i] > h_fit[best_idx]) {
            best_idx = i;
        }
    }
    
    // Stop timing
    clock_t end = clock();
    double execution_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Print results
    printf("\n=== CUDA PARALLEL OPTIMIZATION RESULTS ===\n");
    printf("Temperature Level: %.2f°C\n", h_population[best_idx][0]);
    printf("pH Level: %.2f\n", h_population[best_idx][1]);
    printf("Nutrient Level: %.2f mg/L\n", h_population[best_idx][2]);
    printf("Oxygen Level: %.2f%%\n", h_population[best_idx][3]);
    printf("Moisture Level: %.2f%%\n", h_population[best_idx][4]);
    printf("Ideal Growth Score: %.2f\n", h_fit[best_idx]);
    printf("Execution Time: %.6f seconds\n", execution_time);
    printf("CUDA Blocks: %d\n", BLOCKS);
    printf("Threads per Block: %d\n", THREADS_PER_BLOCK);
    printf("Total Threads: %d\n", BLOCKS * THREADS_PER_BLOCK);
    printf("==========================================\n");
    
    return 0;
}
