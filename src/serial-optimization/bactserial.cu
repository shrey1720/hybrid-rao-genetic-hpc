/**
 * @file bactserial.cu
 * @brief Serial implementation of Hybrid RAO-Genetic Algorithm for Fungal Growth Optimization
 * @author Shrey Panchasara
 * @date 17/08/2025
 * 
 * This file implements the hybrid RAO-Genetic Algorithm in serial mode for optimizing
 * fungal growth parameters. The algorithm combines Rao's optimization method with
 * genetic algorithm operations to balance exploration and exploitation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Algorithm Parameters
#define POPULATION_SIZE 500      ///< Number of individuals in population
#define ITERATIONS 1000000       ///< Maximum number of optimization iterations
#define DIMENSIONS 5             ///< Number of environmental parameters

// Environmental Parameter Bounds
#define TEMP_MIN 20.0           ///< Minimum temperature (°C)
#define TEMP_MAX 60.0           ///< Maximum temperature (°C)
#define PH_MIN 3.0              ///< Minimum pH level
#define PH_MAX 9.0              ///< Maximum pH level
#define NUTRIENT_MIN 5.0        ///< Minimum nutrient level (mg/L)
#define NUTRIENT_MAX 100.0      ///< Maximum nutrient level (mg/L)
#define OXYGEN_MIN 0.0          ///< Minimum oxygen level (%)
#define OXYGEN_MAX 100.0        ///< Maximum oxygen level (%)
#define MOISTURE_MIN 85.0       ///< Minimum moisture level (%)
#define MOISTURE_MAX 95.0       ///< Maximum moisture level (%)

// Genetic Algorithm Parameters
#define MUTATION_RATE 0.3       ///< Probability of mutation
#define CROSSOVER_RATE 0.7      ///< Probability of crossover

/**
 * @brief Generate random float value within specified range
 * @param min Minimum value
 * @param max Maximum value
 * @return Random float value
 */
float randomval(float min, float max) {
    return min + ((float)rand() / RAND_MAX) * (max - min);
}

/**
 * @brief Calculate fungal growth score based on environmental parameters
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
float growth(float t, float p, float n, float o, float m) {
    return -(t - 37) * (t - 37) - (p - 7) * (p - 7) - 
           (n - 60) * (n - 60) - (o - 21) * (o - 21) - 
           (m - 90) * (m - 90) + 100;
}

/**
 * @brief Apply boundary constraints to parameter values
 * @param val Pointer to value to constrain
 * @param min Minimum allowed value
 * @param max Maximum allowed value
 */
void boundary(float *val, float min, float max) {
    if (*val < min) *val = min;
    if (*val > max) *val = max;
}

/**
 * @brief Apply RAO mutation to individual parameters
 * 
 * RAO mutation uses the difference between best and worst solutions
 * to guide the search direction, balancing exploration and exploitation.
 * 
 * @param local Current individual parameters
 * @param best Best solution parameters
 * @param worst Worst solution parameters
 */
void rao_mutation(float local[DIMENSIONS], float best[DIMENSIONS], float worst[DIMENSIONS]) {
    if (randomval(0, 1) <= MUTATION_RATE) {
        for (int i = 0; i < DIMENSIONS; i++) {
            float r = randomval(0, 1);
            local[i] = local[i] + r * (best[i] - worst[i]);
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
 * @brief Apply genetic crossover between best and worst solutions
 * 
 * Crossover creates diversity by combining genetic material from
 * different individuals, helping escape local optima.
 * 
 * @param best Best solution parameters
 * @param worst Worst solution parameters
 */
void crossover(float best[DIMENSIONS], float worst[DIMENSIONS]) {
    if (randomval(0, 1) <= CROSSOVER_RATE) {
        for (int i = 0; i < DIMENSIONS; i++) {
            worst[i] = (best[i] + worst[i]) / 2;
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
 * @brief Main optimization loop implementing the hybrid algorithm
 * 
 * The optimization process:
 * 1. Identifies best and worst solutions
 * 2. Applies RAO mutation to all individuals
 * 3. Performs genetic crossover
 * 4. Updates fitness values
 * 5. Repeats for specified iterations
 * 
 * @param population Array of population parameters
 * @param fit Array of fitness values
 */
void optimize(float population[POPULATION_SIZE][DIMENSIONS], float fit[POPULATION_SIZE]) {
    for (int iteration = 0; iteration < ITERATIONS; iteration++) {
        // Find best and worst solutions
        int best_idx = 0;
        int worst_idx = 0;
        
        for (int j = 1; j < POPULATION_SIZE; j++) {
            if (fit[j] > fit[best_idx]) best_idx = j;
            if (fit[j] < fit[worst_idx]) worst_idx = j;
        }
        
        // Apply RAO mutation to all individuals
        for (int j = 0; j < POPULATION_SIZE; j++) {
            rao_mutation(population[j], population[best_idx], population[worst_idx]);
            fit[j] = growth(population[j][0], population[j][1], population[j][2], 
                           population[j][3], population[j][4]);
        }
        
        // Apply genetic crossover
        crossover(population[best_idx], population[worst_idx]);
        fit[worst_idx] = growth(population[worst_idx][0], population[worst_idx][1], 
                               population[worst_idx][2], population[worst_idx][3], 
                               population[worst_idx][4]);
    }
}

/**
 * @brief Main function - Entry point of the program
 * @return Exit status
 */
int main() {
    // Initialize random seed
    srand(time(0));
    
    // Start timing
    clock_t start = clock();
    
    // Declare population and fitness arrays
    float population[POPULATION_SIZE][DIMENSIONS];
    float fit[POPULATION_SIZE];
    
    // Initialize population with random values
    printf("Initializing population with %d individuals...\n", POPULATION_SIZE);
    for (int i = 0; i < POPULATION_SIZE; i++) {
        population[i][0] = randomval(TEMP_MIN, TEMP_MAX);      // Temperature
        population[i][1] = randomval(PH_MIN, PH_MAX);          // pH
        population[i][2] = randomval(NUTRIENT_MIN, NUTRIENT_MAX); // Nutrients
        population[i][3] = randomval(OXYGEN_MIN, OXYGEN_MAX);  // Oxygen
        population[i][4] = randomval(MOISTURE_MIN, MOISTURE_MAX); // Moisture
        
        // Calculate initial fitness
        fit[i] = growth(population[i][0], population[i][1], population[i][2], 
                       population[i][3], population[i][4]);
    }
    
    printf("Starting optimization with %d iterations...\n", ITERATIONS);
    
    // Run optimization
    optimize(population, fit);
    
    // Find best solution
    int best_idx = 0;
    for (int i = 1; i < POPULATION_SIZE; i++) {
        if (fit[i] > fit[best_idx]) best_idx = i;
    }
    
    // Stop timing
    clock_t stop = clock();
    double execution_time = ((double)(stop - start)) / CLOCKS_PER_SEC;
    
    // Print results
    printf("\n=== OPTIMIZATION RESULTS ===\n");
    printf("Temperature Level: %.2f°C\n", population[best_idx][0]);
    printf("pH Level: %.2f\n", population[best_idx][1]);
    printf("Nutrient Level: %.2f mg/L\n", population[best_idx][2]);
    printf("Oxygen Level: %.2f%%\n", population[best_idx][3]);
    printf("Moisture Level: %.2f%%\n", population[best_idx][4]);
    printf("Ideal Growth Score: %.2f\n", fit[best_idx]);
    printf("Execution Time: %.6f seconds\n", execution_time);
    printf("===========================\n");
    
    return 0;
}
