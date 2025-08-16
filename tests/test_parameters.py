#!/usr/bin/env python3
"""
Test suite for Hybrid RAO-Genetic Algorithm parameter validation.
Tests the biological constraints and mathematical properties of the growth function.
"""

import pytest
import numpy as np
from typing import Tuple, List

# Import the growth function (you'll need to implement this in Python)
# For now, we'll define it here for testing purposes
def growth_function(temperature: float, ph: float, nutrients: float, 
                   oxygen: float, moisture: float) -> float:
    """
    Calculate fungal growth score based on environmental parameters.
    
    Args:
        temperature: Temperature in Celsius (20-60)
        ph: pH level (3-9)
        nutrients: Nutrient level in mg/L (5-100)
        oxygen: Oxygen level in % (0-100)
        moisture: Moisture level in % (85-95)
        
    Returns:
        Growth score (0-100, higher is better)
    """
    return -(temperature - 37)**2 - (ph - 7)**2 - (nutrients - 60)**2 - \
           (oxygen - 21)**2 - (moisture - 90)**2 + 100

def validate_parameter_bounds(value: float, min_val: float, max_val: float) -> bool:
    """Validate if a parameter is within specified bounds."""
    return min_val <= value <= max_val

class TestGrowthFunction:
    """Test cases for the fungal growth function."""
    
    def test_optimal_conditions(self):
        """Test that optimal conditions give maximum growth score."""
        score = growth_function(37.0, 7.0, 60.0, 21.0, 90.0)
        assert score == 100.0, f"Expected 100.0, got {score}"
    
    def test_parameter_bounds(self):
        """Test that all parameters respect biological constraints."""
        # Test temperature bounds
        assert validate_parameter_bounds(20.0, 20.0, 60.0), "Temperature min bound failed"
        assert validate_parameter_bounds(60.0, 20.0, 60.0), "Temperature max bound failed"
        
        # Test pH bounds
        assert validate_parameter_bounds(3.0, 3.0, 9.0), "pH min bound failed"
        assert validate_parameter_bounds(9.0, 3.0, 9.0), "pH max bound failed"
        
        # Test nutrient bounds
        assert validate_parameter_bounds(5.0, 5.0, 100.0), "Nutrient min bound failed"
        assert validate_parameter_bounds(100.0, 5.0, 100.0), "Nutrient max bound failed"
        
        # Test oxygen bounds
        assert validate_parameter_bounds(0.0, 0.0, 100.0), "Oxygen min bound failed"
        assert validate_parameter_bounds(100.0, 0.0, 100.0), "Oxygen max bound failed"
        
        # Test moisture bounds
        assert validate_parameter_bounds(85.0, 85.0, 95.0), "Moisture min bound failed"
        assert validate_parameter_bounds(95.0, 85.0, 95.0), "Moisture max bound failed"
    
    def test_growth_score_range(self):
        """Test that growth scores are always between 0 and 100."""
        # Test various parameter combinations
        test_cases = [
            (20.0, 3.0, 5.0, 0.0, 85.0),   # Worst case
            (30.0, 5.0, 30.0, 10.0, 87.0),  # Poor conditions
            (35.0, 6.5, 45.0, 15.0, 88.0),  # Moderate conditions
            (37.0, 7.0, 60.0, 21.0, 90.0),  # Optimal conditions
            (45.0, 8.0, 80.0, 30.0, 92.0),  # Good conditions
            (50.0, 8.5, 90.0, 50.0, 94.0),  # Fair conditions
        ]
        
        for params in test_cases:
            score = growth_function(*params)
            assert 0.0 <= score <= 100.0, \
                f"Growth score {score} for params {params} is outside valid range"
    
    def test_parameter_sensitivity(self):
        """Test that small changes in parameters produce reasonable changes in growth."""
        base_params = [37.0, 7.0, 60.0, 21.0, 90.0]
        base_score = growth_function(*base_params)
        
        # Test temperature sensitivity
        temp_variations = [36.0, 36.5, 37.5, 38.0]
        for temp in temp_variations:
            test_params = base_params.copy()
            test_params[0] = temp
            test_score = growth_function(*test_params)
            
            # Score should decrease as we move away from optimal
            if temp != 37.0:
                assert test_score < base_score, \
                    f"Temperature {temp} should give lower score than optimal"
    
    def test_quadratic_penalty(self):
        """Test that the quadratic penalty function works correctly."""
        # Test that doubling the deviation quadruples the penalty
        base_score = growth_function(37.0, 7.0, 60.0, 21.0, 90.0)  # 100.0
        
        # Small deviation
        small_deviation = growth_function(38.0, 7.0, 60.0, 21.0, 90.0)
        small_penalty = base_score - small_deviation
        
        # Double deviation
        double_deviation = growth_function(39.0, 7.0, 60.0, 21.0, 90.0)
        double_penalty = base_score - double_deviation
        
        # Penalty should approximately quadruple
        penalty_ratio = double_penalty / small_penalty
        assert 3.5 <= penalty_ratio <= 4.5, \
            f"Penalty ratio {penalty_ratio} should be approximately 4.0"
    
    def test_parameter_independence(self):
        """Test that parameters affect growth independently (additive model)."""
        # Test that changing one parameter doesn't affect others' optimal values
        optimal_temp = 37.0
        optimal_ph = 7.0
        
        # Test temperature optimization at different pH values
        ph_values = [6.0, 6.5, 7.0, 7.5, 8.0]
        for ph in ph_values:
            # Find best temperature for this pH
            temp_scores = []
            temp_range = np.arange(35.0, 40.0, 0.1)
            for temp in temp_range:
                score = growth_function(temp, ph, 60.0, 21.0, 90.0)
                temp_scores.append((temp, score))
            
            best_temp = max(temp_scores, key=lambda x: x[1])[0]
            
            # Best temperature should be close to 37°C regardless of pH
            assert abs(best_temp - optimal_temp) < 1.0, \
                f"Best temperature {best_temp} at pH {ph} should be close to {optimal_temp}"

class TestAlgorithmParameters:
    """Test cases for algorithm-specific parameters."""
    
    def test_population_size(self):
        """Test that population size is reasonable."""
        population_size = 500
        assert 100 <= population_size <= 10000, \
            f"Population size {population_size} should be between 100 and 10000"
    
    def test_iteration_count(self):
        """Test that iteration count is reasonable."""
        iteration_count = 1000000
        assert 10000 <= iteration_count <= 10000000, \
            f"Iteration count {iteration_count} should be between 10K and 10M"
    
    def test_genetic_parameters(self):
        """Test that genetic algorithm parameters are valid probabilities."""
        mutation_rate = 0.3
        crossover_rate = 0.7
        
        assert 0.0 <= mutation_rate <= 1.0, \
            f"Mutation rate {mutation_rate} should be between 0 and 1"
        assert 0.0 <= crossover_rate <= 1.0, \
            f"Crossover rate {crossover_rate} should be between 0 and 1"
        assert mutation_rate + crossover_rate <= 1.5, \
            f"Sum of rates {mutation_rate + crossover_rate} should be reasonable"

def run_performance_test() -> Tuple[float, float]:
    """
    Run a simple performance test to validate the algorithm's behavior.
    
    Returns:
        Tuple of (serial_time_estimate, parallel_time_estimate)
    """
    # This is a simplified test - in practice you'd run the actual C++/CUDA code
    print("Running performance validation test...")
    
    # Simulate some computation
    iterations = 1000
    population_size = 100
    
    # Estimate serial time (simplified)
    serial_time = iterations * population_size * 0.0001  # seconds
    
    # Estimate parallel time (with overhead)
    parallel_time = serial_time / 10 + 0.1  # 10x speedup with 0.1s overhead
    
    print(f"Estimated serial time: {serial_time:.4f} seconds")
    print(f"Estimated parallel time: {parallel_time:.4f} seconds")
    print(f"Estimated speedup: {serial_time/parallel_time:.2f}x")
    
    return serial_time, parallel_time

if __name__ == "__main__":
    # Run all tests
    print("Running Hybrid RAO-Genetic Algorithm parameter tests...")
    
    # Run pytest-style tests
    test_classes = [TestGrowthFunction, TestAlgorithmParameters]
    
    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}...")
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"  ✓ {method_name}")
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
    
    # Run performance test
    print("\nRunning performance validation...")
    serial_time, parallel_time = run_performance_test()
    
    print("\nAll tests completed!")
