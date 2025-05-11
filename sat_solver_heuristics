import time
import tracemalloc
import os
import numpy as np
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
from typing import Tuple, Optional, List, Set, Dict

class SATInstance:
    __slots__ = ['variables', 'clauses']

    def __init__(self, cnf_file: str):
        self.variables = set()
        self.clauses = []
        self._parse_dimacs(cnf_file)
        
    def _parse_dimacs(self, filename: str):
        """Optimized DIMACS parser for parsing CNF files."""
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith(('c', 'p')):
                    continue
                clause = list({int(x) for x in line.split()[:-1]})
                if clause:
                    self.clauses.append(clause)
                    self.variables.update(abs(lit) for lit in clause)

# --------------------------
# Optimized Algorithms
# --------------------------

def dpll_solver(instance: SATInstance, heuristic: str = 'jw') -> Tuple[Optional[bool], float, int, List[float]]:
    """Optimized DPLL with heuristic options"""
    tracemalloc.start()
    start = time.perf_counter()
    
    clauses = [np.array(c, dtype=np.int32) for c in instance.clauses]
    assignment = {}
    
    memory_usage = []
    
    def _dpll(clauses, assignment):
        units = [c[0] for c in clauses if len(c) == 1]
        while units:
            lit = units.pop()
            var, val = abs(lit), lit > 0
            assignment[var] = val
            
            new_clauses = []
            for c in clauses:
                if lit in c: continue
                new_c = c[c != -lit]
                if len(new_c) == 0: return None
                if len(new_c) == 1: units.append(new_c[0])
                new_clauses.append(new_c)
            clauses = new_clauses
        
        if not clauses: return True
        
        if heuristic == 'jw':
            # Jeroslow-Wang heuristic
            scores = {}
            for c in clauses:
                scores.update({abs(l): scores.get(abs(l), 0) + 2**-len(c) for l in c})
            var = max(scores.items(), key=lambda x: x[1])[0]
        elif heuristic == 'moms':
            # MOMS (Maximum Occurrence in Clauses Heuristic)
            var = max(clauses, key=lambda c: len(c))[0]  # Simplified MOMS
        else:
            # Default heuristic
            var = abs(clauses[0][0])
            
        for val in [True, False]:
            result = _dpll([c[c != var] if val else c[c != -var] for c in clauses], 
                          {**assignment, var: val})
            if result is not None:
                return result
        return None
    
    result = _dpll(clauses, assignment)
    
    # Track memory usage during each iteration
    memory_usage.append(tracemem_get_current())
    
    tracemalloc.stop()
    return result, time.perf_counter() - start, tracemalloc.get_traced_memory()[0], memory_usage

def tracemem_get_current() -> float:
    """Get the current memory usage in KB."""
    return tracemalloc.get_traced_memory()[1] / 1024  # Convert to KB

# --------------------------
# Benchmark Runner
# --------------------------

def visualize_comparison(benchmark_name: str, df):
    """Generate and save a comparison plot for heuristics"""
    plt.figure(figsize=(10, 6))
    for heuristic in df['Heuristic'].unique():
        subset = df[df['Heuristic'] == heuristic]
        
        # Plot time vs memory for each heuristic
        for _, row in subset.iterrows():
            plt.scatter(row['Time (s)'], row['Memory (KB)'], label=f"{heuristic} - {row['Benchmark']}", alpha=0.7)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Memory (KB)')
    plt.title(f'Heuristic Comparison: Time vs Memory ({benchmark_name})')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to the results folder
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)  # Ensure the 'results' folder exists
    plt.savefig(f"{result_dir}/{benchmark_name}_heuristics_comparison.png")
    plt.close()  # Close the plot after saving it

def run_benchmarks(test_dir: str, timeout: int = 300):
    """Optimized benchmark runner with heuristic comparison"""
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Directory {test_dir} not found.")
    
    results = []
    for file in os.listdir(test_dir):
        if not file.endswith('.cnf'): continue
        
        instance = SATInstance(os.path.join(test_dir, file))
        print(f"\nBenchmark: {file} (Vars: {len(instance.variables)}, Clauses: {len(instance.clauses)})")
        
        benchmark_results = []
        for heuristic in ['jw', 'moms']:  # Add more heuristics here if needed
            try:
                result, time_used, mem, memory_usage = dpll_solver(instance, heuristic=heuristic)
                print(f"DPLL-{heuristic}  | {'SAT' if result else 'UNSAT':5} | {time_used:6.2f}s | {mem/1024:6.1f} KB")
                benchmark_results.append((file, "dpll_solver", result, time_used, mem, memory_usage, heuristic))
            except Exception as e:
                print(f"DPLL-{heuristic} failed: {str(e)}")
                continue
        
        # Create a DataFrame and visualize the heuristic comparison
        import pandas as pd
        df = pd.DataFrame(benchmark_results, columns=["Benchmark", "Algorithm", "Result", "Time (s)", "Memory (KB)", "Memory Usage (Over Time)", "Heuristic"])
        visualize_comparison(file, df)  # Save the comparison plot for this benchmark
                
    return results

if __name__ == "__main__":
    results = run_benchmarks("benchmarks")
