"""
SAT Solver Comparison: Resolution, DP, and DPLL Algorithms
"""
import time
import tracemalloc
import random
from typing import List, Set, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------
# Core Data Structures
# --------------------------

class SATInstance:
    def __init__(self, cnf_file: str):
        self.variables = set()
        self.clauses = []
        self._parse_dimacs(cnf_file)
        
    def _parse_dimacs(self, filename: str):
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('c') or line.startswith('p'):
                    continue
                clause = [int(x) for x in line.split()[:-1]]
                if clause:
                    self.clauses.append(clause)
                    for lit in clause:
                        self.variables.add(abs(lit))

# --------------------------
# Algorithm Implementations
# --------------------------

def resolution_solver(instance: SATInstance) -> Tuple[bool, float, int]:
    """Resolution-based SAT solver"""
    start_time = time.time()
    tracemalloc.start()
    
    clauses = [frozenset(clause) for clause in instance.clauses]
    new_clauses = set(clauses)
    steps = 0
    
    while True:
        derived = set()
        for c1 in new_clauses:
            for c2 in new_clauses:
                if c1 == c2:
                    continue
                # Find complementary literals
                for lit in c1:
                    if -lit in c2:
                        resolvent = (c1 - {lit}) | (c2 - {-lit})
                        if not resolvent:
                            # Empty clause found - unsatisfiable
                            current_mem = tracemalloc.get_traced_memory()[1]
                            tracemalloc.stop()
                            return False, time.time() - start_time, current_mem
                        derived.add(resolvent)
                        steps += 1
        
        if not derived - new_clauses:
            # No new clauses - satisfiable
            current_mem = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            return True, time.time() - start_time, current_mem
        
        new_clauses.update(derived)

def dp_solver(instance: SATInstance) -> Tuple[bool, float, int]:
    """Davis-Putnam algorithm"""
    start_time = time.time()
    tracemalloc.start()
    
    clauses = [set(clause) for clause in instance.clauses]
    variables = instance.variables.copy()
    steps = 0
    
    while variables:
        var = variables.pop()
        pos_clauses = [c for c in clauses if var in c]
        neg_clauses = [c for c in clauses if -var in c]
        
        # Generate resolvents
        new_clauses = []
        for pc in pos_clauses:
            for nc in neg_clauses:
                resolvent = (pc - {var}) | (nc - {-var})
                if not resolvent:
                    current_mem = tracemalloc.get_traced_memory()[1]
                    tracemalloc.stop()
                    return False, time.time() - start_time, current_mem
                new_clauses.append(resolvent)
                steps += 1
        
        # Remove old clauses and add resolvents
        clauses = [c for c in clauses if var not in c and -var not in c] + new_clauses
        
    current_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return True, time.time() - start_time, current_mem

def dpll_solver(instance: SATInstance, 
                heuristic: str = 'random') -> Tuple[Optional[bool], float, int]:
    """DPLL solver with configurable heuristics"""
    start_time = time.time()
    tracemalloc.start()
    
    # Convert to set representation
    clauses = [set(clause) for clause in instance.clauses]
    assignment = {}
    steps = 0
    
    def _dpll(clauses, assignment):
        nonlocal steps
        
        # Unit propagation
        changed = True
        while changed:
            changed = False
            unit_clauses = [c for c in clauses if len(c) == 1]
            for uc in unit_clauses:
                lit = next(iter(uc))
                var = abs(lit)
                val = lit > 0
                assignment[var] = val
                steps += 1
                
                # Remove satisfied clauses and false literals
                new_clauses = []
                for c in clauses:
                    if lit in c:
                        continue  # Clause satisfied
                    new_c = c - {-lit}
                    if not new_c:
                        return None  # Conflict
                    new_clauses.append(new_c)
                
                if len(new_clauses) < len(clauses):
                    changed = True
                clauses = new_clauses
        
        # Check for completion
        if not clauses:
            return True
        
        # Pure literal elimination
        literals = {lit for c in clauses for lit in c}
        pure_literals = set()
        for lit in literals:
            if -lit not in literals:
                pure_literals.add(lit)
        
        if pure_literals:
            for lit in pure_literals:
                var = abs(lit)
                assignment[var] = lit > 0
                steps += 1
            clauses = [c for c in clauses if not any(l in pure_literals for l in c)]
            return _dpll(clauses, assignment)
        
        # Variable selection heuristic
        if heuristic == 'random':
            var = random.choice([abs(l) for c in clauses for l in c])
        elif heuristic == 'moms':  # Maximum Occurrence in Minimum Size clauses
            min_len = min(len(c) for c in clauses)
            candidates = [l for c in clauses if len(c) == min_len for l in c]
            var = abs(max(set(candidates), key=candidates.count))
        elif heuristic == 'jw':  # Jeroslow-Wang
            jw_scores = {}
            for c in clauses:
                for lit in c:
                    jw_scores[abs(lit)] = jw_scores.get(abs(lit), 0) + 2**(-len(c))
            var = max(jw_scores.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError(f"Unknown heuristic: {heuristic}")
        
        # Recursive case
        for val in [True, False]:
            new_assignment = assignment.copy()
            new_assignment[var] = val
            new_clauses = []
            for c in clauses:
                if var in c and val or -var in c and not val:
                    continue  # Clause satisfied
                new_c = c - {var, -var}
                if not new_c:
                    break  # Conflict
                new_clauses.append(new_c)
            else:
                result = _dpll(new_clauses, new_assignment)
                if result is not None:
                    return result
        
        return None
    
    result = _dpll(clauses, assignment)
    current_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return result, time.time() - start_time, current_mem

# --------------------------
# Test Framework
# --------------------------

def run_benchmarks(test_dir: str, output_dir: str):
    """Run all algorithms on benchmark files"""
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for filename in os.listdir(test_dir):
        if not filename.endswith('.cnf'):
            continue
            
        filepath = os.path.join(test_dir, filename)
        instance = SATInstance(filepath)
        print(f"\nTesting {filename} with {len(instance.variables)} variables, {len(instance.clauses)} clauses")
        
        # Run all solvers
        for algo in ['resolution', 'dp', 'dpll']:
            for heuristic in ['random', 'moms', 'jw'] if algo == 'dpll' else [None]:
                print(f"Running {algo}" + (f" ({heuristic})" if heuristic else ""))
                
                try:
                    if algo == 'resolution':
                        result, time_used, memory = resolution_solver(instance)
                    elif algo == 'dp':
                        result, time_used, memory = dp_solver(instance)
                    elif algo == 'dpll':
                        result, time_used, memory = dpll_solver(instance, heuristic)
                    
                    results.append({
                        'file': filename,
                        'algorithm': algo,
                        'heuristic': heuristic,
                        'result': result,
                        'time': time_used,
                        'memory': memory,
                        'vars': len(instance.variables),
                        'clauses': len(instance.clauses)
                    })
                    
                    print(f"  Result: {'SAT' if result else 'UNSAT' if result is False else 'UNKNOWN'}")
                    print(f"  Time: {time_used:.3f}s")
                    print(f"  Memory: {memory / 1024:.1f} KB")
                
                except Exception as e:
                    print(f"  Error: {str(e)}")
    
    # Save results
    with open(os.path.join(output_dir, 'results.csv'), 'w') as f:
        f.write("file,algorithm,heuristic,result,time,memory,vars,clauses\n")
        for r in results:
            f.write(f"{r['file']},{r['algorithm']},{r['heuristic'] or ''},{r['result']},")
            f.write(f"{r['time']},{r['memory']},{r['vars']},{r['clauses']}\n")
    
    return results

def visualize_results(results: list, output_dir: str):
    """Generate performance comparison plots"""
    # Prepare data
    algorithms = sorted(set(r['algorithm'] for r in results))
    heuristics = sorted(set(r['heuristic'] for r in results if r['heuristic']))
    
    # Time comparison
    plt.figure(figsize=(10, 6))
    for algo in algorithms:
        algo_times = [r['time'] for r in results if r['algorithm'] == algo and r['heuristic'] is None]
        if algo_times:
            plt.plot(algo_times, label=algo, marker='o')
    
    plt.xlabel('Benchmark Instance')
    plt.ylabel('Time (seconds)')
    plt.yscale('log')
    plt.title('SAT Solver Runtime Comparison')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'runtime_comparison.png'))
    plt.close()
    
    # Memory comparison
    plt.figure(figsize=(10, 6))
    for algo in algorithms:
        algo_mem = [r['memory'] / 1024 for r in results if r['algorithm'] == algo and r['heuristic'] is None]
        if algo_mem:
            plt.plot(algo_mem, label=algo, marker='o')
    
    plt.xlabel('Benchmark Instance')
    plt.ylabel('Memory (KB)')
    plt.yscale('log')
    plt.title('SAT Solver Memory Usage')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'memory_comparison.png'))
    plt.close()
    
    # DPLL Heuristics comparison
    if any(r['algorithm'] == 'dpll' for r in results):
        plt.figure(figsize=(10, 6))
        for heuristic in heuristics:
            h_times = [r['time'] for r in results if r['algorithm'] == 'dpll' and r['heuristic'] == heuristic]
            if h_times:
                plt.plot(h_times, label=heuristic, marker='o')
        
        plt.xlabel('Benchmark Instance')
        plt.ylabel('Time (seconds)')
        plt.yscale('log')
        plt.title('DPLL Heuristic Performance')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'dpll_heuristics.png'))
        plt.close()

# --------------------------
# Main Execution
# --------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SAT Solver Comparison')
    parser.add_argument('--test-dir', default='benchmarks', help='Directory with CNF files')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    args = parser.parse_args()
    
    results = run_benchmarks(args.test_dir, args.output_dir)
    visualize_results(results, args.output_dir)
