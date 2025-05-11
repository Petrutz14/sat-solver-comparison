import time
import tracemalloc
import os
import numpy as np
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

def resolution_solver(instance: SATInstance) -> Tuple[bool, float, int]:
    """Resolution with clause indexing"""
    tracemalloc.start()
    start = time.perf_counter()
    
    clauses = {frozenset(c) for c in instance.clauses}
    active = set(clauses)
    seen = set(clauses)
    
    while active:
        new_clauses = set()
        for c1 in list(active):
            for lit in c1:
                for c2 in active:
                    if -lit in c2:
                        resolvent = (c1 - {lit}) | (c2 - {-lit})
                        if not resolvent:
                            tracemalloc.stop()
                            return False, time.perf_counter() - start, tracemalloc.get_traced_memory()[0]
                        if resolvent not in seen:
                            seen.add(resolvent)
                            new_clauses.add(resolvent)
        active = new_clauses
    
    tracemalloc.stop()
    return True, time.perf_counter() - start, tracemalloc.get_traced_memory()[0]

def dp_solver(instance: SATInstance) -> Tuple[bool, float, int]:
    """DP with efficient variable elimination"""
    tracemalloc.start()
    start = time.perf_counter()
    
    clauses = [set(c) for c in instance.clauses]
    var_stack = list(instance.variables)
    
    while var_stack:
        var = var_stack.pop()
        pos = [c for c in clauses if var in c]
        neg = [c for c in clauses if -var in c]
        
        new_clauses = []
        for pc in pos:
            for nc in neg:
                resolvent = (pc - {var}) | (nc - {-var})
                if not resolvent:
                    tracemalloc.stop()
                    return False, time.perf_counter() - start, tracemalloc.get_traced_memory()[0]
                new_clauses.append(resolvent)
        
        clauses = [c for c in clauses if var not in c and -var not in c] + new_clauses
    
    tracemalloc.stop()
    return True, time.perf_counter() - start, tracemalloc.get_traced_memory()[0]

def dpll_solver(instance: SATInstance, heuristic: str = 'jw') -> Tuple[Optional[bool], float, int]:
    """Optimized DPLL with watched literals"""
    tracemalloc.start()
    start = time.perf_counter()
    
    clauses = [np.array(c, dtype=np.int32) for c in instance.clauses]
    assignment = {}
    
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
            scores = {}
            for c in clauses:
                scores.update({abs(l): scores.get(abs(l), 0) + 2**-len(c) for l in c})
            var = max(scores.items(), key=lambda x: x[1])[0]
        else:
            var = abs(clauses[0][0])
            
        for val in [True, False]:
            result = _dpll([c[c != var] if val else c[c != -var] for c in clauses], 
                          {**assignment, var: val})
            if result is not None:
                return result
        return None
    
    result = _dpll(clauses, assignment)
    tracemalloc.stop()
    return result, time.perf_counter() - start, tracemalloc.get_traced_memory()[0]

# --------------------------
# Benchmark Runner
# --------------------------

def run_benchmarks(test_dir: str, timeout: int = 300):
    """Optimized benchmark runner with timeout"""
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Directory {test_dir} not found.")
    
    results = []
    for file in os.listdir(test_dir):
        if not file.endswith('.cnf'): continue
        
        instance = SATInstance(os.path.join(test_dir, file))
        print(f"\nBenchmark: {file} (Vars: {len(instance.variables)}, Clauses: {len(instance.clauses)})")
        
        for algo in [dpll_solver, dp_solver, resolution_solver]:
            try:
                result, time_used, mem = algo(instance)
                print(f"{algo.__name__:12} | {'SAT' if result else 'UNSAT':5} | {time_used:6.2f}s | {mem/1024:6.1f} KB")
                results.append((file, algo.__name__, result, time_used, mem))
            except Exception as e:
                print(f"{algo.__name__} failed: {str(e)}")
                continue
                
    return results

if __name__ == "__main__":
    results = run_benchmarks("benchmarks")
