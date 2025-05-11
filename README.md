# SAT Solver Benchmarking

This project provides an implementation of three popular SAT solvers and benchmarks them on a set of SAT instances. The solvers implemented are:

- **DPLL (Davis-Putnam-Logemann-Loveland)** Solver
- **DP (Davis-Putnam)** Solver
- **Resolution** Solver

The solvers are applied to SAT instances encoded in the **DIMACS CNF** format. The benchmarking results, including execution time and memory consumption, are saved and visualized for performance comparison.

## Solvers Implemented

### 1. **DPLL Solver (Davis-Putnam-Logemann-Loveland)**
   The DPLL algorithm is an optimized version of the **Davis-Putnam** algorithm. It uses **backtracking** and **unit propagation** to find a satisfying assignment for a given Boolean formula in Conjunctive Normal Form (CNF).

   - The algorithm can be enhanced using different heuristics (like **Jeroslow-Wang (JW)** and **Maximum Occurrence in Clauses (MOMS)**).
   - The solver tracks the **time** and **memory** consumption for each execution.

### 2. **DP Solver (Davis-Putnam)**
   The DP algorithm is an earlier version of DPLL. It performs **variable elimination** and **unit propagation** to check satisfiability.

   - The DP solver is straightforward but can be slower for large problems.

### 3. **Resolution Solver**
   The **Resolution** algorithm applies the **resolution rule** to derive new clauses from existing ones. The algorithm proceeds iteratively, combining clauses until either a contradiction is found (indicating unsatisfiability) or all possible clauses have been derived.

   - The **clause indexing** and **deduplication** of clauses helps improve performance.

## Key Features

- **Solvers**: DPLL, DP, Resolution
- **Benchmarking**: Time and memory usage are measured and compared.
- **Visualization**: The results of the benchmark are visualized with **Matplotlib** to compare the time vs. memory usage for different solvers.
- **DIMACS CNF**: The project supports reading SAT instances from CNF files in the **DIMACS format**.
- **Performance Comparison**: A graphical comparison of the solvers' performance (time and memory) is automatically generated.

## Requirements

- Python 3.x
- `numpy`
- `matplotlib`
- `pandas` (for results analysis and visualization)

You can install the required dependencies using **pip**:

```bash
pip install numpy matplotlib pandas

```
## Project Structure

```bash
sat_solver_benchmark/
│
├── benchmarks/              # Directory containing the benchmark CNF files
├── results/                 # Directory where benchmark results and plots will be saved
├── sat_solver.py            # Main script implementing the solvers
└── README.md                # This file
```

## Output (Example)
```python
Benchmark: sample1.cnf (Vars: 50, Clauses: 150)
DPLL-jw     | SAT   |  1.23s  |  450.3 KB
DPLL-moms   | SAT   |  1.05s  |  410.1 KB
DP          | UNSAT |  0.78s  |  330.2 KB
Resolution  | SAT   |  2.12s  |  600.8 KB
```
