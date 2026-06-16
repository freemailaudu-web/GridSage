# Solver Setup

GridSage uses Pyomo for baseline optimal scheduling. The default solver is Gurobi.

## What Works Without Gurobi

- Backend import and health API
- Session creation
- Scenario parsing and validation
- Scenario Skill matching
- Grid snapshot endpoint
- Frontend UI build and local interaction
- Lightweight smoke tests

## What Requires Gurobi

- `python test_run.py`
- Full baseline optimal scheduling
- Baseline-vs-RL evaluation workflows that call the baseline optimizer

## Windows Checklist

1. Install Gurobi Optimizer.
2. Activate a valid Gurobi license.
3. Ensure `gurobi_cl` is available in a terminal.
4. Ensure the Python binding matches the Python version used by GridSage.
5. Test the binding:

```powershell
python -c "import gurobipy as gp; print(gp.gurobi.version())"
python -c "from pyomo.environ import SolverFactory; print(SolverFactory('gurobi').available())"
```

If `SolverFactory('gurobi').available()` is true but `import gurobipy` fails with a DLL error, Pyomo can see the solver wrapper but the Gurobi Python extension is not correctly loadable.

## Common Failure

```text
ImportError: DLL load failed while importing _batch
pyomo.common.errors.ApplicationError: Solver (gurobi) did not exit normally
```

This usually means the Gurobi installation, Python binding, DLL path, or license is not configured correctly. It does not necessarily mean the GridSage source code is incomplete.

## Future Improvement

To lower the entry barrier, consider supporting an optional open-source solver path:

- HiGHS for LP/QP and selected MIP workflows
- CBC for open-source MIP
- GLPK for small examples

Before advertising an open-source fallback, validate result quality and model compatibility on small deterministic cases.
