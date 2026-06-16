# GridSage

GridSage is a natural-language-driven distribution grid simulation and scenario configuration platform. It helps users describe distribution-grid scenarios in natural language, convert those instructions into structured simulation configurations, validate scenario safety, and run simulation workflows through an interactive frontend-backend system.

The project combines a web-based interface, a backend scenario parsing and validation service, and a distribution-grid simulation core. It is intended for research and development on intelligent power-grid configuration, scenario generation, simulation execution, and reinforcement-learning-based evaluation.

## Key Features

- Natural language scenario configuration
- Structured scenario state management
- Scenario Skill based intent recognition
- Safety validation for grid simulation parameters
- Frontend-backend interactive workflow
- Distribution grid simulation execution
- Simulation result visualization and interpretation
- Reinforcement learning training and evaluation workflows
- Logging support for reproducibility and later analysis

## System Architecture

GridSage is organized as a full-stack research prototype:

```text
GridSage/
|-- backend/                 # Backend API, scenario parser, validation, executor
|   |-- main.py              # FastAPI backend entry
|   |-- agent.py             # Natural language configuration logic
|   |-- executor.py          # Simulation execution interface
|   |-- schema.py            # Scenario state and API schemas
|   |-- validation.py        # General safety validation
|   `-- skills/              # Scenario Skill definitions
|-- frontend/                # Vue / Vite frontend
|   |-- src/                 # Frontend source code
|   `-- package.json         # Frontend dependencies and scripts
|-- tests/                   # Lightweight regression tests
|-- docs/                    # Solver and release documentation
|-- test_run.py              # Full baseline simulation test
`-- README.md
```

## Main Workflow

1. The user enters a natural language instruction.
2. The backend identifies scenario intent and extracts configuration parameters.
3. The system updates the structured scenario state.
4. Scenario validation checks whether the configuration is safe and executable.
5. The simulation core runs the configured distribution-grid scenario.
6. The frontend displays scenario state, validation results, topology information, and simulation metrics.

## Example Instructions

```text
increase PV generation to 1.5x
set load from 18:00 to 22:00 to 1.5x
add 12 EV chargers at bus 18
train PPO for 5000 steps
use the trained SAC model
```

## Current Distribution Notice

At the current stage, GridSage supports local use through a downloaded source package.

This project is not currently distributed as a Python package, npm package, Docker image, or online hosted service.

## Requirements

Recommended environment:

- Windows
- Python 3.11
- Node.js 18 or later
- npm
- Gurobi, only for full baseline optimization

The backend API, frontend UI, scenario validation, skill matching, and grid snapshot can be checked without Gurobi. Full baseline optimization requires a working solver environment.

## Installation and Local Usage

### 1. Create a Python Virtual Environment

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2. Install Backend Dependencies

```powershell
pip install -r backend\requirements.txt
```

### 3. Install Frontend Dependencies

```powershell
cd frontend
npm install
cd ..
```

### 4. Start the Backend

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The backend API will start at:

```text
http://127.0.0.1:8000
```

The health endpoint is:

```text
http://127.0.0.1:8000/api/health
```

### 5. Start the Frontend

Open another terminal:

```powershell
cd frontend
npm run dev
```

The frontend interface will start at:

```text
http://localhost:3000
```

## Minimal Verification

These checks do not require a full Gurobi optimization run:

```powershell
python -B -m unittest -v tests.test_backend_smoke

cd frontend
npm run build
cd ..
```

## Full Simulation Check

```powershell
python test_run.py
```

This command runs the baseline optimization path and usually requires Gurobi. If it fails with a Gurobi DLL, license, or solver error while the smoke tests pass, the source package is probably intact and the solver environment needs attention. See [docs/solver.md](docs/solver.md).

## Documentation

- [Solver setup](docs/solver.md)
- [Release checklist](docs/release_checklist.md)
- [Chinese README](README.zh-CN.md)

## Research Purpose

GridSage is currently a research-oriented prototype. It is suitable for studying:

- Natural language based power-grid scenario configuration
- Human-computer interaction for distribution-grid simulation
- Structured scenario state representation
- Scenario validation and safety checking
- Simulation workflow automation
- Reinforcement learning training and evaluation interfaces for grid control

Experimental results, benchmark comparisons, and performance claims should be added only when supported by verified experiment records.

## License

Please add a license before public release. Common options include MIT License, Apache License 2.0, and GPL-3.0 License.

## Citation

If this project is used in academic work, please cite the related paper or project repository once available.
