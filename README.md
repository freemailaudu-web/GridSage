# GridSage

GridSage is a natural-language-driven distribution grid simulation and scenario configuration platform. It is designed to help users describe power distribution network scenarios in natural language, convert user instructions into structured simulation configurations, validate scenario safety, and execute simulation workflows through an interactive frontend-backend system.

The project combines a web-based interface, a backend scenario parsing and validation service, and a distribution grid simulation core. It is intended for research and development on intelligent power grid configuration, scenario generation, simulation execution, and future reinforcement learning based evaluation.

## Key Features

* Natural language scenario configuration
* Structured scenario state management
* Scenario Skill based intent recognition
* Safety validation for grid simulation parameters
* Frontend-backend interactive workflow
* Distribution grid simulation execution
* Simulation result visualization and interpretation
* Support for reinforcement learning training and evaluation workflows
* Logging support for reproducibility and later analysis

## Bilingual Availability

GridSage can be used in both English and Chinese versions.

The English version is suitable for international demonstrations, academic presentation, and English-based natural language interaction. The Chinese version is suitable for Chinese users, Chinese academic writing, and Chinese natural language scenario configuration.

Users may choose the version that best fits their interaction language and research context.

## System Architecture

GridSage is organized as a full-stack research prototype:

```text
GridSage/
├── backend/                 # Backend API, scenario parser, validation logic, simulation executor
│   ├── main.py              # FastAPI backend entry
│   ├── agent.py             # Natural language configuration logic
│   ├── executor.py          # Simulation execution interface
│   ├── skill_*.py           # Scenario Skill related modules
│   └── vgridsim_core/       # Distribution grid simulation core
│
├── frontend/                # Web frontend
│   ├── src/                 # Frontend source code
│   ├── package.json         # Frontend dependencies and scripts
│   └── ...
│
├── tests/                   # Regression tests
├── start_lvgs.py            # One-click local startup script
└── README.md
```

## Main Workflow

A typical GridSage workflow includes the following steps:

1. The user enters a natural language instruction.
2. The backend identifies the scenario intent and extracts configuration parameters.
3. The system updates the structured scenario state.
4. Scenario validation checks whether the configuration is safe and executable.
5. The simulation core runs the configured distribution grid scenario.
6. The frontend displays the scenario state, validation results, topology information, and simulation metrics.

## Example Instructions

Example English instructions:

```text
increase PV generation to 1.5x
set load from 18:00 to 22:00 to 1.5x
add 12 EV chargers at bus 18
train PPO for 5000 steps
use the trained SAC model
```

Example Chinese instructions are also supported in the Chinese version:

```text
将光伏出力提高到1.5倍
把18点到22点的负荷设置为1.5倍
在18号节点添加12个电动汽车充电桩
训练PPO模型5000步
使用训练好的SAC模型进行评估
```

## Current Distribution Notice

At the current stage, GridSage only supports local use through a downloaded ZIP package.

Please download the compressed package, extract it locally, and run the project from the extracted folder.

This project is not currently distributed as a Python package, npm package, Docker image, or online hosted service.

## Requirements

The project requires both Python and Node.js.

Recommended environment:

```text
Python 3.8 or later
Node.js 16 or later
npm
```

The backend is based on Python, while the frontend is based on Vite.

## Installation and Local Usage

### 1. Download and Extract

Download the GridSage ZIP package and extract it to a local folder.

```text
GridSage-main.zip
```

After extraction, enter the project root directory.

### 2. Create a Python Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment.

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Backend Dependencies

```bash
pip install -r backend/requirements.txt
pip install -r backend/vgridsim_core/requirements.txt
```

### 4. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 5. Start the System

From the project root directory, run:

```bash
python start_lvgs.py
```

The backend API will start at:

```text
http://127.0.0.1:8000
```

The frontend interface will start at:

```text
http://localhost:3000
```

Open the frontend address in a browser to use GridSage.

## Backend Manual Startup

If you want to start the backend manually:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

## Frontend Manual Startup

If you want to start the frontend manually:

```bash
cd frontend
npm run dev
```

## Research Purpose

GridSage is currently a research-oriented prototype. It is suitable for studying:

* Natural language based power grid scenario configuration
* Human-computer interaction for distribution grid simulation
* Structured scenario state representation
* Scenario validation and safety checking
* Simulation workflow automation
* Reinforcement learning training and evaluation interfaces for grid control

The current version focuses on system architecture, scenario configuration, validation workflow, and simulation execution. Experimental results, benchmark comparisons, and performance claims should be added only when supported by verified experiment records.

## License

Please add a license before public release.

Common options include:

```text
MIT License
Apache License 2.0
GPL-3.0 License
```

Choose a license according to your intended open source usage and redistribution policy.

## Citation

If this project is used in academic work, please cite the related paper or project repository once available.

## Contact

For questions, issues, or collaboration, please use the GitHub Issues page after the repository is published.
