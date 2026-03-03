# NeuralGrasp_sim2real_project
This is the project for the course robot learning (IFT6163)

## Directory Structure

```
NeuralGrasp_sim2real_project/
├── configs/            # Configuration files (YAML/JSON) for training, simulation, and robot
├── data/
│   ├── raw/            # Raw datasets collected from simulation or real robot
│   └── processed/      # Preprocessed and augmented datasets
├── docs/               # Project documentation and guides
├── models/             # Saved model checkpoints and weights
├── scripts/            # Utility scripts for data collection, training, and evaluation
├── src/
│   ├── networks/       # Neural network architectures for grasp prediction
│   ├── simulation/     # Simulation environment interface (e.g., Isaac Gym, MuJoCo)
│   ├── real_robot/     # Real robot interface and control
│   └── utils/          # Shared utility functions
└── tests/              # Unit and integration tests
```
