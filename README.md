# Model of the Mouse Visual-Motor System

This project implements a biologically-inspired model of the mouse visual-motor system for predator detection and evasion, developed as part of SYDE 552.

## Prerequisites

- **Unity 6.0.4.0f1** or later
- **Python 3.8+** with PyTorch
- Git

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Kriti1400/Syde552-Project.git
cd Syde552-Project
```

### 2. Open in Unity

1. Launch Unity Hub
2. Click "Open" → "Add project from disk"
3. Navigate to the cloned repository folder (`Syde552-Project`)
4. Select the folder and click "Open"

The project should load with all necessary Unity packages and dependencies.

### 3. Set Up Python Environment (for Vision Model Training)

If you want to train or modify the vision models:

```bash
# Create a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install torch torchvision numpy pandas matplotlib
```

### 4. Running the Simulation

1. In Unity, open the main scene from `Assets/Scenes/`
2. Press the Play button to start the simulation
3. The mouse agent will use the trained vision model to detect and evade predators

## Project Structure

- `Assets/`: Unity assets and scripts
  - `ML-Agents/`: Machine learning agents for training
  - `Scenes/`: Unity scenes
  - `results/`: Training results and configurations
- `kriti-vision/`: Python code for vision model training
- `python/`: Additional Python utilities

## Training the Vision Model

To train the mouse visual CNN:

```bash
cd kriti-vision
python train.py
```

This will train the model on synthetic data generated from a Python script.
