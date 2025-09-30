# Model_Collapse_Codes

This repository contains all code for the experiments in our study on model collapse, our paper is at \url{https://arxiv.org/abs/2502.18049} It includes both simulation studies and real-data applications. Repository link: [https://github.com/MukaiCodes/Model_Collapse_Codes](https://github.com/MukaiCodes/Model_Collapse_Codes.git)

## 1. Repository Structure


## In Simulation, each scenario has its own folder
- **Scenario1/**: Contains all codes for Scenario 1. Running the corresponding Python file (S1_Model_Collapse.py) will directly generate all necessary figures (Figure 7 (a)-(f)).
- **Scenario2/**: Contains all codes for Scenario 2. Running the corresponding Python file (S2_GoldenRatio.py) will directly generate all necessary figures (Figure 8 (a)-(f)).
- **Scenario3/**: Contains all codes for Scenario 3. Running the corresponding Python file (S3_Large_M.py) will directly generate all necessary figures ( Figure 9 (a)-(f)).
- **real_data/**: Contains scripts for real-data experiments
  - Real_Data_Experiment.py: Run this Python file will generate Figure 10 (d)
  - Real_Data_Experiment_Discrete.py: Run this Python file will generate Figure 10 (a)-(c)
- **requirements.txt**: Lists all Python dependencies.  
- **README.md**: This file.  

## 2. Installation

It is recommended to use a virtual environment to avoid conflicts:
```bash
# Clone the repository
git clone https://github.com/MukaiCodes/Model_Collapse_Codes.git
cd Model_Collapse_Codes

# Create a virtual environment
python -m venv env

# Activate the environment
# Windows
.\env\Scripts\activate
# macOS / Linux
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

## 3. Simulation

# Get Figure 7 (a)-(f)
cd Scenario_1
python S1_Model_Collapse.py

# Get Figure 8 (a)-(f)
cd Scenario_2
python S2_GoldenRatio.py

# Get Figure 9 (a)-(f)
cd Scenario_3
python S3_Large_M.py

cd Real_Data
python Real_Data_Experiment.py  # Get Figure 10 (d)
python Real_Data_Experiment_Discrete.py # Get Figure 10 (a)-(c)

