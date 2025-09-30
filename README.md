# Model_Collapse_Codes

This repository contains all code for the experiments in our study on model collapse. It includes both simulation studies and real-data applications.  

Repository link: [https://github.com/MukaiCodes/Model_Collapse_Codes](https://github.com/MukaiCodes/Model_Collapse_Codes.git)

## 1. Repository Structure

Model_Collapse_Codes/
│
├─ simulations/
│ ├─ Scenario_1/
│ │ └─ run S1_Model_Collapse.py
│ ├─ Scenario_2/
│ │ └─ run S2_GoldenRatio.py
│ └─ Scenario_3
│ │ └─ run S3_Large_M.py
├─ Real_Data/
│ └─ run Real_Data_Experiment.py
│ └─ run Real_Data_Experiment_Discrete.py
├─ requirements.txt
└─ README.md


- **simulations/**: Contains all simulation scenarios. Each scenario has its own folder. Running the corresponding Python file will directly generate all necessary figures.  
- **real_data/**: Contains scripts for real-data experiments.  
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

