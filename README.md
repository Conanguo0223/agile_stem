# 📋 Project Overview
This simulation models a water bottle manufacturing process using SimPy, including:
1. Blow Molding
2. Cleaning
3. Filling
4. Capping & Labeling
5. Packaging

The simulation supports:
✅ Deterministic and stochastic demand
✅ Deterministic and stochastic processing times
✅ Single-run and multiple-run/replication simulations for performance analysis.
## 🚀 Project Structure
/project-folder/
├── config.py                          # Configuration file for simulation parameters
├── process.py                         # Manufacturing process class definition
├── manufacturing_simulation.py         # Main simulation file (single run)
├── manufacturing_simulation_replication.py  # Simulation with multiple replications
└── README.docx                         # This guide (Word format)
## 🛠 Installation & Setup
1️⃣ Install Required Libraries
Ensure you have Python 3.7+ installed. Run the following command:

pip install simpy numpy

2️⃣ Configure Simulation Parameters (`config.py`)
🎬 Running the Simulation
✅ Single Run Simulation:
Run the single-run simulation using:

python manufacturing_simulation.py

✅ Multiple Runs Simulation (Replications):
Run the multiple-run simulation using:

python manufacturing_simulation_replication.py

📈 Understanding the Output
Single Run Output:
- Total Bottles Produced
- Throughput (bottles/hour)
- Average and Maximum Queue Length
- Machine Utilization per stage

Multiple Runs Output (Replications):
- Mean and Standard Deviation for Total Bottles Produced, Throughput, Queue Lengths, and Machine Utilization.
🔄 Customizing the Simulation
1️⃣ Change Demand Characteristics:
- Deterministic: DEMAND_RATE = 5
- Stochastic: DEMAND_RATE = lambda: random.expovariate(1/5)

2️⃣ Modify Processing Times:
- Fixed times: PROCESS_TIMES = [5, 3, 2, 4, 5]
- Stochastic times: Adjust lambda functions in config.py.

3️⃣ Adjust Number of Replications:
NUM_REPLICATIONS = 10  # Modify this number for more or fewer runs.
💡 Notes & Best Practices
- Set a random seed for reproducibility: random.seed(42)
- Experiment by adjusting SIM_TIME, MACHINE_CAPACITIES, and CONVEYOR_CAPACITIES.
💬 Troubleshooting
1. ModuleNotFoundError:
   - Ensure Python is in PATH.
   - Run pip install simpy numpy.

2. Stochastic Results Vary Widely:
   - Increase the number of replications for stable averages.

3. Performance Issues:
   - Reduce SIM_TIME or simplify stochastic functions.
🙌 Contributors
Simulation Design: Omar Ashour
Documentation: ChatGPT
📜 License
This project is licensed under the MIT License.
Feel free to use, modify, and share.
