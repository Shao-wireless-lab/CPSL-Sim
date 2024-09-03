# Cooperative Sensing and Unmanned Aerial Vehicle Formation Control Optimization for Chemical Plume Source Localization (CPSL) Using Deep Reinforcement Learning

![Multi-Agent Cooperative CPSL](images/Ch5-CPSL-Behavior.mp4)

A Custom Gym Environment for Cooperative Sensing and Formation Control 

1. Create and activate virtual environment in anaconda
```bash
conda create --cuas_env
```

2. Install tensorflow and other packages in <cuas_env-package-list.txt> file: 
```bash
conda install tensorflow-gpu
```

3. To generate plume data (concentration and wind velocity measurements) --> see <example1_revised.m> file in Generate_Plume_Data_Files folder
   
4. To train model: 
```bash
python run_experimentv2.py train 
```
Note: see <parse_arguments()> function in <run_experimentv2.py> file for different training settings. 

Note: to store the training files in a specific directory, use: 
```bash
python --log_dir <name of file path> run_experimentv2.py train
```
Otherwise, it will save to the <results> folder. 

5. To test model:
```bash
python run_experimentv2.py test --checkpoint <checkpoint>
```
Note: use the most recent generated checkpoint file to test the model 

6. To visualize the training results:
   
-- Install and activate the jupyterlab environment 
   ```bash
   conda install conda-forge::jupyterlab
   ```

-- Install required packages from <jupyterenv-package-list.txt> file 

-- Open <Plume_Plot_Results.py> file 

-- To plot results, use: 
   ```bash 
   basedir = r'<training results folder path>'
   ``` 

