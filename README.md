# Cooperative Sensing and Unmanned Aerial Vehicle Formation Control Optimization for Chemical Plume Source Localization (CPSL) Using Deep Reinforcement Learning

### A Custom Gym Environment for Cooperative Sensing and Formation Control 

---
## Watch Our Demo
### Click [this link](https://www.youtube.com/watch?v=93PKTPt_GJI) or the image below to watch the CPSL demo on YouTube:
<p align="left">
  <a href="https://www.youtube.com/watch?v=93PKTPt_GJI">
    <img src="./results/plots/demo_screenshot.png" width="50%" alt="CPSL Demo Screenshot">
  </a>
</p>

---
## Data Generation
### To generate plume data, follow these steps:

### 1. Modify Parameters in the Plume Settings Script
Edit the file  
[`example_1_settings_revised.m`](./Generate_Plume_Data_Files/example_1_settings_revised.m)  
to configure the **noise level** and **emitter location**.

#### Change `a`, `b`, and `G` for different noise levels:
```matlab
% ff_params.abG = [0.005, 0.02, 1];
ff_params.abG = [0.005, 0.02, 3];
```
#### Change emitter location:
```matlab
% plume_params(1,1).p0    = [80 60 0];   % set initial position 3 dim
plume_params(1,1).p0    = [60 120 0];   % set initial position 3 dim
```

### 2. Run the Plume Generation Script

Execute [`example1_revised.m`](./Generate_Plume_Data_Files/example1_revised.m)  
to generate the plume data and store it in the MATLAB workspace.

### 3. Understand the Data Format

For each height level (e.g., **2m** or **5m**), the script generates a matrix of size:
```
600,000 × 4
```

Each **1,000 rows** represent one timestep. The columns are as follows:

- **Columns 1–3:** wind vector components (u, v, w)
- **Column 4:** concentration value at the grid point

This can be conceptually reshaped into a 4D tensor of shape:
```
6000 × 100 × 100 × 4
```

Where:
- **6000** timesteps  
- **100 × 100** spatial grid  
- **4** attributes per grid cell (3 for wind, 1 for concentration)

### 4. Save Data to Files

Run [`data_save.m`](./Generate_Plume_Data_Files/data_save.m) to export the generated data:

```matlab
% Define the folder to save files
save_folder = 'G=1_60_120';
```

Update the **save_folder path** to specify your desired local directory. 

The script will save:
- Wind data: 600,000 × 300 matrix
- Concentration data: 600,000 × 100 matrix

Both are saved in **.mat** format and ready for downstream use.

---
# MARL Workflow

Follow the steps below to set up the environment, generate data, train models, and visualize results for the MARL-based CPSL simulation.

---

### 1. Create and Activate a Virtual Environment

Install required packages using `pip`:

#### Using [`requirements.txt`](./requirements.txt) (pip):
```bash
python -m venv CPSL_env
source CPSL_nv/bin/activate
pip install -r requirements.txt
```


### 2.Generate Plume Data
Follow the instructions provided in the [Data Generation](#data-generation) section above.

### 3.Important Note
A certain error is very likely to occur due to the particular version of the Ray package used, 
please refer to the last part [Ray Problem Fix](#Ray-Problem-Fix) of this document to fix it.


### 4.Train the Model

There are two training scripts available ([`run_experiment_v1.py`](./run_experiment_v1.py) 
and [`run_experiment_v2.py`](./run_experiment_v2.py)). Both use the same training process.

- Simulation configuration: [`sim_config.cfg`](./configs/sim_config.cfg)

  Setting for training config and environment.

- Plume scenario configuration: [`plume_scenarios.json`](./configs/plume_scenarios.json)

  Setting for different scenarios.

You can view and modify training settings by inspecting the `parse_arguments()` function in any **run_experiment.py**.

#### Example training command:

```bash
python run_experiment_v1.py train --duration 15000
```

#### To specify a custom directory for saving training results:
```bash
python run_experiment_v1.py train --log_dir <path_to_output_directory>
```

If `--log_dir` is not specified, results will be saved to the default `results/` folder.

###  5.Test the Model
There are two testing scripts depending on your evaluation needs:

To specify a custom directory for saving training results:

- [`run_experiment_v1.py`](./run_experiment_v1.py)

    Saves **per-timestep** episode data in `.json` format for detailed evaluation and visualization.

- [`run_experiment_v2.py`](./run_experiment_v2.py)

    Computes **summary statistics** across a specified number of test episodes.

#### Example test command:

```bash
python run_experiment_v1.py test --checkpoint <path_to_checkpoint> --max_num_episodes 10
```

#### Full example:
```bash
python run_experiment_v1.py test --checkpoint /home/ece213/CPSL-Sim/results/ppo/CPSL_2025-04-10-02-30/debug/MyTrainer_CPSL_0v0o5_095d4_00000_0_observation_type=local,custom_model=DeepsetModel_2025-04-10_02-30-19/checkpoint_000250/checkpoint-250 --max_num_episodes 100

```
> Make sure to use the most recent checkpoint file when testing.

#### Important:
In all `run_experiment` scripts, the environment is initialized using the `env_config` stored during training.
To test with a different setup, you must manually override this configuration (see `lines 349–362`) in the script.


### 6.Visualize Training and Testing Results
#### For training progress visualization:
Run [`plot_training.py`](plot/plot_paper/plot_training.py)
#### For testing/demo visualization:
Run [`plot_demo.py`](./plot/plot_demo.py)

These scripts will generate plots for performance tracking, qualitative analysis or animation of selected test episode.

## Notes

- A pre-trained checkpoint is available and ready to use at: `results/ppo/CPSL_2025-04-10-02-30/`.
- Due to data size, plume data and testing results are not included. Please refer to [Data Generation](#data-generation) and [Test the model](#5test-the-model).
- Some scripts provide an option to either **`save`** or **`show`** the output. Be sure to choose the appropriate mode depending on whether you want to visualize the result or save it to a file.
- **Important:** Make sure to update any file paths in the **`plot scripts`** and **`plume_scenarios.json`**  to match the actual paths on your system.

--- 
# Ray Problem Fix
Need to fix the ray file refer to the following solution.

Issue with ray 1.13 and PyTorch 1.13: 
https://github.com/ray-project/ray/issues/26781
https://github.com/ray-project/ray/issues/26557

change line 765 in torch_policy.py located at: `<your venv path>/envs/py3915/lib/python3.9/site-packages/ray/rllib/policy/torch_policy.py`


```python 
    def set_state(self, state: dict) -> None:
        # Set optimizer vars first.
        optimizer_vars = state.get("_optimizer_variables", None)
        if optimizer_vars:
            assert len(optimizer_vars) == len(self._optimizers)
            for o, s in zip(self._optimizers, optimizer_vars):
#start fix
                for v in s["param_groups"]:
                    if "foreach" in v.keys():
                        v["foreach"] = False if v["foreach"] is None else v["foreach"]
                for v in s["state"].values():
                    if "momentum_buffer" in v.keys():
                        v["momentum_buffer"] = False if v["momentum_buffer"] is None else v["momentum_buffer"]
#end fix
                optim_state_dict = convert_to_torch_tensor(s, device=self.device)
                o.load_state_dict(optim_state_dict)
        # Set exploration's state.
        if hasattr(self, "exploration") and "_exploration_state" in state:
            self.exploration.set_state(state=state["_exploration_state"])
        # Then the Policy's (NN) weights.
        super().set_state(state)
```
# Fluxotaxis implementation
To test the effectiveness of the CPSL DRL algorithm, a three agent version of the Fluxotaxis algorithm was implemented. The algorithm is based on the Large Swarm Implementation version (see Section 9.4.5 in Physicomimetics book by Spears). The algorithm is broken down into two major steps: computation of the formation control vector, and the computation of the Fluxotaxis control vector. In an attempt to enforce the priority of the swarm formation, a gain of 10 was put on the formation unit control vector and a gain of 1 was used with the Fluxotaxis unit control vector. 

## Formation control vector
To control the formation of the agents and provide a way for seamless obstacle avoidance, a generalized version of the Lennard-Jones potential is used to generate a formation control vector,
$$F_{formation} = {n_F} = {c_1\frac{F(D)^{c_3}}{r^{c_5}} - c_2\frac{F(D)^{c_4}}{r^{c_6}}}$$

$$\mathcal{F}(D)=(\frac{c_1D^{c_6-c_5}}{c_2})^{\frac{1}{c_4-c_3}},$$
where hyperparameters are chosen for simplicity to be the same as those found in Spears et. al. ($c_1=c_2=c_3=1$, $c_4=c_5=2$, and $c_6=3$), such that $F_{formation} = \mathbf{n}_F = \mathcal{F}(D)/r^2 - \mathcal{F}(D)^2/r^3$ and $\mathcal{F}(D)=D^{1/2}$. The variable $D$ represents the desired separation distance between the agents, which was set to 5m.

## Fluxotaxis control vector
The decentralized scheme involves calculating the $i$-th agent's relative flux between the $j$-th neighboring agents given by Spears et. al., that is, 
$$GDMF_i \approx F_{ij} = \rho_j |V_j|\cos{\theta_r}=\rho_j\frac{r_{ij}}{|r_{ij}|} \cdot V_j.$$
The angle between $V_j$ and separation vector, $r_{ij}$, is given by $\theta_r$. The resulting control vector for the Fluxotaxis algorithm is then given as $F^n_{ij}$, which depends on whether the set of GDMF vectors contain a negative value or positive value. If the value is negative, the control vector is set to be the arg min of the set. If the there are no negative GDMF values, then the control vector is set to be the arg max, where $v_i$ is the velocity vector control of the $i$-th agent, and $n_{ij}=r_{ij}/|r_{ij}|$ points in the direction of $F^n_{ij}$. If no agents are around or no agent detects a concentration, $n_{ij} \rightarrow 0$. 

The update rule for the fluxotaxis algorithm becomes, 
$$\frac{dv_i}{dt} = n_F+n_{ij},$$

When no flux is detected, other strategies can be deferred to. The secondary source seeking strategies we consider are: chemotaxis, anemotaxis, and casting. In this example, chemotaxis is chosen to help keep the agents near the plume.
## References
- Spears, William M., and Diana F. Spears, eds. Physicomimetics: Physics-based swarm intelligence. Springer Science & Business Media, 2012.
- Hollenbeck, Derek, et al. "Swarm Robotic Source Seeking with Fractional Fluxotaxis." 2023 International Conference on Fractional Differentiation and Its Applications (ICFDA). IEEE, 2023.
