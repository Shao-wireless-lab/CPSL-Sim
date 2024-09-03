import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
from cuas.models.base_model import BaseModel


class DeepsetModel(BaseModel):
    def __init__(
        self, obs_space, act_space, num_outputs, model_config, name, *args, **kwargs
    ):
        BaseModel.__init__(
            self, obs_space, act_space, num_outputs, model_config, name, *args, **kwargs
        )
        self.pooling_type = model_config["custom_model_config"]["pooling_type"]

        if self.pooling_type == "sum":
            self.pooling_func = torch.sum
        elif self.pooling_type == "mean":
            self.pooling_func = torch.mean
        elif self.pooling_type == "max":
            self.pooling_func = torch.amax

        # size of tensor [batch, input, output]
        hidden_layer_size = 256
        #self.phi_evader = nn.Sequential(
        #    nn.Linear(self.num_evader_other_agent_states, hidden_layer_size),
        #    nn.ReLU(),
        #    nn.Linear(hidden_layer_size, hidden_layer_size),
        #)
        #self.rho_evader = nn.Sequential(
        #    nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()
        #)
        #######################
        # States relative to other agents 
        self.phi_agents = nn.Sequential(
            nn.Linear(self.num_other_agent_states, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
        )
        self.rho_agents = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()
        )
        #######################
        # States relative to other agent measurements 
        self.phi_agents_meas = nn.Sequential(
            nn.Linear(self.num_other_agent_meas_states, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
        )
        self.rho_agents_meas = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()
        )
        #######################
        # States relative to location with max plume reading 
        self.phi_agents_max_plume = nn.Sequential(
            nn.Linear(self.num_sensed_plume_states, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
        )
        self.rho_agents_max_plume = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()
        )
        #######################
        # States relative to obstacles 
        self.phi_obs = nn.Sequential(
            nn.Linear(self.num_obstacle_states, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
        )
        self.rho_obs = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()
        )

        # concatenate the agent, evader, other_agents and obstacles
        self.last_state = nn.Sequential(
            nn.Linear(
                self.num_agent_states
                + self.num_agent_meas_states
                + self.num_agent_form_states
                + hidden_layer_size
                + hidden_layer_size
                + hidden_layer_size
                + hidden_layer_size,
                hidden_layer_size,
            ),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
        )
        self.policy_fn = nn.Linear(hidden_layer_size, num_outputs)
        self.value_fn = nn.Linear(hidden_layer_size, 1)

    def forward(self, input_dict, state, seq_lens):
        #####################################################################
        # Agents states 
        index = 0
        end_index = self.num_agent_states
        main_agent = input_dict["obs"]["observations"][:, index:end_index]

        #######################
        # Agent measurement states
        index = end_index
        end_index += self.num_agent_meas_states
        main_agent_meas = input_dict["obs"]["observations"][:, index:end_index]
        
        #######################
        # Agent states relative to formation 
        index = end_index
        end_index += self.num_agent_form_states
        main_agent_form_meas = input_dict["obs"]["observations"][:, index:end_index]

        #######################
        # Other agents states
        index = end_index
        end_index += self.num_other_agent_states * (self.max_num_agents - 1)
        other_agents = input_dict["obs"]["observations"][:, index:end_index]
        other_agents = torch.reshape(
            other_agents,
            (-1, self.max_num_agents - 1, self.num_other_agent_states),
        )
        
        # other agent weights
        # only get active agents in environment
        other_agents = other_agents[:, : self.num_agents - 1, :]
        
        #######################
        # Other agents measurement states
        index = end_index
        end_index += self.num_other_agent_meas_states * (self.max_num_agents - 1)
        other_agents_meas = input_dict["obs"]["observations"][:, index:end_index]
        other_agents_meas = torch.reshape(
            other_agents_meas,
            (-1, self.max_num_agents - 1, self.num_other_agent_meas_states),
        )
        
        # other agent meas weights
        # only get active agents in environment
        other_agents_meas = other_agents_meas[:, : self.num_agents - 1, :]

        #######################
        # States of location with max plume reading 
        index = end_index
        end_index += self.num_sensed_plume_states 
        main_agent_meas_to_plume_reading = input_dict["obs"]["observations"][:, index:end_index]
        main_agent_meas_to_plume_reading = torch.reshape(
            main_agent_meas_to_plume_reading,
            (-1, 1, self.num_sensed_plume_states),
        )
      
        # location meas weights
        # only get active relative measurements in environment
        main_agent_meas_to_plume_reading = main_agent_meas_to_plume_reading[:, : 1, :]


        #######################
        # obstacle states
        index = end_index
        end_index += self.num_obstacle_states * self.max_num_obstacles
        obstacles = input_dict["obs"]["observations"][:, index:end_index]

        obstacles = torch.reshape(
            obstacles, (-1, self.max_num_obstacles, self.num_obstacle_states)
        )

        # obstacle weights
        # only get active obstacles in environment, just in case there's no obstacles, add a dummy obstacle
        self.num_obstacles = 1 if self.num_obstacles == 0 else self.num_obstacles
        obstacles = obstacles[:, : self.num_obstacles, :]

        ############################################################################################
        # other agent weights deepset weights
        x_agents = self.phi_agents(other_agents)
        x_agents = self.pooling_func(x_agents, dim=1)
        x_agents = self.rho_agents(x_agents)
        
         # other agent weights deepset weights
        x_agents_meas = self.phi_agents_meas(other_agents_meas)
        x_agents_meas = self.pooling_func(x_agents_meas, dim=1)
        x_agents_meas = self.rho_agents_meas(x_agents_meas)

        # other agent weights deepset weights
        x_agents_meas_max_plume = self.phi_agents_max_plume(main_agent_meas_to_plume_reading)
        x_agents_meas_max_plume = self.pooling_func(x_agents_meas_max_plume, dim=1)
        x_agents_meas_max_plume = self.rho_agents_max_plume(x_agents_meas_max_plume)

        # obstacles deepset weights
        x_obs = self.phi_obs(obstacles)
        x_obs = self.pooling_func(x_obs, dim=1)
        x_obs = self.rho_obs(x_obs)

        #print(len(main_agent))
        #print(len(main_agent_meas))
        #print(len(main_agent_form_meas))
        #print(len(x_agents))
        #print(len(x_agents_meas))
        #print(len(x_obs))
        #quit()



        x = torch.cat((main_agent, 
                       main_agent_meas, 
                       main_agent_form_meas, 
                       x_agents, 
                       x_agents_meas, 
                       x_agents_meas_max_plume, 
                       x_obs
                       ), dim=1)
        x = self.last_state(x)

        # Save for value function
        self._value_out = self.value_fn(x)

        logits = self.policy_fn(x)
        logits = self.proj_safe_actions(input_dict, logits)
        return logits, state

    def value_function(self):
        return self._value_out.flatten()
