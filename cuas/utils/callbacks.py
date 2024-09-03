from typing import Dict, Tuple

import numpy as np
from gym.spaces import Box
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


class TrainCallback(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        # print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))

        episode.user_data["target_found"] = []
        episode.user_data["final_cent_dist_to_emitter"] = []
        episode.user_data["time_tfinal"] = []

        episode.user_data["current_cent_dist_to_emitter"] = []

        episode.user_data["drop_found"] = []
        episode.user_data["time_tdrop"] = []
        episode.user_data["at_drop_cent_dist_to_emitter"] = []

        episode.user_data["avg_agent_to_cent_dist"] = []

        episode.user_data["circ_boun_dist2emitter"] = []

        episode.user_data["dist_closestUAV2emitter"] = []

        episode.user_data["avg_UAV2emitter"] = []
        
        episode.user_data["R_task"] = []
        episode.user_data["R_plume"] = []
        episode.user_data["R_upwind"] = []
        episode.user_data["R_col"] = []
        episode.user_data["R_d"] = []
        episode.user_data["R_theta"] = []

        episode.user_data["R_total"] = []
                        
        episode.user_data["agent_collisions"] = []
        episode.user_data["obstacle_collisions"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        agent_ids = episode.get_agents()
        
        target_found = []
        final_cent_dist_to_emitter = []
        time_tfinal = []

        current_cent_dist_to_emitter = []

        drop_found = []
        time_tdrop = []
        at_drop_cent_dist_to_emitter = []
        
        avg_agent_to_cent_dist = []

        circ_boun_dist2emitter = []

        dist_closestUAV2emitter = []

        avg_UAV2emitter = []
        
        cum_R_task = 0
        cum_R_upwind = 0
        cum_R_plume = 0
        cum_R_col = 0
        cum_R_d = 0
        cum_R_theta = 0

        cum_R_total = 0

        cum_agent_collisions = 0
        cum_obstacle_collisions = 0

        for agent_id in agent_ids:
            last_info = episode.last_info_for(agent_id)

            target_found.append(last_info["target_found"])
            final_cent_dist_to_emitter.append(last_info["final_cent_dist_to_emitter"])
            time_tfinal.append(last_info["time_tfinal"])

            current_cent_dist_to_emitter.append(last_info["current_cent_dist_to_emitter"])

            drop_found.append(last_info["drop_found"])
            time_tdrop.append(last_info["time_tdrop"])
            at_drop_cent_dist_to_emitter.append(last_info["at_drop_cent_dist_to_emitter"])
            
            avg_agent_to_cent_dist.append(last_info["avg_agent_to_cent_dist"])

            circ_boun_dist2emitter.append(last_info["circ_boun_dist2emitter"])

            dist_closestUAV2emitter.append(last_info["dist_closestUAV2emitter"])

            avg_UAV2emitter.append(last_info["avg_UAV2emitter"])

            cum_R_task += last_info["R_task"]
            cum_R_plume += last_info["R_plume"]
            cum_R_upwind += last_info["R_upwind"]
            cum_R_col += last_info["R_col"]
            cum_R_d += last_info["R_d"]
            cum_R_theta += last_info["R_theta"]

            cum_R_total += last_info["R_total"]

            cum_agent_collisions += last_info["agent_collision"]
            cum_obstacle_collisions += last_info["obstacle_collision"]

        target_found = max(target_found)
        final_cent_dist_to_emitter = max(final_cent_dist_to_emitter)
        time_tfinal = max(time_tfinal)

        current_cent_dist_to_emitter = max(current_cent_dist_to_emitter)

        drop_found = max(drop_found)
        time_tdrop = max(time_tdrop)
        at_drop_cent_dist_to_emitter = max(at_drop_cent_dist_to_emitter)

        avg_agent_to_cent_dist = max(avg_agent_to_cent_dist)

        circ_boun_dist2emitter = max(circ_boun_dist2emitter)

        dist_closestUAV2emitter = max(dist_closestUAV2emitter)

        avg_UAV2emitter = max(avg_UAV2emitter)

        episode.user_data["target_found"].append(target_found)
        episode.user_data["final_cent_dist_to_emitter"].append(final_cent_dist_to_emitter)
        episode.user_data["time_tfinal"].append(time_tfinal)

        episode.user_data["current_cent_dist_to_emitter"].append(current_cent_dist_to_emitter)

        episode.user_data["drop_found"].append(drop_found)
        episode.user_data["time_tdrop"].append(time_tdrop)
        episode.user_data["at_drop_cent_dist_to_emitter"].append(at_drop_cent_dist_to_emitter)

        episode.user_data["avg_agent_to_cent_dist"].append(avg_agent_to_cent_dist)

        episode.user_data["circ_boun_dist2emitter"].append(circ_boun_dist2emitter)

        episode.user_data["dist_closestUAV2emitter"].append(dist_closestUAV2emitter)

        episode.user_data["avg_UAV2emitter"].append(avg_UAV2emitter)

        episode.user_data["R_task"].append(cum_R_task)
        episode.user_data["R_plume"].append(cum_R_plume)
        episode.user_data["R_upwind"].append(cum_R_upwind)
        episode.user_data["R_col"].append(cum_R_col)
        episode.user_data["R_d"].append(cum_R_d)
        episode.user_data["R_theta"].append(cum_R_theta)

        episode.user_data["R_total"].append(cum_R_total)

        episode.user_data["agent_collisions"].append(cum_agent_collisions)
        episode.user_data["obstacle_collisions"].append(cum_obstacle_collisions)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.policy_config["batch_mode"] == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )
        # pole_angle = np.mean(episode.user_data["pole_angles"])
        # print(
        #     "episode {} (env-idx={}) ended with length {} and pole "
        #     "angles {}".format(
        #         episode.episode_id, env_index, episode.length, pole_angle
        #     )
        # )
        # episode.custom_metrics["pole_angle"] = pole_angle
        # episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]

        target_found = np.max(episode.user_data["target_found"])
        episode.custom_metrics["target_found"] = target_found
        final_cent_dist_to_emitter = np.max(episode.user_data["final_cent_dist_to_emitter"])
        episode.custom_metrics["final_cent_dist_to_emitter"] = final_cent_dist_to_emitter
        time_tfinal = np.max(episode.user_data["time_tfinal"])
        episode.custom_metrics["time_tfinal"] = time_tfinal

        current_cent_dist_to_emitter = np.max(episode.user_data["current_cent_dist_to_emitter"])
        episode.custom_metrics["current_cent_dist_to_emitter"] = current_cent_dist_to_emitter

        drop_found = np.max(episode.user_data["drop_found"])
        episode.custom_metrics["drop_found"] = drop_found
        time_tdrop = np.max(episode.user_data["time_tdrop"])
        episode.custom_metrics["time_tdrop"] = time_tdrop 
        at_drop_cent_dist_to_emitter = np.max(episode.user_data["at_drop_cent_dist_to_emitter"])
        episode.custom_metrics["at_drop_cent_dist_to_emitter"] = at_drop_cent_dist_to_emitter

        avg_agent_to_cent_dist = np.max(episode.user_data["avg_agent_to_cent_dist"])
        episode.custom_metrics["avg_agent_to_cent_dist"] = avg_agent_to_cent_dist

        circ_boun_dist2emitter = np.max(episode.user_data["circ_boun_dist2emitter"])
        episode.custom_metrics["circ_boun_dist2emitter"] = circ_boun_dist2emitter

        dist_closestUAV2emitter = np.max(episode.user_data["dist_closestUAV2emitter"])
        episode.custom_metrics["dist_closestUAV2emitter"] = dist_closestUAV2emitter

        avg_UAV2emitter = np.max(episode.user_data["avg_UAV2emitter"])
        episode.custom_metrics["avg_UAV2emitter"] = avg_UAV2emitter

        R_task = np.sum(episode.user_data["R_task"])
        episode.custom_metrics["R_task"] = R_task
        R_plume = np.sum(episode.user_data["R_plume"])
        episode.custom_metrics["R_plume"] = R_plume
        R_upwind = np.sum(episode.user_data["R_upwind"])
        episode.custom_metrics["R_upwind"] = R_upwind
        R_col = np.sum(episode.user_data["R_col"])
        episode.custom_metrics["R_col"] = R_col
        R_d = np.sum(episode.user_data["R_d"])
        episode.custom_metrics["R_d"] = R_d
        R_theta = np.sum(episode.user_data["R_theta"])
        episode.custom_metrics["R_theta"] = R_theta

        R_total = np.sum(episode.user_data["R_total"])
        episode.custom_metrics["R_total"] = R_total

        agent_collisions = np.sum(episode.user_data["agent_collisions"])
        episode.custom_metrics["agent_collisions"] = agent_collisions
        obstacle_collisions = np.sum(episode.user_data["obstacle_collisions"])
        episode.custom_metrics["obstacle_collisions"] = obstacle_collisions



    # def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
    #     print("returned sample batch of size {}".format(samples.count))

    # def on_train_result(self, *, trainer, result: dict, **kwargs):
    #     print(
    #         "trainer.train() result: {} -> {} episodes".format(
    #             trainer, result["episodes_this_iter"]
    #         )
    #     )
    #     # you can mutate the result dict to add new fields to return
    #     result["callback_ok"] = True

    # def on_learn_on_batch(
    #     self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    # ):
    #     pass
    #     # result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"])
    #     # print(
    #     #     "policy.learn_on_batch() result: {} -> sum actions: {}".format(
    #     #         policy, result["sum_actions_in_train_batch"]
    #     #     )
    #     # )

    # def on_postprocess_trajectory(
    #     self,
    #     *,
    #     worker: RolloutWorker,
    #     episode: Episode,
    #     agent_id: str,
    #     policy_id: str,
    #     policies: Dict[str, Policy],
    #     postprocessed_batch: SampleBatch,
    #     original_batches: Dict[str, Tuple[Policy, SampleBatch]],
    #     **kwargs
    # ):
    #     print("postprocessed {} steps".format(postprocessed_batch.count))
    #     if "num_batches" not in episode.custom_metrics:
    #         episode.custom_metrics["num_batches"] = 0
    #     episode.custom_metrics["num_batches"] += 1


class FillInActions(DefaultCallbacks):
    """Fills in the opponent actions info in the training batches.
    Keep track of the single agent action space. It is declared here but should be changed if declared elsewhere in the env.
    """

    def on_postprocess_trajectory(
        self,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs
    ):
        action_space_shape = 2
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        action_encoder = ModelCatalog.get_preprocessor_for_space(
            Box(low=-1, high=1, dtype=np.float32, shape=(action_space_shape,))
        )

        # get all the actions, clip the actions just in case
        all_actions = []
        num_agents = len(original_batches)
        for agent_id in range(num_agents):
            _, single_agent_batch = original_batches[agent_id]
            single_agent_action = np.array(
                [
                    action_encoder.transform(np.clip(a, -1, 1))
                    # action_encoder.transform(a)
                    for a in single_agent_batch[SampleBatch.ACTIONS]
                ]
            )

            all_actions.append(single_agent_action)


        all_actions = np.array(all_actions)
        num_agent_actions = num_agents * action_space_shape
        all_actions = all_actions.reshape(-1, num_agent_actions)
        # print(f"action shape: {all_actions.shape}")
        # print(f"to_update shape: {to_update.shape}")
        to_update[:, -num_agent_actions:] = all_actions
