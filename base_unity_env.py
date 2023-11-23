from gymnasium.spaces import Box, MultiDiscrete, Discrete, Tuple as TupleSpace
import logging
import numpy as np
import random
import time
from typing import Callable, Optional, Tuple

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import MultiAgentDict, PolicyID, AgentID

logger = logging.getLogger(__name__)


@PublicAPI
class Unity3DEnv(MultiAgentEnv):
    # Default base port when connecting directly to the Editor
    _BASE_PORT_EDITOR = 5004
    # Default base port when connecting to a compiled environment
    _BASE_PORT_ENVIRONMENT = 5005
    # The worker_id for each environment instance
    _WORKER_ID = 0

    def __init__(
        self,
        file_name: str = None,
        port: Optional[int] = None,
        seed: int = 0,
        no_graphics: bool = False,
        timeout_wait: int = 300,
        episode_horizon: int = 1000,
    ):

        # Skip env checking as the nature of the agent IDs depends on the game
        # running in the connected Unity editor.
        self._skip_env_checking = True

        super().__init__()

        if file_name is None:
            print(
                "No game binary provided, will use a running Unity editor "
                "instead.\nMake sure you are pressing the Play (|>) button in "
                "your editor to start."
            )

        import mlagents_envs
        from mlagents_envs.environment import UnityEnvironment

        # Try connecting to the Unity3D game instance. If a port is blocked
        port_ = None
        while True:
            # Sleep for random time to allow for concurrent startup of many
            # environments (num_workers >> 1). Otherwise, would lead to port
            # conflicts sometimes.
            if port_ is not None:
                time.sleep(random.randint(1, 10))
            port_ = port or (
                self._BASE_PORT_ENVIRONMENT if file_name else self._BASE_PORT_EDITOR
            )
            # cache the worker_id and
            # increase it for the next environment
            worker_id_ = Unity3DEnv._WORKER_ID if file_name else 0
            Unity3DEnv._WORKER_ID += 1
            try:
                self.unity_env = UnityEnvironment(
                    file_name=file_name,
                    worker_id=worker_id_,
                    base_port=port_,
                    seed=seed,
                    no_graphics=no_graphics,
                    timeout_wait=timeout_wait,
                )
                print("Created UnityEnvironment for port {}".format(port_ + worker_id_))
            except mlagents_envs.exception.UnityWorkerInUseException:
                pass
            else:
                break

        # ML-Agents API version.
        self.api_version = self.unity_env.API_VERSION.split(".")
        self.api_version = [int(s) for s in self.api_version]

        # Reset entire env every this number of step calls.
        self.episode_horizon = episode_horizon
        # Keep track of how many times we have called `step` so far.
        self.episode_timesteps = 0

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """Performs one multi-agent step through the game.

        Args:
            action_dict: Multi-agent action dict with:
                keys=agent identifier consisting of
                [MLagents behavior name, e.g. "Goalie?team=1"] + "_" +
                [Agent index, a unique MLAgent-assigned index per single agent]

        Returns:
            tuple:
                - obs: Multi-agent observation dict.
                    Only those observations for which to get new actions are
                    returned.
                - rewards: Rewards dict matching `obs`.
                - dones: Done dict with only an __all__ multi-agent entry in
                    it. __all__=True, if episode is done for all agents.
                    ......0
                - infos: An (empty) info dict.
        """
        from mlagents_envs.base_env import ActionTuple

        # Set only the required actions (from the DecisionSteps) in Unity3D.
        all_agents = []
        for behavior_name in self.unity_env.behavior_specs:
            # New ML-Agents API: Set all agents actions at the same time
            # via an ActionTuple. Since API v1.4.0.
            if self.api_version[0] > 1 or (
                self.api_version[0] == 1 and self.api_version[1] >= 4
            ):
                actions = []
                for agent_id in self.unity_env.get_steps(behavior_name)[0].agent_id:

                    key = behavior_name + "_{}".format(agent_id)
                    all_agents.append(key)
                    actions.append(action_dict[key])




                if actions:
                    if isinstance(actions[0],tuple):

                        action=actions[0]
                        if action[0].dtype == np.float32:
                            action_tuple = ActionTuple(continuous=np.array(actions))
                        else:
                            action_tuple = ActionTuple(discrete=np.array(actions))

                    else:
                        if actions[0].dtype == np.float32:
                            action_tuple = ActionTuple(continuous=np.array(actions))
                        else:
                            action_tuple = ActionTuple(discrete=np.array(actions))


                    self.unity_env.set_actions(behavior_name, action_tuple)
            # Old behavior: Do not use an ActionTuple and set each agent's
            # action individually.
            else:
                for agent_id in self.unity_env.get_steps(behavior_name)[
                    0
                ].agent_id_to_index.keys():
                    key = behavior_name + "_{}".format(agent_id)
                    all_agents.append(key)
                    self.unity_env.set_action_for_agent(
                        behavior_name, agent_id, action_dict[key]
                    )
        # Do the step.
        self.unity_env.step()

        obs, rewards, terminateds, truncateds, infos = self._get_step_results()

        # Global horizon reached? -> Return __all__ truncated=True, so user
        # can reset. Set all agents' individual `truncated` to True as well.
        self.episode_timesteps += 1
        if self.episode_timesteps > self.episode_horizon:
            return (
                obs,
                rewards,
                terminateds,
                dict({"__all__": True}, **{agent_id: True for agent_id in all_agents}),
                infos,
            )

        return obs, rewards, terminateds, truncateds, infos

    def reset(
        self, *, seed=None, options=None
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """Resets the entire Unity3D scene (a single multi-agent episode)."""
        self.episode_timesteps = 0
        self.unity_env.reset()
        obs, _, _, _, infos = self._get_step_results()
        return obs, infos

    def _get_step_results(self):
        """Collects those agents' obs/rewards that have to act in next `step`.

        Returns:
            Tuple:
                obs: Multi-agent observation dict.
                    Only those observations for which to get new actions are
                    returned.
                rewards: Rewards dict matching `obs`.
                dones: Done dict with only an __all__ multi-agent entry in it.
                    __all__=True, if episode is done for all agents.
                infos: An (empty) info dict.
        """
        obs = {}
        rewards = {}
        infos = {}

        for behavior_name in self.unity_env.behavior_specs:
            decision_steps, terminal_steps = self.unity_env.get_steps(behavior_name)
            # Important: Only update those sub-envs that are currently
            # available within _env_state.
            # Loop through all envs ("agents") and fill in, whatever
            # information we have.
            # for agent_id,idx in decision_steps.agent_id_to_index.items():
            #     print("agent_id",agent_id)
            #     print("idx",idx)
            #
            for agent_id, idx in decision_steps.agent_id_to_index.items():
                key = behavior_name + "_{}".format(agent_id)
                os = tuple(o[idx] for o in decision_steps.obs)
                os = os[0] if len(os) == 1 else os
                obs[key] = os
                rewards[key] = (
                    decision_steps.reward[idx] + decision_steps.group_reward[idx]
                )
            for agent_id, idx in terminal_steps.agent_id_to_index.items():
                key = behavior_name + "_{}".format(agent_id)
                # Only overwrite rewards (last reward in episode), b/c obs
                # here is the last obs (which doesn't matter anyways).
                # Unless key does not exist in obs.
                if key not in obs:
                    os = tuple(o[idx] for o in terminal_steps.obs)
                    obs[key] = os = os[0] if len(os) == 1 else os
                rewards[key] = (
                    terminal_steps.reward[idx] + terminal_steps.group_reward[idx]
                )

        # Only use dones if all agents are done, then we should do a reset.
        return obs, rewards, {"__all__": False}, {"__all__": False}, infos

    @staticmethod
    def get_policy_configs_for_game(
        game_name: str,
    ) -> Tuple[dict, Callable[[AgentID], PolicyID]]:

        # The RLlib server must know about the Spaces that the Client will be
        # using inside Unity3D, up-front.
        obs_spaces = {
            #Large_obs
            "Large_Obs": TupleSpace(
                [
                    Box(float("-inf"), float("inf"), (3, 6)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (36,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (11,))

                ]

            ),

            ##Large_WPM_Obs

            "Large_WPM_Obs": TupleSpace(
                [
                    Box(float("-inf"), float("inf"), (3, 6)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (36,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (11,))

                ]

            ),

            "Large_WPM_Obs_Dense": TupleSpace(
                [
                    Box(-1, 1, (3, 6)), #possible wrong dimension(3,6)
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (36,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(-10, 10, (11,))

                ]

            ),


            


            "XL_WPM_Obs": TupleSpace(
                [
                    Box(float("-inf"), float("inf"), (3, 6)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (36,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (11,))

                ]

            ),

            
            "XL_WPM_Obs_Dense": TupleSpace(
                [
                    Box(float("-inf"), float("inf"), (3, 6)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (738,)),
                    Box(float("-inf"), float("inf"), (36,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (378,)),
                    Box(float("-inf"), float("inf"), (11,))

                ]

            ),

        }
        action_spaces = {
            #Large_Obs

            # "Large_Obs":TupleSpace(
            #     [
            #         Box(-1.0,1.0,(3,),dtype=np.float32),
            #         MultiDiscrete([2,])
            #
            #     ]
            # ),


            "Large_Obs":  Box(-1.0,1.0,(3,),dtype=np.float32),

            # "Large_WPM_Obs": MultiDiscrete([9,]),
            "Large_WPM_Obs":TupleSpace([Discrete(9),]),
            "Large_WPM_Obs_Dense":TupleSpace([Discrete(9),]),
            "XL_WPM_Obs":TupleSpace([Discrete(9),]),
            "XL_WPM_Obs_Dense":TupleSpace([Discrete(9),])

        }

        # Policies (Unity: "behaviors") and agent-to-policy mapping fns, our setting only applies for agents in two separate teams.

        policies= {
                "PurplePlayer": PolicySpec(
                    observation_space=obs_spaces[game_name],
                    action_space=action_spaces[game_name],
                ),
                "BluePlayer": PolicySpec(
                    observation_space=obs_spaces[game_name],
                    action_space=action_spaces[game_name],
                ),

            }

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            # print(agent_id)
            return "BluePlayer" if "1_" in agent_id else "PurplePlayer"
        


        return policies, policy_mapping_fn
