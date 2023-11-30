from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
# Use the Algorithm's `from_checkpoint` utility to get a new algo instance
# that has the exact same state as the old one, from which the checkpoint was
# created in the first place:
path_to_checkpoint="checkpoints\\checkpoint_001865"
# algo_config=PPOConfig().multi_agent(
#     policies={
#           "PurplePlayer",
#           "BluePlayer"

#     },
#     policy_mapping_fn=(
#         lambda agent_id, episode, worker, **kw: (
#             "BluePlayer" if "1_" in agent_id else "PurplePlayer"
#         )),

#     policies_to_train=["PurplePlayer","BluePlayer"]

# )
my_new_ppo = Algorithm.from_checkpoint(path_to_checkpoint)

# Continue training.
# my_new_ppo.train()

policies=my_new_ppo.get_policy()
policies.export_model("my_models",onnx=True)


