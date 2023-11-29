import argparse
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
# from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from base_unity_env import Unity3DEnv
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.replay_buffers.multi_agent_prioritized_replay_buffer import MultiAgentPrioritizedReplayBuffer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

### parser adding arguments
parser = argparse.ArgumentParser()

### Env setting, could be enlarged
parser.add_argument(
    "--env",
    type=str,
    default="Large_WPM_Obs",
    choices=[
        "Large_WPM_Obs",
        "Large_WPM_Obs_Dense",
        "XL_WPM_Obs",
        "XL_WPM_Obs_Dense",
    ],
    help="The name of the Env to run in the Unity3D editor",
)


### for compiled games
parser.add_argument(
    "--file-name",
    type=str,
    default=None,
    help="The Unity3d binary (compiled) game, e.g. "
    "'/home/ubuntu/soccer_strikers_vs_goalie_linux.x86_64'. Use `None` for "
    "a currently running Unity3D editor.",
)


### for training from checkpoints
parser.add_argument(
    "--from-checkpoint",
    type=str,
    default=None,
    help="Full path to a checkpoint file for restoring a previously saved "
    "Algorithm state.",
)


parser.add_argument("--num-workers", type=int, default=0)


parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=9999, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=20000000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward",
    type=float,
    default=9999.0,
    help="Reward at which we stop training.",
)
parser.add_argument(
    "--horizon",
    type=int,
    default=3000,
    help="The max. number of `step()`s for any episode (per agent) before "
    "it'll be reset again automatically.",
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--address",
    
    default=None,
    help="The ray address specifier.",
)


def gpu_test():


    # Define a simple feedforward neural network
    class SimpleModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

   # Generate some random data for demonstration
    input_size = 64
    hidden_size = 128
    output_size = 10
    batch_size = 32
    num_epochs = 10

    # Create random input and target tensors (replace this with your data)
    dummy_input = torch.randn(batch_size, input_size)
    dummy_target = torch.randn(batch_size, output_size)

    # Create DataLoader for the random data
    dataset = TensorDataset(dummy_input, dummy_target)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Check if a GPU is available and move the model to the GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel(input_size, hidden_size, output_size).to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(num_epochs):
        

        total_loss = 0.0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            print("inputs on {}".format(device))

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {total_loss / len(data_loader)}")



if __name__ == "__main__":
   
    os.environ["RAY_TMPDIR"] = "/tmp/my_tmp_dir"
    args = parser.parse_args()
    ray.init(_temp_dir=f"/tmp/my_tmp_dir",num_cpus=10,num_gpus=1,ignore_reinit_error=True)

    

    tune.register_env(
        "unity3d",
        lambda c: Unity3DEnv(
            file_name=c["file_name"],
            no_graphics=c["file_name"] is not None,
            episode_horizon=c["episode_horizon"],
        ),
    )

    # Get policies (different agent types; "behaviors" in MLAgents) and
    # the mappings from individual agents to Policies.
    policies, policy_mapping_fn = Unity3DEnv.get_policy_configs_for_game(args.env)

    config = (
        PPOConfig()
        .environment(
            "unity3d",
            env_config={
                "file_name": args .file_name,
                "episode_horizon": args.horizon,
            },
        )
        .framework(args.framework,torch_compile_learner=True)
        # For running in editor, force to use just one Worker (we only have
        # one Unity running)!
        .rollouts(
            num_rollout_workers=args.num_workers if args.file_name else 0,
            rollout_fragment_length='auto',
            batch_mode="complete_episodes"
        )
        .training(
            lr=0.0003,
            lambda_=0.95,
            gamma=0.99,
            sgd_minibatch_size=256,
            train_batch_size=4000,
            num_sgd_iter=20,
            clip_param=0.2,
            model={"fcnet_hiddens": [512, 512]},
            _enable_learner_api=True
            # replay_buffer_config={
            #     "type":"MultiAgentPrioritizedReplayBuffer",
            #     "storage_unit": "episodes"}

        )
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        # .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .resources(num_gpus=1,num_gpus_per_learner_worker=1)
        #.resources(num_gpus=0)   
    )



    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }




    # Run the experiment.
    results = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            stop=stop,
            verbose=1,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
        ),
    ).fit()

    # And check the results.
    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
