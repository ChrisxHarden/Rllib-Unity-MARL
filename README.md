# Rllib-Unity-MARL
This is the repo for conducting MARL algorithms from Rllib in Unity with self-made Unity scenes.

## Notice:
As some of the scenes require to use `MultiDiscrete` datatype for their action space and Rllib's support for `MultiDiscrete` while using algorithms implemented by `torch` is not enough as some errors will emerge, this repo also provides customized `Rllib` and `ml-agents` which is from `unity`. It's recommended however, not required, to use our customized `Rllib`.

If using the original `Rllib`, you might encounter problems like:
1.  ![Attribute error: 'Torch Categorical' object has no attribute 'log_prob'](https://discuss.ray.io/t/running-into-attributeerror-torchcategorical-object-has-no-attribute-log-prob-when-training-mappo-in-a-unity-scene/12321/4?u=sebastianenyu).
2. ![AttributeError: '<class 'ray.rllib.models.torch.torch_distributions' object has no attribute 'cats'](https://github.com/ray-project/ray/issues/39421)



## Appendix:
1. An article which can be used as a guide to this repo is ![reinforcement-learning-with-rllib-in-the-unity-game-engine](https://medium.com/distributed-computing-with-ray/reinforcement-learning-with-rllib-in-the-unity-game-engine-1a98080a7c0d). In this post, the author gave a little introduction on combining the `Rllib` with `ml-agents` provided by `unity` using `unity` released scenes.
2. The ![Unity_asset](https://github.com/ChrisxHarden/Rllib-Unity-MARL/tree/main/Unity_asset) folder is directly cloned from ![Caleb's work](https://github.com/calebkoresh/ml-agents-dodgeball-env-ICT). We might add more scenes in the future.