# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import ray
    import torch
    from ray import tune
    from ray.rllib.algorithms import ppo

    ray.init()
    print(torch.cuda.is_available())
    num=ray.get_gpu_ids()
    print(num)



    ray.shutdown()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
