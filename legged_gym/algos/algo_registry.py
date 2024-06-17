from .algorithms.ppo import PPO

algo_registry = {}


def regist_algo(name, algo_class):
    algo_registry[name] = algo_class


def get_algo_class(name):
    return algo_registry[name]
