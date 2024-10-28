from .algorithms.ppo import PPO

algo_registry = {}
net_registry = {}


def regist_algo(name, algo_class):
    algo_registry[name] = algo_class


def get_algo_class(name):
    return algo_registry[name]


def regist_net(name, net_class):
    net_registry[name] = net_class


def get_net_class(name):
    return net_registry[name]
