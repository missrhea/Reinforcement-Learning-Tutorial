""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle as pickle
import gym

env = gym.make('Pong-v0') 
env.reset() 
for _ in range(1000): 
  env.render() 
  env.step(env.action_space.sample()) 
env.close()
