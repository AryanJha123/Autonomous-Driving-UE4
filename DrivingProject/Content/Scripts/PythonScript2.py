from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
#import reverb

import tensorflow as tf

import tf_agents
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.specs import array_spec
from tf_agents.utils import common
#from tf_agents.replay_buffers import py_uniform_replay_buffer
#from tf_agents.replay_buffers import tf_uniform_replay_buffer
import unreal_engine as ue
from unreal_engine import FVector
from tf_agents.environments import utils
from tf_agents.specs import BoundedArraySpec
from tf_agents.specs import ArraySpec
from unreal_engine.classes import ActorComponent, Actor, Blueprint 
from tf_agents.trajectories import time_step as ts


blueprint = ue.load_object(Blueprint, '/Game/VehicleBP/Sedan/Sedan.Sedan')
BpAsActor = blueprint.GeneratedClass.get_cdo()

AgentScore = BpAsActor.Score 

step_type = tf_agents.specs.ArraySpec((), np.int32)
# Potential steps: First, Mid, or Last

num_iterations = 2000
initial_collect_steps = 100  
collect_steps_per_iteration = 1 
replay_buffer_max_length = 100000 

batch_size = 64 
learning_rate = 1e-3  
log_interval = 200  

num_eval_episodes = 10  
eval_interval = 1000  

algorithm_Forward = False
algorithm_Backward = False
algorithm_Left = False
algorithm_Right = False

class MovementSwitch:
    def switch(self, moveindex):
        if moveindex >= 0:
            return getattr(self, 'Move' + str(moveindex), lambda: default)()
        if moveindex < 0:
            return getattr(self, 'BackMove' + str(abs(moveindex)), lambda: default)()
    def Move0(self):
        BpAsActor.Forward = False
        BpAsActor.Backward = False
        BpAsActor.Left = False
        BpAsActor.Right = False
        print('pog')
    def Move1(self):
        BpAsActor.Forward = True
        BpAsActor.Backward = False
        print('woo')
    def BackMove1(self):
        BpAsActor.Forward = True
        BpAsActor.Backward = False
    def Move2(self):
        BpAsActor.Left = True
        BpAsActor.Right = False
    def BackMove2(self):
        BpAsActor.Left = False
        BpAsActor.Right = True
    def Move3(self):
        BpAsActor.Forward = True
        BpAsActor.Backward = False
        BpAsActor.Left = False
        BpAsActor.Right = True
    def BackMove3(self):
        BpAsActor.Forward = False
        BpAsActor.Backward = True
        BpAsActor.Left = False
        BpAsActor.Right = True
    def Move4(self):
        BpAsActor.Forward = True
        BpAsActor.Backward = False
        BpAsActor.Left = True
        BpAsActor.Right = False
    def BackMove4(self):
        BpAsActor.Forward = False
        BpAsActor.Backward = True
        BpAsActor.Left = True
        BpAsActor.Right = False
    def Move5(self):
        BpAsActor.Handbrake
        BpAsActor.HandbrakeInput = True
    def BackMove5(self):
        BpAsActor.Handbrake
        BpAsActor.HandbrakeInput = False



class Driving(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = BoundedArraySpec((), np.int32, minimum=-5, maximum=5, name='action')
    # Potential Actions: #0 is Move Forward (Range of -1 to 1) (speed), #1 is Move Right (Range of -1 to 1) (speed) 
    self._observation_spec = ArraySpec((5,2), np.float64, name='observation')
    # Things that are being observed: Camera 1 x + y, Camera 2 x + y, Camera 3 x + y, Camera 4 x + y, Speed
    self.state = 0
    self.episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = 0
    self.episode_ended = False
    return ts.restart(np.array([self._state], dtype=np.int32))

  def _step(self, action):
    print('a')
    MovementSwitch().switch(action)
    if self.episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()
    print('yay')
    if BpAsActor.Done == True:
        reward = AgentScore 
        return ts.termination(reward)
    elif AgentScore <= -100:
        reward = AgentScore 
        return ts.termination(reward)
    else:
        return ts.transition(np.array([self._state], dtype=np.int32),reward=AgentScore, discount=0.1)
    #if action[2] == 1:
        #BpAsActor.JumpFunction()    

#action = np.array([1], dtype=np.int32)

environment = Driving()
tf_env = tf_py_environment.TFPyEnvironment(environment)
# reset() creates the initial time_step after resetting the environment.
time_step = tf_env.reset()
num_steps = 3
transitions = []
reward = 0

data_spec =  (tf.TensorSpec([5], tf.float32, 'action'),(tf.TensorSpec([1], tf.float32, 'speed'),tf.TensorSpec([4, 2], tf.float32, 'camera')))

class test(ActorComponent):
    def __init__(self):
        BpAsActor.Forward = False
        BpAsActor.Backward = False
        BpAsActor.Left = False
        BpAsActor.Right = False
        environment = Driving()
        time_step = environment.reset()
        #action = np.array([0,0], dtype=np.int32)
    #def BeginPlay(self):
    def Tick(self):
        #print(action)                     
        get_new_card_action = np.array(3, dtype=np.int32)
        end_round_action = np.array(-3, dtype=np.int32)

        time_step = environment.reset()
        print(time_step)
        cumulative_reward = time_step.reward

        for _ in range(10):
          time_step = environment.step(get_new_card_action)
          print(time_step)
          cumulative_reward += time_step.reward

        time_step = environment.step(end_round_action)
        print(time_step)
        cumulative_reward += time_step.reward
        print('Final Reward = ', cumulative_reward)
        with tf.device('/device:GPU:0'):
          a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
          b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
          c = tf.matmul(a, b)