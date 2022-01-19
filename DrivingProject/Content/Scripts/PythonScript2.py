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
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import unreal_engine as ue
from unreal_engine import FVector
from tf_agents.environments import utils
from tf_agents.specs import BoundedArraySpec
from tf_agents.specs import ArraySpec
from unreal_engine.classes import ActorComponent, Actor, Blueprint 
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.metrics import py_metrics 

blueprint = ue.load_object(Blueprint, '/Game/VehicleBP/Sedan/Sedan.Sedan')
BpAsActor = blueprint.GeneratedClass.get_cdo()

AgentScore = BpAsActor.Score 

step_type = tf_agents.specs.ArraySpec((), np.int32)
# Potential steps: First, Mid, or Last

num_iterations = 2000
initial_collect_steps = 100  
collect_steps_per_iteration = 1 
collect_episodes_per_iteration = 2
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
    def Move6(self):
        BpAsActor.Forward = True
        BpAsActor.Backward = False
    def Move2(self):
        BpAsActor.Left = True
        BpAsActor.Right = False
    def Move7(self):
        BpAsActor.Left = False
        BpAsActor.Right = True
    def Move3(self):
        BpAsActor.Forward = True
        BpAsActor.Backward = False
        BpAsActor.Left = False
        BpAsActor.Right = True
    def Move8(self):
        BpAsActor.Forward = False
        BpAsActor.Backward = True
        BpAsActor.Left = False
        BpAsActor.Right = True
    def Move4(self):
        BpAsActor.Forward = True
        BpAsActor.Backward = False
        BpAsActor.Left = True
        BpAsActor.Right = False
    def Move9(self):
        BpAsActor.Forward = False
        BpAsActor.Backward = True
        BpAsActor.Left = True
        BpAsActor.Right = False
    def Move5(self):
        BpAsActor.Handbrake
        BpAsActor.HandbrakeInput = True
    def Move10(self):
        BpAsActor.Handbrake
        BpAsActor.HandbrakeInput = False



class Driving(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = BoundedArraySpec((), np.int32, minimum=0, maximum=10, name='action')
    # Potential Actions: #0 is Move Forward (Range of -1 to 1) (speed), #1 is Move Right (Range of -1 to 1) (speed) 
    self._observation_spec = ArraySpec((13,), np.float32, name='observation')
    # Things that are being observed: Camera 1 x r,g,b, Camera 1 y r,g,b, Camera 2 x r,g,b, Camera 2 y r,g,b, Camera 3 x r,g,b, Camera 3 y r,g,b, Camera 4 x r,g,b, Camera 4 y r,g,b, Speed
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
fc_layer_params = (100, 50)

environment = Driving()
tf_env = tf_py_environment.TFPyEnvironment(environment)
# reset() creates the initial time_step after resetting the environment.
time_step = tf_env.reset()
num_steps = 3
transitions = []
reward = 0

actor_net = actor_distribution_network.ActorDistributionNetwork(tf_env.observation_spec(),tf_env.action_spec(),fc_layer_params=fc_layer_params)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

tf_agent = reinforce_agent.ReinforceAgent(environment.time_step_spec(),environment.action_spec(),actor_network=actor_net,optimizer=optimizer,normalize_returns=True,train_step_counter=train_step_counter)
tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

#data_spec =  (tf.TensorSpec([5], tf.float32, 'action'),(tf.TensorSpec([1], tf.float32, 'speed'),tf.TensorSpec([4, 2], tf.float32, 'camera')))
#eplay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec, batch_size=batch_size, max_length=1000)

#data_spec = (tf.TensorSpec([5], tf.float32, 'action'),(tf.TensorSpec([12, 2] tf.float32, 'camera')))
data_spec =  (
        tf.TensorSpec([5], tf.float32, 'action'),
        (
            tf.TensorSpec([12, 2], tf.float32, 'camera')
        )
)

batch_size = 32
max_length = 1000

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec,
    batch_size=batch_size,
    max_length=max_length)

def collect_episode(environment, policy, num_episodes):

  driver = py_driver.PyDriver(
    environment,
    py_tf_eager_policy.PyTFEagerPolicy(
      policy, use_tf_function=True),
    [rb_observer],
    max_episodes=num_episodes)
  initial_time_step = environment.reset()
  driver.run(initial_time_step)

class test(ActorComponent):
    def __init__(self):
        BpAsActor.Forward = False
        BpAsActor.Backward = False
        BpAsActor.Left = False
        BpAsActor.Right = False
        environment = Driving()
        time_step = environment.reset()
        #action = np.array([0,0], dtype=np.int32)
    def BeginPlay(self):
        #try:
         # %%time
        #except:
         # pass
        # Evaluate the agent's policy once before training.
        avg_return = compute_avg_return(environment, tf_agent.policy, num_eval_episodes)
        returns = [avg_return]

    def Tick(self):
        environment.observation_spec = np.array([1,1,1,1,1,1,1,1,1,1,1,BpAsActor.Speed]) 
        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        tf_agent.train = common.function(tf_agent.train)

        # Reset the train step
        tf_agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        avg_return = compute_avg_return(environment, tf_agent.policy, num_eval_episodes)
        returns = [avg_return]

        for _ in range(num_iterations):

          # Collect a few episodes using collect_policy and save to the replay buffer.
         # collect_episode(environment, tf_agent.collect_policy, collect_episodes_per_iteration)

          # Use data from the buffer and update the agent's network.
          iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
          trajectories, _ = next(iterator)
          train_loss = tf_agent.train(experience=trajectories)  

          replay_buffer.clear()

          step = tf_agent.train_step_counter.numpy()

          if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

          if step % eval_interval == 0:
            avg_return = compute_avg_return(environment, tf_agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)
        
        action = tf.constant(1 * np.ones(data_spec[0].shape.as_list(), dtype=np.float32))
        camera = tf.constant(3 * np.ones(data_spec[1][1].shape.as_list(), dtype=np.float32))

        values = (action, camera)
        values_batched = tf.nest.map_structure(lambda t: tf.stack([t] * batch_size),values)

        replay_buffer.add_batch(values_batched)