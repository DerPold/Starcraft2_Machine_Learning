from __future__ import print_function

import sys
sys.path.append("game/")

import logging
import random
import numpy as np
from collections import deque
#import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Lambda, merge
from keras import backend as K
from keras.optimizers import RMSprop
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
import tensorflow as tf
logger = logging.getLogger('starcraft_agent')

#GAME = 'move_to_beacon' # the name of the game being played for log files
#CONFIG = 'nothreshold'
ACTIONS = 8 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
#OBSERVATION = 3200. # timesteps to observe before training
#EXPLORE = 1000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0005 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 6000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
#FRAME_PER_ACTION = 8
LEARNING_RATE = 1e-4

#img_rows, img_cols = 80, 80
#Convert image into Black and white

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

STEP = 0
DURATION = 0
TRAIN = True
DONE = False
STEP_DISTANCE = 8

ACTION_DO_NOTHING = 'donothing'
ACTION_ATTACK_SINGLE = 'attack'
ACTION_ATTACK_ALL = 'attackall'
ACTION_MOVE_N = 'moveNord'
ACTION_MOVE_NE = 'moveNordeast'
ACTION_MOVE_NW = 'moveNordwest'
ACTION_MOVE_W = 'moveWest'
ACTION_MOVE_SW = 'moveSouthwest'
ACTION_MOVE_S = 'moveSouth'
ACTION_MOVE_SE = 'moveSoutheast'
ACTION_MOVE_E = 'moveEast'

smart_actions = [
    ACTION_MOVE_N,
    ACTION_MOVE_NE,
    ACTION_MOVE_NW,
    ACTION_MOVE_W,
    ACTION_MOVE_SW,
    ACTION_MOVE_S,
    ACTION_MOVE_SE,
    ACTION_MOVE_E,
]


class DuelingDDQNAgent(base_agent.BaseAgent):
    def __init__(self, train):
        super(DuelingDDQNAgent, self).__init__()
        tf.compat.v1.disable_eager_execution()
        self.state_size = 2
        self.action_size = len(smart_actions)
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.001
        self.learning_rate = 0.00025
        self.model = self._build_model()
        self.target_model = self._build_model()

        self.previous_action = None
        self.previous_state = None
        self.next_action = None
        self.done = False
        self.train = train
        self.copy = True
        self.temp_episode = True

        self.random_action = 0
        self.chosen_action = 0
        self.rand = 0

        self.scores = []
        self.stepcount = 0
        self.stepcounts = []
        self.episode = 0
        # Each of our moves requires 2 steps, keep track of which step we're on in move_number
        self.move_number = 0

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(16, input_dim=2, activation='relu'))
        model.add(Dense(32, activation='relu'))

        model.add(Dense((self.action_size + 1), activation='linear'))

        def lambda_function(x):
            y = K.expand_dims(x[:, 0], axis=-1) + x[:, 1:] - K.mean(x[:, 1:], keepdims = True, axis=1)
            return y

        model.add(Lambda(lambda_function))
        model.add(Dense(8))
        #y = K.expand_dims(x[:, 0], axis=-1) + x[:, 1:] - K.mean(x[:, 1:], keepdims = True, axis=1)
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate))


        return model




    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            self.random_action += 1
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        self.chosen_action += 1

        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            states.append(state[0])
            targets.append(target_f[0])

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0, batch_size=batch_size)

        if self.epsilon > self.epsilon_min and self.temp_episode:
            self.epsilon -= self.epsilon_decay
            self.temp_episode = False

    # Some functions to get the current state
    def get_marine(self, obs):
        marine = next(unit for unit in obs.observation.feature_units
                      if unit.alliance == _PLAYER_SELF)
        return marine

    def getBeaconPosition(self, obs):
        beacon = next(unit for unit in obs.observation.feature_units
                      if unit.alliance == _PLAYER_NEUTRAL)
        beaconx, beacony = beacon.x, beacon.y
        return beaconx, beacony

    def step(self, obs):
        super(DuelingDDQNAgent, self).step(obs)
        self.stepcount += 1
        #self.rand = np.random.uniform()
        x = obs
        #print(x)
        if obs.first():
            #When to just run the Agent
            if self.episode == 0 and not self.train:
                self.epsilon = FINAL_EPSILON
                self.model.load_weights('DuelingDQNweights.h5')
                rm_prop = RMSprop(self.learning_rate)
                self.model.compile(loss='mse', optimizer=rm_prop)
                print('loaded Network')
            #Select the unit for one episode
            return actions.FUNCTIONS.select_army("select")

        #update target network every 10 episodes
        if self.episode % 10 == 0 and self.train:
            if self.copy:
                self.target_model.set_weights(self.model.get_weights())
                self.copy = False
                print('Created Target Network')

        #Modelling the states
        current_state = np.zeros(2)
        #x and y distance, marine to beacon
        marine = self.get_marine(obs)
        beaconX, beaconY = self.getBeaconPosition(obs)
        unit_x, unit_y = marine.x, marine.y
        distanceY = beaconY - unit_y
        distanceX = beaconX - unit_x

        distance_y_normalized = (distanceY + 63) / (2*64-1)
        distance_x_normalized = (distanceX + 63) / (2*64-1)

        current_state[0] = distance_x_normalized
        current_state[1] = distance_y_normalized

        current_state = np.reshape(current_state,[1, 2])

        reward = obs.reward
        self.done = reward > 0

        #only train when training the Agent
        if self.train:
            #start remember states after the first step
            if self.previous_action is not None:
                self.remember(self.previous_state, self.previous_action, reward, current_state, self.done)
                #when Replay Memory is big enough, start training
                if len(self.memory) >= REPLAY_MEMORY:
                    self.replay(32)

        if obs.last():

            score = obs.observation['score_cumulative'][0]
            self.copy = True
            self.scores.append(score)
            self.ma = sum(self.scores[-50:])/min(len(self.scores), 50)
            logger.info('Avg score (prev. 50): %s', self.ma)
            logger.info('Max score (prev. 50): %s', max(self.scores[-50:]))

            self.stepcounts.append(self.stepcount)
            logger.info('Game steps: %s', self.stepcount)
            logger.info('Average Game steps: %s', sum(self.stepcounts[-50:])/min(len(self.stepcounts), 50))

            self.previous_action = None
            self.previous_state = None

            self.rewards = []

            self.move_number = 0
            self.temp_episode = True

            if self.train:
                self.model.save_weights("DuelingDQNweights.h5", overwrite=True)
                #self.target_model.save_weights('DuelingDQNweights_Model2.h5', overwrite=True)
            return actions.FunctionCall(_NO_OP, [])

        rl_action = self.act(current_state)

        self.previous_state = current_state
        self.previous_action = rl_action

        smart_action = smart_actions[self.previous_action]

        marine_x = unit_x
        marine_y = unit_y

        if smart_action == ACTION_MOVE_N:
            destx = marine_x
            desty = marine_y - STEP_DISTANCE

        elif smart_action == ACTION_MOVE_NE:
            destx = marine_x + STEP_DISTANCE
            desty = marine_y - STEP_DISTANCE

        elif smart_action == ACTION_MOVE_NW:
            destx = marine_x - STEP_DISTANCE
            desty = marine_y - STEP_DISTANCE

        elif smart_action == ACTION_MOVE_W:
            destx = marine_x - STEP_DISTANCE
            desty = marine_y

        elif smart_action == ACTION_MOVE_SW:
            destx = marine_x - STEP_DISTANCE
            desty = marine_y + STEP_DISTANCE

        elif smart_action == ACTION_MOVE_S:
            destx = marine_x
            desty = marine_y + STEP_DISTANCE

        elif smart_action == ACTION_MOVE_SE:
            destx = marine_x + STEP_DISTANCE
            desty = marine_y + STEP_DISTANCE

        if smart_action == ACTION_MOVE_E:
            destx = marine_x + STEP_DISTANCE
            desty = marine_y

        if destx > 63:
            destx = 63
        elif destx < 0:
            destx = 0

        if desty > 63:
            desty = 63
        elif desty < 0:
            desty = 0

        return actions.FUNCTIONS.Move_screen("now", (destx, desty))
