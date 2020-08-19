from __future__ import print_function

import sys

sys.path.append("game/")

import logging
import random
import numpy as np
import math
from collections import deque

import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import RMSprop
from pysc2.agents import base_agent
from pysc2.lib import actions
# from pysc2.lib import features
import tensorflow as tf
import keras.backend as K

logger = logging.getLogger('starcraft_agent')

FINAL_EPSILON = 0.0005  # final value of epsilon
REPLAY_MEMORY = 6000  # number of previous transitions to remember
BATCH = 32  # size of minibatch

_NO_OP = actions.FUNCTIONS.no_op.id

_PLAYER_SELF = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4

_SMART_ACTIONS = 1
_STATE_SECTORS = 5


class DDQNAgent(base_agent.BaseAgent):
    def __init__(self, train, screen_size, square_count):
        super(DDQNAgent, self).__init__()
        self.state_size = _STATE_SECTORS * _STATE_SECTORS + 1
        self.action_size = square_count * square_count + _SMART_ACTIONS
        self.screen_size = screen_size
        self.square_count = square_count
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.0025  # 0.001
        self.learning_rate = 0.00025

        self.session = None

        self.session = tf.Session()
        K.set_session(self.session)

        self.model = self._build_model()
        self.target_model = self._build_model()  # self.copy_model()

        self.previous_action = None
        self.previous_state = [[]]
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
        self.units_previous = []
        self.enemey_units_previous = []

        self.action_done = False
        self.prev_x1 = 0
        self.prev_x2 = 0
        self.prev_y1 = 0
        self.prev_y2 = 0
        self.select_reward = 0

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
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
                # print(next_state.reshape(1,-1).shape)
                shape = next_state
                target = reward + self.gamma * (
                self.target_model.predict(shape)[0][np.argmax(self.model.predict(shape))])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            states.append(state[0])
            targets.append(target_f[0])

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0, batch_size=batch_size)

        if self.epsilon > self.epsilon_min and self.temp_episode:
            self.epsilon -= self.epsilon_decay
            self.temp_episode = False

    def copy_model(self):
        self.model.save('temp_model')
        return keras.models.load_model('temp_model')

    # State design: Unit Count own army, Unit Count Enemy Army, Position own Army, Position closest enemy unit
    # Actions: Atack_screen(Position), Move Screen(Position)
    # To deternine Position to Attack or Move
    #

    # functions to define the state
    def get_army_count(self, obs):
        return len([unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_SELF])

    def get_enemy_army_count(self, obs):
        return len([unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_HOSTILE])

    def get_own_army_postition(self, obs):
        army_units = [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_SELF]
        if len(army_units) > 0:
            army_x = math.floor(sum([unit.x for unit in army_units]) / len(army_units))
            army_y = math.floor(sum([unit.y for unit in army_units]) / len(army_units))
            return army_x, army_y
        else:
            return 0, 0

    def get_closest_enemy_unit_position(self, obs):
        enemy_units = [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_HOSTILE]
        army_units = [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_SELF]
        army_pos_x = 0
        army_pos_y = 0
        if len(army_units) > 0:
            army_pos_x = sum([unit.x for unit in army_units]) / len(army_units)
            army_pos_y = sum([unit.y for unit in army_units]) / len(army_units)
        closest = enemy_units[0]
        distance = np.sqrt(np.square(army_pos_x - closest.x) + np.square(army_pos_y - closest.y))
        for eu in enemy_units:
            distance_tmp = np.sqrt(np.square(army_pos_x - eu.x) + np.square(army_pos_y - eu.y))
            if distance_tmp < distance:
                distance = distance_tmp
                closest = eu
        return closest.x, closest.y

    def get_enemy_lowest_health(self, obs):
        enemy_units = [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_HOSTILE]
        lowest = 999
        en = None
        for unit in enemy_units:
            if unit.health < lowest:
                lowest = unit.health
                en = unit
        if not en is None:
            return en.x, en.y
        else:
            return 0, 0

    def get_army_combined_health(self, obs):
        army = [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_SELF]
        health = 0
        for unit in army:
            health = health + unit.health
        return health

    def get_enemy_army_combined_health(self, obs):
        army = [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_HOSTILE]
        health = 0
        for unit in army:
            health = health + unit.health
        return health

    def enclose_value(self, value):
        new_value = value
        for i in range(0, 100):
            if i / 100 > new_value:
                new_value = i / 100
                return new_value
        return new_value

    def sector_states(self, obs):

        enemy_squares = []
        player_squares = []
        under_attack_squares = []
        units_attacking = 0
        for unit in obs.observation.feature_units:
            if unit.alliance == _PLAYER_HOSTILE:
                y = math.floor(unit.y / (_STATE_SECTORS * _STATE_SECTORS))
                x = math.ceil(unit.x) % _STATE_SECTORS
                square_id = _STATE_SECTORS * y + x
                enemy_squares.append(square_id)
            if unit.alliance == _PLAYER_SELF:
                y = math.floor(unit.y / (_STATE_SECTORS * _STATE_SECTORS))
                x = math.ceil(unit.x) % _STATE_SECTORS
                square_id = _STATE_SECTORS * y + x
                if unit.weapon_cooldown > 0:
                    units_attacking += 1
                player_squares.append(square_id)
                for ally in self.units_previous:
                    if unit.x == ally.x and unit.y == ally.y:
                        if not unit.health == ally.health:
                            under_attack_squares.append(square_id)

        # state = np.array([np.zeros( _STATE_SECTORS* _STATE_SECTORS * 2)])
        # state = np.array([[0 for j in range(_STATE_SECTORS * _STATE_SECTORS * 2)]])
        # state = np.zeros(0,(_STATE_SECTORS* _STATE_SECTORS * 2))
        # state = [_STATE_SECTORS * _STATE_SECTORS *2]
        state = np.array([0. for j in range(_STATE_SECTORS * _STATE_SECTORS + 1)])

        for value in player_squares:
            state[value] = 1
        # for value in enemy_squares:
        #    state[value*2] = 1
        for value in under_attack_squares:
            if not state[value] == 0:
                state[value] = 0.5
        units_attacking_normalized = units_attacking / len(
            [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_SELF]) if len(
            [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_SELF]) > 0 else 1
        if units_attacking_normalized <= 0.25:
            units_attacking_normalized = 0.25
        elif units_attacking_normalized > 0.25 and units_attacking_normalized <= 0.5:
            units_attacking_normalized = 0.5
        elif units_attacking_normalized > 0.5 and units_attacking_normalized <= 0.75:
            units_attacking_normalized = 0.75
        else:
            units_attacking_normalized = 1.
        state[len(state) - 1] = units_attacking_normalized
        state = state.reshape(-1, (_STATE_SECTORS * _STATE_SECTORS + 1))
        return state

    # functions for defining the actions
    def get_rectangle_points(self, action_id):

        x1 = (action_id % self.square_count) * self.square_count
        x2 = (action_id % self.square_count + 1) * self.square_count - 1

        y1 = (math.floor(action_id / self.square_count)) * self.square_count
        y2 = (math.floor(action_id / self.square_count) + 1) * self.square_count - 1

        return x1, y1, x2, y2

    def move_troops_away(self, obs):
        enemy_x, enemy_y = self.get_closest_enemy_unit_position(obs)
        army_x, army_y = self.get_own_army_postition(obs)

        x = (army_x - 8 if enemy_x - army_x > 0 else army_x + 8)
        y = (army_y - 8 if enemy_y - army_y > 0 else army_y + 8)

        if x < 0:
            x = 0
        elif x > 63:
            x = 63

        if y < 0:
            y = 0
        elif y > 63:
            y = 63

        return x, y

    def step(self, obs):
        super(DDQNAgent, self).step(obs)
        self.stepcount += 1

        if obs.first():
            # When to just run the Agent
            if self.episode == 0 and not self.train:
                self.epsilon = FINAL_EPSILON
                self.model.load_weights('DDQNVSRoachesMicroweights.h5')
                rm_prop = RMSprop(self.learning_rate)
                self.model.compile(loss='mse', optimizer=rm_prop)
                print('loaded Network')
            # Select the unit for one episode
            self.units_previous = [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_SELF]
            return actions.FUNCTIONS.select_army("select")

        # update target network every 10 episodes
        if self.episode % 10 == 0 and self.train:
            if self.copy:
                self.target_model.set_weights(self.model.get_weights())
                self.copy = False
                print('Created Target Network')

        # Modelling the states
        # current_state = np.zeros(self.state_size)
        current_state = []
        smart_action = self.action_size + 1
        rl_action = None
        if not self.action_done:
            current_state = self.sector_states(obs)
            rl_action = self.act(current_state)

            smart_action = rl_action

        if self.previous_action is None and self.action_done:
            current_state = self.sector_states(obs)
            rl_action = self.act(current_state)

            smart_action = rl_action

        reward = obs.reward + self.select_reward
        if not self.enemey_units_previous is None:
            self.done = len([unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_HOSTILE]) > len(
                self.enemey_units_previous)
        self.select_reward = 0

        # only train when training the Agent
        if self.train and not self.action_done:
            # start remember states after the first step
            if self.previous_action is not None:
                self.remember(self.previous_state, self.previous_action, reward, current_state, self.done)
                # when Replay Memory is big enough, start training
                if len(self.memory) >= REPLAY_MEMORY:
                    self.replay(32)

        if obs.last():

            score = obs.observation['score_cumulative'][0]
            self.copy = True
            self.scores.append(score)
            self.ma = sum(self.scores[-50:]) / min(len(self.scores), 50)
            logger.info('Avg score (prev. 50): %s', self.ma)
            logger.info('Max score (prev. 50): %s', max(self.scores[-50:]))

            self.stepcounts.append(self.stepcount)
            logger.info('Game steps: %s', self.stepcount)
            logger.info('Average Game steps: %s', sum(self.stepcounts[-50:]) / min(len(self.stepcounts), 50))

            self.action_done = False
            self.previous_action = None
            self.previous_state = []

            self.rewards = []

            self.move_number = 0
            self.temp_episode = True
            if self.train and self.episode % 100 == 0:
                self.model.save_weights("DDQNweights_VS_ROACHES_Micro_Model1.h5", overwrite=True)
                self.target_model.save_weights('DDQNweights_VS_ROACHES_Micro_Model2.h5', overwrite=True)
            return actions.FunctionCall(_NO_OP, [])

        # if not 331 in obs.observation.available_actions:
        #    return actions.FUNCTIONS.select_army("select")

        if not self.action_done:
            self.previous_state = current_state
            self.previous_action = rl_action

        self.units_previous = [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_SELF]
        self.enemey_units_previous = [unit for unit in obs.observation.feature_units if
                                      unit.alliance == _PLAYER_HOSTILE]
        if smart_action < (self.action_size - _SMART_ACTIONS) or self.action_done and self.previous_action < (
                self.action_size - _SMART_ACTIONS):

            if not self.action_done:
                self.action_done = True

                x1, y1, x2, y2 = self.get_rectangle_points(smart_action)
                self.prev_x1 = x1
                self.prev_y1 = y1
                self.prev_x2 = x2
                self.prev_y2 = y2

                return actions.FUNCTIONS.select_rect("select", (x1, y1), (x2, y2))
            else:
                if not 331 in obs.observation.available_actions:
                    self.action_done = False
                    return actions.FUNCTIONS.no_op("no_op")
                else:
                    units = [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_SELF and
                             ((self.prev_x1 - unit.x <= 0 >= unit.x - self.prev_x2) and
                              (self.prev_y1 - unit.y <= 0 >= unit.y - self.prev_y2))]
                    self.action_done = False
                    if len(units) == 0:
                        self.select_reward = - 1
                        return actions.FUNCTIONS.no_op("no_op")
                    x, y = self.move_troops_away(obs)
                    self.select_reward = 1
                    return actions.FUNCTIONS.Move_screen("now", (x, y))
        elif smart_action >= (self.action_size - _SMART_ACTIONS) and smart_action < self.action_size - (
                _SMART_ACTIONS - 1) or self.action_done and self.previous_action >= (
                self.action_size - _SMART_ACTIONS) and self.previous_action < self.action_size - (_SMART_ACTIONS - 1):
            if not self.action_done:
                self.action_done = True
                return actions.FUNCTIONS.select_army("add")
            else:
                if not 331 in obs.observation.available_actions:
                    self.action_done = False
                    return actions.FUNCTIONS.no_op("no_op")
                else:
                    self.action_done = False
                    x, y = self.get_closest_enemy_unit_position(obs)
                    return actions.FUNCTIONS.Attack_screen("now", (x, y))

        return actions.FUNCTIONS.no_op("no_op")
