from __future__ import print_function
import sys
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
import tensorflow as tf

sys.path.append("game/")
logger = logging.getLogger('starcraft_agent')

FINAL_EPSILON = 0.0005  # final value of epsilon
REPLAY_MEMORY = 6000  # number of previous transitions to remember
BATCH = 32  # size of minibatch

_NO_OP = actions.FUNCTIONS.no_op.id

_PLAYER_SELF = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4

_SMART_ACTIONS = 1


class DDQNAgent(base_agent.BaseAgent):
    def __init__(self, train, screen_size, square_count):
        super(DDQNAgent, self).__init__()
        tf.compat.v1.disable_eager_execution()

        self.state_size = 3
        self.action_size = square_count * square_count + _SMART_ACTIONS
        self.screen_size = screen_size
        self.square_count = square_count
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.0025  # 0.001
        self.learning_rate = 0.00025

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
        self.step_count = 0
        self.step_counts = []
        self.moving_average = 0
        self.episode = 0
        # Each of our moves requires 2 steps, keep track of which step we're on in move_number
        self.move_number = 0
        self.enemy_unit_count = 0

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
        mini_batch = random.sample(self.memory, batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
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

    # functions to define the state
    def get_army_count(self, obs):
        return len([unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_SELF])

    def get_enemy_army_count(self, obs):
        return len([unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_HOSTILE])

    def get_own_army_position(self, obs):
        army_units = [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_SELF]
        if len(army_units) > 0:
            army_x = np.round(
                sum([unit.x for unit in army_units]) / len(army_units) / (self.screen_size / self.square_count))
            army_y = np.round(
                sum([unit.y for unit in army_units]) / len(army_units) / (self.screen_size / self.square_count))
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
        if en is None:
            return 0, 0
        else:
            return en.x, en.y

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

    # functions for defining the actions
    def step(self, obs):
        super(DDQNAgent, self).step(obs)
        self.step_count += 1

        if obs.first():
            # When to just run the Agent
            if self.episode == 0 and not self.train:
                self.epsilon = FINAL_EPSILON
                self.model.load_weights('DQNweights.h5')
                rm_prop = RMSprop(self.learning_rate)
                self.model.compile(loss='mse', optimizer=rm_prop)
                print('loaded Network')
            # Select the unit for one episode
            self.enemy_unit_count = len(
                [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_HOSTILE])
            return actions.FUNCTIONS.select_army("select")

        # update target network every 10 episodes
        if self.episode % 10 == 0 and self.train:
            if self.copy:
                self.target_model.set_weights(self.model.get_weights())
                self.copy = False
                print('Created Target Network')

        # Modelling the states
        # Compute the x and y distance of own army with enemy army, normalized between 0 and 1
        army_x, army_y = self.get_own_army_position(obs)
        enemy_army_x, enemy_army_y = self.get_closest_enemy_unit_position(obs)

        distance_y = enemy_army_y - army_y
        distance_x = enemy_army_x - army_x

        distance_y_normalized = np.round((distance_y / self.screen_size), 5)
        distance_x_normalized = np.round((distance_x / self.screen_size), 5)

        enemy_unit_count = self.get_enemy_army_count(obs)
        army_unit_count = self.get_army_count(obs)

        distance_x_normalized = self.enclose_value(distance_x_normalized)
        distance_y_normalized = self.enclose_value(distance_y_normalized)

        # Compute the unit relation which army is stronger
        army_relation = army_unit_count - enemy_unit_count if army_unit_count - enemy_unit_count > 0 else 0
        unit_relation = np.round(army_relation / (army_unit_count + 1), 4)

        unit_relation = self.enclose_value(unit_relation)

        current_state = np.array([[unit_relation, distance_y_normalized, distance_x_normalized]])

        # Reward
        reward = obs.reward

        # One learning interval is ended if a enemy unit is defeated
        self.done = self.enemy_unit_count < len([unit for unit in obs.observation.feature_units
                                                 if unit.alliance == _PLAYER_HOSTILE])

        # only train when training the Agent
        if self.train:
            # start remember states after the first step
            if self.previous_action is not None:
                self.remember(self.previous_state, self.previous_action, reward, current_state, self.done)
                # when Replay Memory is big enough, start training
                if len(self.memory) >= REPLAY_MEMORY:
                    self.replay(32)

        # On the last observation compute moving average and learn from the last sequence
        if obs.last():
            score = obs.observation['score_cumulative'][0]
            self.copy = True
            self.scores.append(score)
            self.moving_average = sum(self.scores[-50:]) / min(len(self.scores), 50)
            logger.info('Avg score (prev. 50): %s', self.moving_average)
            logger.info('Max score (prev. 50): %s', max(self.scores[-50:]))

            self.step_counts.append(self.step_count)
            logger.info('Game steps: %s', self.step_count)
            logger.info('Average Game steps: %s', sum(self.step_counts[-50:]) / min(len(self.step_counts), 50))

            self.previous_action = None
            self.previous_state = []
            self.move_number = 0
            self.temp_episode = True
            if self.train and self.episode % 100 == 0:
                self.model.save_weights("DQNweights_VS_ROACHES_Model1.h5", overwrite=True)
                self.target_model.save_weights('DQNweights_VS_ROACHES_Model2.h5', overwrite=True)
            return actions.FunctionCall(_NO_OP, [])

        # Select action according to state
        rl_action = self.act(current_state)

        self.previous_state = current_state
        self.previous_action = rl_action

        self.enemy_unit_count = len(
            [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_HOSTILE])
        smart_action = rl_action

        # Select army if no unit is selected
        if not 331 in obs.observation.available_actions:
            return actions.FUNCTIONS.select_army("select")
        # Agent has square_count * square_count movement options
        if smart_action < (self.action_size - _SMART_ACTIONS):
            # map value of square id to x and y coordinates
            x = (rl_action % self.square_count + 1) * (self.square_count * self.square_count) \
                - ((self.square_count * self.square_count) / 2)
            y = (math.floor(rl_action / self.square_count) + 1) * (self.square_count * self.square_count) \
                - ((self.square_count * self.square_count) / 2)

            return actions.FUNCTIONS.Move_screen("now", (x, y))
        else:
            attacking = [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_SELF
                         and unit.weapon_cooldown > 0]
            if len(attacking) > 0:
                x, y = self.get_enemy_lowest_health(obs)
                return actions.FUNCTIONS.Attack_screen("now", (x, y))
            x, y = self.get_closest_enemy_unit_position(obs)
            return actions.FUNCTIONS.Attack_screen("now", (x, y))

        return actions.FUNCTIONS.no_op("no_op")
