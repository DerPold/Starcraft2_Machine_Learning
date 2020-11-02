from __future__ import print_function

import sys

sys.path.append("game/")

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import units
# from pysc2.lib import features

# Terran Bot, actions to start with:
# build SCV, need workers
# SCV Actions:
# gather minerals
# gather gas
# scout enemy base
# build supply depot, need pop space. action has to consist of select SCV and place depot       | create method for
# build barracks, for building units. action has to consist of select SCV and place barracks    | selecting a SCV
# build forge, for upgrades. action has to consist of select SCV and place forge                | Choose building space
# build fabric, for tanks. action has to consist of select SCV and place fabric                 | Place Building
# build airport, for flying units. ...
# build command center, for building SUVs and shorter mining paths
#

_PLAYER_SELF = 1
_PLAYER_NEUTRAL = 3
_PLAYER_HOSTILE = 4


class TerranAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TerranAgent, self).__init__()
        self.start_position_command_center_x = 0
        self.start_position_command_center_y = 0
        self.buildable_tiles_small = []
        self.buildable_tiles_medium = []
        self.buildable_tiles_big = []
        self.mining_positions = []

    # get starting position of the first Command Center, for knowing in which area to place the buildings
    def get_starting_position(self, obs):
        command_center = self.get_player_units_by_type(units.Terran.CommandCenter, obs)
        self.start_position_command_center_x = command_center.x
        self.start_position_command_center_y = command_center.y

    # method for getting all units of a given type
    def get_player_units_by_type(self, unit_type, obs):
        return [unit for unit in obs.observation.raw_units if unit.type == unit_type and unit.alliance == _PLAYER_SELF]

    # build a given building by an idle, or the closest SCV (At a given location)
    def build_building(self, building_type, obs):
        scvs = self.get_player_units_by_type(obs, units.Terran.SCV)

    # use an idle unit to scout the enemies position (Also use SCV if no other unit is available)
    def scout_enemy_player(self, unit_type, obs):
        units = [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_SELF]

    # build a unit of the given type.
    # has to find out in which building type the unit can be build
    # should use building which is idle or close to idling
    def build_unit(self, unit_type, obs):
        units = self.get_player_units_by_type(unit_type, obs)

    # the buildings fabric, airport and barracks can build a reactor for enabling the ability to produce two units at
    # once
    def build_reactor(self, building, obs):
        True

    # the buildings fabric, airport and barracks can build a science lab for researching abilities for units
    def build_science_lab(self, building, obs):
        True

    # send SCVs to mine a certain mineral patch
    def mine_minerals(self, pos_x, pos_y, obs):
        True

    def find_building_position(self, obs):
        x = 0
        y = 0

        return x, y

    def step(self, obs):
        super(TerranAgent, self).step(obs)

        return actions.FUNCTIONS.no_op("no_op")
