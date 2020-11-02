from pysc2.env import sc2_env
from pysc2.lib import features
from absl import app

import TerranAgent

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
plt.ion()

_EPISODES = 1
_VISUALIZE = False

_SCREEN = 64
_MINIMAP = 64
_SQUARE_COUNT = 4
_TRAIN = True

_ENEMY_AGENT_Race = sc2_env.Race.random
_DIFFICULTY = sc2_env.Difficulty.very_easy



def main(unuesed_argv):
    agent = TerranAgent.TerranAgent()

    episodes = 0
    while episodes <= _EPISODES:
        try:
            with sc2_env.SC2Env(
                      map_name="Acropolis",
                      players=[sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(_ENEMY_AGENT_Race, _DIFFICULTY)],
                      agent_interface_format=features.AgentInterfaceFormat(
                             feature_dimensions=features.Dimensions(screen=_SCREEN, minimap=_MINIMAP),
                             use_feature_units=True),
                      step_mul=8,
                      visualize=_VISUALIZE
            ) as env:

                while episodes <= _EPISODES:
                    agent.setup(env.observation_spec(), env.action_spec())

                    timesteps = env.reset()
                    agent.reset()

                    while True:
                        step_actions = [agent.step(timesteps[0])]
                        agent.episode = episodes
                        if timesteps[0].last():
                            break
                        timesteps = env.step(step_actions)


                    plt.pause(0.001)
                    episodes += 1


        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    app.run(main)