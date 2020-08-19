from pysc2.env import sc2_env
from pysc2.lib import features
from absl import app
import SARSA_Agent;
import DeepQAgent;
import time
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
matplotlib.use("TkAgg")
plt.ion()

_EPISODES = 2000
_VISUALIZE = False

_SCREEN = 64
_MINIMAP = 64

_TRAIN = True

class DynamicUpdate():
    #Suppose we know the x range
    min_x = 0
    max_x = _EPISODES

    def on_launch(self):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([],[])
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.min_x, self.max_x)
        #Other stuff
        self.ax.grid()

    def on_running(self, xdata, ydata):
        #Update data (with the new _and_ the old points)
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        self.figure.savefig('dqn_plot.png')

def main(unuesed_argv):
    agent = DeepQAgent.DQNAgent(_TRAIN)
    plot = DynamicUpdate()
    plot.on_launch();
    xdata = []
    ydata = []
    episodes = 0
    while episodes <= _EPISODES:
        try:
            with sc2_env.SC2Env(
                      map_name="MoveToBeacon",
                      players=[sc2_env.Agent(sc2_env.Race.terran)],
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
                    xdata.append(episodes)
                    ydata.append(agent.ma)
                    plot.on_running(xdata, ydata)

                    plt.pause(0.001)
                    episodes += 1
                    agent.episode = episodes
                    print('random: ', agent.random_action, ' chosen: ',agent.chosen_action,' epsilon: ', agent.epsilon, ' episode: ', agent.episode)
                    #print(agent.model.get_weights())
                    agent.chosen_action = 0
                    agent.random_action = 0

        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    app.run(main)
