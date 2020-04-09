# OLEG
reinforcement learning library

*This repo is very much a work in progress*

# What (or rather who) is OLEG?
When I started this project, I was solving 3dTetris and wanted a name for my bot. My Dad suggested Oleg, after his Russian
coworker (Tetris is from Russia). So OLEG became the shorthand for my reinforcement learning projects when talking to my parents.

OLEG is a complete rewrite of my original [DeepTetris3d]() project. It is intended to be flexible and implement a variety of models
and algorithms for any simulator environment. My goal is to solve every [OpenAI Gym] as well as some custom implemented games.

# Intended features (strikethrough denotes ~done~)
+ Automatic architecture selection for OpenAI Gyms
+ Support for all OpenAI Gyms
+ Extensible classes/API
+ Whole bunch of reinforcement learning algorithms (~DQN~, ~Double DQN~, Dueling Q, SARSA, TD3, etc.)
+ Non reinforcement learning algorithms (minimax/montecarlo)
+ Prioritized experience replay
+ ~Extensible config file system/argument parser~
+ Computer vision models (i.e. reinforcement learning direct from screen)
