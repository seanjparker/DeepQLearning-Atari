# Refinforcement Learning to play Atari 2600 games

## Dependencies
- OpenAI Gym (need Pong, Breakout and Space Innvaders ROMs)
- Python3 virtualenv (Recommend using pipenv)

## Install
First, clone the reposiory and run the following command inside the directory
```bash
$ pipenv --python 3.7
$ pipenv install
$ pipenv lock --pre
```

## Launch Visualisation

```
$ python3 app.py
```

## Begin training

Train DQN:

`$ python3 agents/runner.py --algo=dqn`

Train DQN with Double Q-Learning:

`$ python3 agents/runner.py --algo=dqn --doubledqn`

Train Duelling DQN:

`$ python3 agents/runner.py --algo=ddqn`

Run the command:
`$ python3 agents/runner.py --help` for more information

