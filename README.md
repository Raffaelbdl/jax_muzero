# JAX MuZero
A JAX implementation of the [MuZero agent](https://www.nature.com/articles/s41586-020-03051-4.pdf).

Everything is implemented in [JAX](https://github.com/google/jax), including the MCTS. The entire search process can be jitted and can run on accelerators such as GPUs.

## Installation
Follow [jax installation](https://github.com/google/jax/#installation), then: 
```shell
pip install -r requirements.txt
pip install autorom
AutoROM -y
pip install gym[atari]
```


## Training
Run the following command for learning to play the Atari game Breakout:
```bash
python -m experiments.breakout
```



