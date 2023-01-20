# Fork of JAX MuZero

🚨 This fork makes the repository work for Python 3.10, not using conda, and makes a package instead of many separated folder. 🚨

See the original work https://github.com/Hwhitetooth/jax_muzero for any details.

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
python -m jax_muzero.experiments.breakout
```



