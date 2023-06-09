#!/usr/bin/env python3

from connex import agent, environment

def main():
  game_config = environment.configuration(
    rows=6,
    columns=7,
    row_length=4,
    action_space_size=7,
    apply_function=environment.drop_apply,
    legal_actions_function=environment.drop_legal_actions)

  model_config = agent.configuration(
    training_steps=int(1000e3),
    checkpoint_interval=int(1e3),
    num_simulations=512,
    window_size=int(1e6),
    batch_size=64,
    num_unroll_steps=5,
    td_steps=game_config.action_space_size,
    learning_rate=1e-3,
    weight_decay=1e-4)

  model = agent.learn(game_config, model_config, verbose=True)

if __name__ == "__main__":
  main()
