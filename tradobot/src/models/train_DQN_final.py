import datetime

from src.config_model_DQN import *
from src.models.DQN_model_fin import Agent, Portfolio
#from functions import *

import logging
import sys
import time
import click
import pandas as pd
import matplotlib.pyplot as plt



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Trains the FinRL model.
    """
    logger = logging.getLogger(__name__)

    df = pd.read_csv(f'{input_filepath}/{TRAIN_DATASET}')
    NUM_STOCKS = len(df.tic.unique())

    h = hmax # muser defined maximum amount to buy

    num_actions = NUM_ACTIONS

    # TODO: make a processing data for this state, using pandas - t= 100 lag or something similar
    data = makeData(df)

    NUM_FEATURES = data.shape[1]

    portfolio = Portfolio(num_stocks=NUM_STOCKS, balance=initial_amount)
    agent = Agent(num_stocks=NUM_STOCKS, num_actions=NUM_ACTIONS, num_features=NUM_FEATURES, balance=portfolio.balance,
                  gamma=GAMMA, epsilon=EPSILON, epsilon_min=EPSILON_MIN,
                  epsilon_decay=EPSILON_DECAY, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE)

    l = len(data)-1

    loss_history = [] # will populate with: [(l1,epoch1),(l2,e1),..(l1,e_n),(l2,e_n)]
    epoch_numbers_history = []

    for e in range(NUM_EPOCHS):
        print(f'Epoch {e}')

        agent.reset()
        state = getState(data, 0)
        total_profit = 0
        agent.inventory = []

        for t in range(l):
            actions = agent.act(state)

            # next state
            next_state = getState(data, t+1)
            reward = 0

            stock_i = 0
            for action in actions:
                agent.execute_transaction(action, stock_i, h)

                stock_i +=1
            next_state = getState(data, t+1)
            reward = 0 # TODO: revise if questionable

            done= True if l==l-1 else False
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > agent.batch_size:
                agent.expReplay(e) # will also save the model on the last epoch
                loss_history.extend(agent.batch_loss_history)
                epoch_numbers_history.extent(agent.epoch_numbers)

        current_date = datetime.now()

        # Format the date as a readable string
        date_string = current_date.strftime("%d_%m_%Y")

        # Plot the running loss values after each epoch
        plt.plot(epoch_numbers_history, loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Running Loss')
        plt.title('Training Loss')
        plt.savefig(f'./reports/figures/DQN_training_loss_plot_{date_string}.png')
        plt.show()

    # After all the epochs plot the loss values in a final plot and save them


