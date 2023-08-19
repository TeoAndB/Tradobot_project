data_window = validation_data.loc[(validation_data['date'] == year_episode_dates[t])]

# replace NaN values with 0.0
data_window = data_window.fillna(0)
# get the date
dates = data_window['date'].tolist()
# get the year for assuming yearly profits
data_window = data_window.copy()
data_window['date'] = pd.to_datetime(data_window['date'])
years = data_window['date'].dt.strftime('%Y')
year = years.iloc[0]

state = getState(data_window, t, agent)

# take action a, observe reward and next_state
closing_prices = data_window['close'].tolist()

# act epsilon-greedy
action_index = agent.act(state, closing_prices)

indices = np.where(action_index_arr_mask == action_index)
stock_i, action_index_for_stock_i = map(int, indices)

# Execute action
agent.execute_action(action_index_for_stock_i, closing_prices, stock_i, h, e, dates)

# Next state should append the t+1 data and portfolio_state. It also updates the position of agent portfolio based on agent position

if t < (l_validation - 1):
    next_data_window = validation_data.loc[(validation_data['date'] == year_episode_dates[t + 1])]
    next_state = getState(next_data_window, t + 1, agent)
    next_closing_prices = next_data_window['close'].tolist()
    next_dates = next_data_window['date'].tolist()
else:
    next_state = state
    next_closing_prices = closing_prices
    next_dates = dates

# Update portfolio and observe reward
reward = agent.update_portfolio(next_closing_prices, next_dates, stock_i)

done = True if t == l_validation - 1 else False

agent.remember(state=state, actions=(action_index, action_index_for_stock_i, stock_i, h), closing_prices=closing_prices,
               reward=reward, next_state=next_state, done=done)

for i in range(NUM_SAMPLING):
    if len(agent.memory) >= agent.batch_size:
        agent.expReplay(e)  # will also save the model on the last epoch

        batch_loss_history = agent.batch_loss_history.copy()
        validation_loss_history.extend(batch_loss_history)

if (t % 50 == 0 or t == (l_validation - 1)) and len(validation_loss_history) > 0:
    loss_per_epoch_log = sum(validation_loss_history) / len(validation_loss_history)
    loss_per_50_timesteps.append(loss_per_epoch_log)
    print(f'Episode {e + 1}, Training Loss: {loss_per_epoch_log:.4f}')
    print(f'Episode {e + 1}, Balance: {agent.portfolio_state[0, 0]:.4f}')

# next_data_window['date'] = pd.to_datetime(next_data_window['date'])
# next_year = next_data_window['date'].iloc[0].year
next_data_window = next_data_window.copy()
next_data_window['date'] = pd.to_datetime(next_data_window['date'])
next_years = next_data_window['date'].dt.strftime('%Y')
next_year = next_years.iloc[0]

if year != next_year:
    # keep track of yearly profit
    years_list_validation.append(year)
    yearly_profit_validation.append(agent.portfolio_state[0, 0])
    agent.reset_portfolio()

if e == (agent.num_epochs - 1):

    if year == final_validation_year:
        # track cumulated profits
        cumulated_profit_per_epoch = agent.portfolio_state[0, 0]
        cumulated_profits_list_validation.append(cumulated_profit_per_epoch)
        timestamps_list_validation.append(agent.timestamp_portfolio)

done = True if t == l_validation - 1 else False

if done:
    # keep track of yearly profit
    years_list_validation.append(year)
    yearly_profit_validation.append(agent.portfolio_state[0, 0])