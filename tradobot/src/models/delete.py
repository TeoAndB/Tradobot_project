# extract the years
data_window = data_window.copy()
data_window['date'] = pd.to_datetime(data_window['date'])
data_window['date'] = pd.to_datetime(data_window['date'])
years = data_window['date'].dt.strftime('%Y')
year = years.iloc[0]

next_data_window = next_data_window.copy()
next_data_window['date'] = pd.to_datetime(next_data_window['date'])
next_data_window['date'] = pd.to_datetime(next_data_window['date'])
next_years = next_data_window['date'].dt.strftime('%Y')
next_year = next_years.iloc[0]

if year != next_year:
    cumulated_profits_list_training.append(agent_balanced.portfolio_state[0, 0])
    dates_training.append(agent_balanced.timestamp_portfolio)

    years_list_training.append(year)
    yearly_profit_training.append(agent_balanced.portfolio_state[0, 0])
    agent_balanced.reset_portfolio()

    # re-buy equal amount
    for stock_i in range(agent_balanced.num_stocks):
        reward = agent_balanced.buy_1(closing_prices, stock_i, equal_amount, e, dates)
    reward = agent_balanced.update_portfolio(next_closing_prices, next_dates, stock_i)
    agent_balanced.reward = reward

# track profits only from the most recent year:
if year == final_training_year:
    cumulated_profit_per_epoch = agent_balanced.portfolio_state[0, 0]
    cumulated_profits_list_training.append(cumulated_profit_per_epoch)
    timestamps_list_training.append(agent_balanced.timestamp_portfolio)

# note the last profit at the end of the final year
if done:
    # keep track of yearly profit
    years_list_training.append(year)
    yearly_profit_training.append(agent_balanced.portfolio_state[0, 0])