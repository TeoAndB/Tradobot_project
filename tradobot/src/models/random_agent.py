# RANDOM AGENT #########################################################################################################
########################################################################################################################
# EPSILON_DECAY = 1.0 #no decay
# EPSILON = 2.0 # no chance of randomness, as porbability is extracted from a unfirom distribution
#
#
# agent_random = Agent(num_stocks=NUM_STOCKS, actions_dict=ACTIONS_DICTIONARY, h=h, num_features=num_features, balance=INITIAL_AMOUNT, name_stocks=cols_stocks,
#               gamma=GAMMA, epsilon=EPSILON, epsilon_min=EPSILON_MIN,
#               epsilon_decay=EPSILON_DECAY, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, tau=TAU, num_epochs=NUM_EPOCHS)
#
# action_index_arr_mask = np.arange(0, agent_random.num_actions * agent_random.num_stocks, 1, dtype=int).reshape(agent_random.num_stocks,
#                                                                                                  agent_random.num_actions)
#
# # RUNNING BASELINE FOR VALIDATION PERIOD  #######################################################################
#
# unique_dates_validation = validation_data['date'].unique()
#
# epoch_numbers_history_validation = []
# cumulated_profits_list_validation = [INITIAL_AMOUNT]
# dates_validation = [unique_dates_validation[0]]
#
# agent_random.reset()
# agent_random.Q_network.eval()  # Set the model to evaluation mode
# agent_random.epsilon = 0.0  # no exploration
# unique_dates_validation = validation_data['date'].unique()
#
# l_validation = len(unique_dates_validation)
#
# for t in range(l_validation):
#     ####
#     data_window = validation_data.loc[(validation_data['date'] == unique_dates_validation[t])]
#
#     # replace NaN values with 0.0
#     data_window = data_window.fillna(0)
#     dates = data_window['date'].tolist()
#
#     state = getState(data_window, t, agent_random)
#
#     closing_prices = data_window['close'].tolist()
#
#     # take action a, observe reward and next_state
#     agent_random.epsilon = 2.0
#
#     action_index = agent_random.act(state, closing_prices)
#
#     # Find the indeces (stock_i, action_index_for_stock_i) where action_index  is present in action_index_arr_mask
#     indices = np.where(action_index_arr_mask == action_index)
#     stock_i, action_index_for_stock_i = map(int, indices)
#
#     reward = agent_random.execute_action(action_index_for_stock_i, closing_prices, stock_i, h, e, dates)
#
#     updated_balance = agent_random.portfolio_state[0, 0]
#     cumulated_profits_list_validation.append(updated_balance)
#     dates_validation.append(agent_random.timestamp_portfolio)
#
#     # Next state should append the t+1 data and portfolio_state. It also updates the position of agent_random portfolio based on agent_random position
#     # next_state, agent_random = getState(data_window, t + 1, agent_random)
#     # state = next_state
#
#     done = True if t == l_validation - 1 else False
#
# # printing portfolio state for validation at the end
# df_portfolio_state = pd.DataFrame(agent_random.portfolio_state, columns=cols_stocks)
# df_portfolio_state.insert(0, 'TIC', agent_random.portfolio_state_rows)
# print(f'Validation Period Random Agent: Portfolio state is \n: {df_portfolio_state}')
#
# # saving the explainability file
# current_date = datetime.datetime.now()
# date_string = current_date.strftime("%Y-%m-%d_%H_%M")
# agent_random.explainability_df.to_csv(
#     f'./reports/results_DQN/baseline_results/minute_frequency_data/random_agent_validation_period_{dataset_name}_{date_string}_{selected_data_entries}.csv', index=False)


# RUNNING BASELINE FOR TESTING PERIOD  #######################################################################
# print('Validation Phase for Random Agent')
# agent_random.reset()
# agent_random.epsilon = 0.0 # no exploration
# agent_random.memory = []
#
# unique_dates_testing = test_data['date'].unique()
# cumulated_profits_list_testing = [INITIAL_AMOUNT]
# dates_testing = [unique_dates_testing[0]]
# e = agent_random.num_epochs-1 #for explainability
#
# l_testing = len(unique_dates_testing)
# for t in range(l_testing):
#     ####
#     data_window = test_data.loc[(test_data['date'] == unique_dates_testing[t])]
#
#     # replace NaN values with 0.0
#     data_window = data_window.fillna(0)
#     dates = data_window['date'].tolist()
#
#     state = getState(data_window, t, agent_random)
#
#     closing_prices = data_window['close'].tolist()
#
#     # take action a, observe reward and next_state
#     agent_random.epsilon = 2.0
#     action_index = agent_random.act(state, closing_prices)
#
#     # Find the indeces (stock_i, action_index_for_stock_i) where action_index  is present in action_index_arr_mask
#     indices = np.where(action_index_arr_mask == action_index)
#     stock_i, action_index_for_stock_i = map(int, indices)
#
#     reward = agent_random.execute_action(action_index_for_stock_i, closing_prices, stock_i, h, e, dates)
#
#     updated_balance = agent_random.portfolio_state[0, 0]
#     cumulated_profits_list_testing.append(updated_balance)
#     dates_testing.append(dates[0])
#
#     # Next state should append the t+1 data and portfolio_state. It also updates the position of agent_random portfolio based on agent_random position
#     # next_state, agent_random = getState(data_window, t + 1, agent_random)
#     # state = next_state
#
#     done = True if t == l_testing - 1 else False
#
# # printing portfolio state for testing at the end
# df_portfolio_state = pd.DataFrame(agent_random.portfolio_state, columns=cols_stocks)
# df_portfolio_state.insert(0, 'TIC', agent_random.portfolio_state_rows)
# print(f'Testing Period Random Agent: Portfolio state is \n: {df_portfolio_state}')
#
# # saving the explainability file
# current_date = datetime.datetime.now()
# date_string = current_date.strftime("%Y-%m-%d_%H_%M")
# agent_random.explainability_df.to_csv(
#     f'./reports/results_DQN/baseline_results/minute_frequency_data/random_agent_testing_period_{dataset_name}_{date_string}_{selected_data_entries}.csv', index=False)
#
# # Plotting
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
#
# # Plot for testing data
# ax1.plot(dates_validation, cumulated_profits_list_validation)
# ax1.set_title("Random Agent: Cumulated Profits Over Time (Validation Period))")
# ax1.set_xlabel("Dates")
# ax1.set_ylabel("Cumulated Profits")
# ax1.tick_params(axis='x', rotation=45)
# ax1.grid(True)
#
# # Reduce the number of dates shown on the x-axis for testing data
# num_dates = 6
# skip = max(1, len(dates_validation) // num_dates)
# ax1.set_xticks(range(0, len(dates_validation), skip))
# ax1.set_xticklabels(dates_validation[::skip])
# ax1.set_xlim(0, len(dates_validation))
# ax1.tick_params(axis='x', labelsize=8)
# fig.autofmt_xdate(bottom=0.2)
#
# # Plot for validation data
# ax2.plot(dates_testing, cumulated_profits_list_testing)
# ax2.set_title("Random Agent: Cumulated Profits Over Time (Testing Period)")
# ax2.set_xlabel("Dates")
# ax2.set_ylabel("Cumulated Profits")
# ax2.tick_params(axis='x', rotation=45)
# ax2.grid(True)
#
# # Reduce the number of dates shown on the x-axis for validation data
# num_dates = 6
# skip = max(1, len(dates_testing) // num_dates)
# ax2.set_xticks(range(0, len(dates_testing), skip))
# ax2.set_xticklabels(dates_testing[::skip])
# ax2.set_xlim(0, len(dates_testing))
# ax2.tick_params(axis='x', labelsize=8)
# fig.autofmt_xdate(bottom=0.2)
#
# # Save the figure in the specified folder path
# plt.savefig(
#     f'./reports/figures/baseline_models/minute_frequency_data/random_agent_testing_and_validation_periods_profits_for_{dataset_name}_{date_string}_{selected_data_entries}.png')
#
# # Show the figures
# plt.show()

# BALANCED AGENT #########################################################################################################
##########################################################################################################################
# divides the investing amount equally between stocks and holds ##########################################################
