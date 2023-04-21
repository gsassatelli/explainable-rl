hyperparam_dict_ds_data_suggest = {
    'states': ['lead_time', 'length_of_stay',
               'competitor_price_difference_bin', 'demand_bin', 'price'],
    'actions': [price_bin / 10 for price_bin in range(1, 11)],
    'bins': [10, 10, 4, 4, 10, 10],
    'rewards': ['reward'],
    'n_samples': 1000,
    'data_path': 'data/ds-data/my_example_data.parquet',
    'col_delimiter': '|',
    'cols_to_normalise': ['lead_time', 'length_of_stay',
                          'competitor_price_difference_bin', 'demand_bin',
                          'price', 'reward'],  # TODO: remove this and normalise everything
    'agent_type': 'q_learner',
    'env_type': 'strategic_pricing_suggest',
    'num_episodes': 500,
    'num_steps': 1,
    'train_test_split': 0.2,
    'shap_num_samples': 1
}

# TODO: group parameters by type into dictionary. E.g. all of the agent params in one dict.
# TODO: add all hyperparameters in the code
# TODO: change actions so that they are always e.g. 'price' and if env_type == suggest then action is the price bins.
# TODO: state definitions become a dictionary with state: number of bins, then in Engine we convert this into lists.
# TODO: change the datahandler and engine so they just take the hyperparam dict as an argument
# TODO:


hyperparam_dict_kaggle_data_suggest = {
    'states': ['competitorPrice', 'adFlag', 'availability', 'price'],
    'actions': [price_bin / 10 for price_bin in range(1, 11)],
    'rewards': ['revenue'],
    'feature_types': {
        'competitorPrice': "continuous",
        'adFlag': "discrete",
        'availability': "discrete",
        'price': "continuous",
        'revenue': "continuous"
    },
    'bins': [10, 2, 2, 10, 10],
    'n_samples': 2000,
    'data_path': 'data/kaggle-dummy-dataset/train.csv',
    'col_delimiter': '|',
    'cols_to_normalise': ['competitorPrice', 'adFlag', 'availability',
                          'price'],
    'agent_type': 'q_learner',
    'env_type': 'strategic_pricing_suggest',
    'num_episodes': 1000,
    'num_steps': 1,
    'train_test_split': 0.2,
    'shap_num_samples': 1
}

hyperparam_dict_ds_data_predict = {
    'states': ['lead_time', 'length_of_stay',
               'competitor_price_difference_bin', 'demand_bin'],
    'actions': ['price'],
    'rewards': ['reward'],
    'bins': [10, 10, 4, 4, 10],
    'feature_types': {
        'lead_time': "continuous",
        'length_of_stay': "continuous",
        'competitor_price_difference_bin': "discrete",
        'demand_bin': "discrete",
        'price': "continuous",
        'reward': "continuous"
    },
    'n_samples': 100000,
    'data_path': 'data/ds-data/my_example_data.parquet',
    'col_delimiter': '|',
    'cols_to_normalise': ['lead_time', 'length_of_stay',
                          'competitor_price_difference_bin', 'demand_bin',
                          'price', 'reward'],
    'agent_type': 'q_learner',
    'env_type': 'strategic_pricing_predict',
    'num_episodes': 5000,
    'num_steps': 1,
    'train_test_split': 0.2,
    'shap_num_samples': 10
}

hyperparam_dict_kaggle_data_predict = {
    'states': ['competitorPrice', 'adFlag', 'availability'],
    'actions': ['price'],
    'rewards': ['revenue'],
    'feature_types': {
        'competitorPrice': "continuous",
        'adFlag': "discrete",
        'availability': "discrete",
        'price': "continuous",
        'revenue': "continuous"
    },
    'bins': [10, 2, 2, 10],
    'n_samples': 20000,
    'data_path': 'data/kaggle-dummy-dataset/train.csv',
    'col_delimiter': '|',
    'cols_to_normalise': ['competitorPrice', 'adFlag', 'availability',
                          'price'],
    'agent_type': 'q_learner',
    'env_type': 'strategic_pricing_predict',
    'num_episodes': 1000,
    'num_steps': 1,
    'train_test_split': 0.2,
    'shap_num_samples': 1
}