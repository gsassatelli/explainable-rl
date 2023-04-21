hyperparam_dict_ds_data_suggest = {
    'states': {'lead_time': 10, 'length_of_stay': 10, 'competitor_price_difference_bin': 4, 'demand_bin': 4,
               'price': 4},
    'actions': {'price': 10},
    'rewards': ['reward'],
    'n_samples': 1000,
    'data_path': 'data/ds-data/my_example_data.parquet',
    'col_delimiter': '|',
    'agent_type': 'q_learner',
    'env_type': 'strategic_pricing_suggest',
    'num_episodes': 500,
    'num_steps': 1,
    'train_test_split': 0.2,
    'shap_num_samples': 1,

    # new hyper-parameters
    "gamma": 0.3,
    "epsilon": 0.4,
    "epsilon_decay": 0.1,
    "epsilon_minimum": 0.1,
    "learning_rate": 0.1,
    "learning_rate_decay": 0.1,
    "learning_rate_minimum": 0.1,
    "lambda": 0.2,
    "use_uncertainty": True,
    "q_importance": 0.7,
    "normalisation": True,
    "verbose": True,
}

hyperparam_dict = {
    "dimensions": {'states': {'lead_time': 10,
                              'length_of_stay': 10,
                              'competitor_price_difference_bin': 4,
                              'demand_bin': 4,
                              'price': 4},
                   'actions': {'price': 10},
                   'rewards': ['reward']
                   },

    "dataset": {'data_path': 'data/ds-data/my_example_data.parquet',
                'col_delimiter': '|',
                'n_samples': 1000,
                'normalisation': True},

    "training": {'env_type': 'strategic_pricing_suggest',
                 'num_episodes': 500,
                 'num_steps': 1,
                 'train_test_split': 0.2},

    "agent": {'agent_type': 'q_learner',
              "gamma": 0.3,
              "epsilon": 0.4,
              "epsilon_decay": 0.1,
              "epsilon_minimum": 0.1,
              "learning_rate": 0.1,
              "learning_rate_decay": 0.1,
              "learning_rate_minimum": 0.1,
              "lambda": 0.2,
              "use_uncertainty": True,
              "q_importance": 0.7,
              },

    "explainability": {'shap_num_samples': 1},

    "program_flow": {"verbose": True}
}

# TODO: change the data handler and engine so they just take the hyperparam dict as an argument

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