hyperparam_dict = {
    "dimensions": {'states': {'competitorPrice': 10,
                              'adFlag': 2,
                              'availability': 2
                              },
                   'actions': {'price': 5},
                   'rewards': ['revenue']
                   },

    "dataset": {'data_path': 'tests/test_env_data.csv',
                'col_delimiter': ',',
                'n_samples': 50,
                'normalisation': True},

    "training": {'env_type': 'strategic_pricing_predict',
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
