from src.evaluation.evaluator import Evaluator

if __name__ == "__main__":

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
                              'competitor_price_difference_bin', 'demand_bin', 'price', 'reward'],
        'agent_type': 'q_learner',
        'env_type': 'strategic_pricing_predict',
        'num_episodes': 50000,
        'n_eval_steps': 5000,
        'num_steps': 1,
        'train_test_split': 0.2,
        'shap_num_samples': 1,
    }

    hyperparam_dict = hyperparam_dict_ds_data_predict

    evaluator = Evaluator(hyperparam_dict,
                          agent_list=['double_q_learner','sarsa','sarsa_lambda'],
                          run_path = 'runs/ds_strategicpricing',
                          n_runs=3)

    # Run all experiments and save results in the run_path
    #evaluator.run_all(verbose=False)

    # Plot evaluation graphs
    evaluator.plot_training_curve()
    evaluator.plot_reward_distribution()