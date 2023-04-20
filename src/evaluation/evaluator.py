from src.foundation.engine import Engine
from src.data_handler.data_handler import DataHandler
from src.explainability.pdp import PDP
from src.explainability.shap_values import ShapValues
from datetime import datetime
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import ipdb



class Evaluator():
    """ Evaluator class to perform several experiemnts and show evaluation graphs."""

    __slots__ = ["hyperparam_dict","agent_list","run_path","n_runs"]
    
    def __init__(self,
                 hyperparam_dict,
                 agent_list,
                 run_path,
                 n_runs=3):
        """Args:
            hyperparam_dict: hyperparameter settings
            agent_list (list): agents to test
            n_runs: number of experiments to perform for each agent
            run_path: directory to save the results
        """

        self.hyperparam_dict = hyperparam_dict
        self.agent_list = agent_list
        self.n_runs = n_runs
        self.run_path = run_path


    def train_evaluate_agent(self,
                             hyperparam_dict,
                             verbose=True):
        """ Train and evaluate agent specified by hyperparam_dict.
        
        Args:
            hyperparam_dict: hyperparameter settings
            verbose: verbose flag
        
        Returns:
            eval_results (dict): dictionary containing evaluation results        
        """

        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f"{timestamp}: Load data")
        state_labels = hyperparam_dict['states']
        action_labels = hyperparam_dict['actions']
        reward_labels = hyperparam_dict['rewards']
        n_samples = hyperparam_dict['n_samples']
        n_episodes = hyperparam_dict['num_episodes']
        shap_num_samples = hyperparam_dict['shap_num_samples']
        dh = DataHandler(data_path=hyperparam_dict['data_path'],
                        state_labels=state_labels,
                        action_labels=action_labels,
                        reward_labels=reward_labels,
                        n_samples=n_samples)

        # Preprocess the data
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        if verbose:
            print(f"{timestamp}: Preprocess data")
        dh.prepare_data_for_engine(col_delimiter=hyperparam_dict['col_delimiter'],
                                cols_to_normalise=hyperparam_dict[
                                    'cols_to_normalise'])

        # Create engine
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        if verbose:
            print(f"{timestamp}: Initialize Engine")
        engine = Engine(dh,
                        agent_type=hyperparam_dict['agent_type'],
                        env_type=hyperparam_dict['env_type'],
                        num_episodes=hyperparam_dict['num_episodes'],
                        num_steps=hyperparam_dict['num_steps'],
                        bins=hyperparam_dict['bins'],
                        train_test_split=hyperparam_dict['train_test_split']
                        )
        # Create world
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        if verbose:
            print(f"{timestamp}: Create the world")
        engine.create_world()

        # Train agent
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        if verbose:
            print(f"{timestamp}: Train the agent on {n_samples} samples and {n_episodes} episodes")
        engine.train_agent(evaluate=True,n_eval_steps=hyperparam_dict['n_eval_steps'])

        # Evaluate Agent
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        if verbose:
            print(f"{timestamp}: Evaluate agent")

        eval_results = engine.evaluate_agent()
        
        return eval_results

    

    def run_all(self,
                verbose=False):
        """ Trains and evaluates agents for a number of runs.

        Args:
            hyperparam_dict (dict): hyper-parameter settings
            agent_list (list): agents to test
            n_runs: number of experiments to perform for each agent
            run_path: directory to save the results
        """
        # Create run directory
        run_path = self.run_path
        if not os.path.exists(run_path):
            os.makedirs(run_path)

        # Perform all the experiments specified in hyperparam_dict
        for agent in self.agent_list:
            if verbose:
                print(f"Testing agent {agent}...")
            hyperparam_dict = self.hyperparam_dict.copy()
            hyperparam_dict['agent_type'] = agent

            for run in tqdm(range(self.n_runs)):
                eval_results = self.train_evaluate_agent(hyperparam_dict, verbose)
                eval_results['agent'] = agent # save agent
                # Save eval results in the experiment dict
                if verbose:
                    print('Saving results...')
                with open(f"{run_path}/{agent}_run_{run}.pkl", 'wb') as f:  
                    pickle.dump(eval_results, f)

    
    def _get_all_experiments_results(self):
        """ Retrieves all experiment results from run path.
        
        Returns:
            eval_list (list[dict]): list containing evaluation dict results
        
        """
        eval_list = []
        files = os.listdir(self.run_path)
        for file_path in files:
            with open(self.run_path + '/' + file_path,'rb') as f:
                eval = pickle.load(f)
                eval_list.append(eval)
        return eval_list
        
    def plot_training_curve(self):
        """ Plot the training reward for a list of runs.
  
        """
        eval_list = self._get_all_experiments_results()
        n_eval_steps = self.hyperparam_dict['n_eval_steps']
        train_agent_reward = []
        train_hist_reward = []
        for eval_results in eval_list:
            agent = eval_results['agent']
            train_agent_reward.extend([
                [agent,
                episode*n_eval_steps,
                eval_results['agent_cumrewards'][episode]]
                for episode in range(len(eval_results['agent_cumrewards']))
            ])

            train_hist_reward.extend([
                ['historical',
                episode*n_eval_steps,
                eval_results['hist_cumrewards']]
                for episode in range(len(eval_results['agent_cumrewards']))
            ])        
        train_agent_reward_df = pd.DataFrame(
            train_agent_reward,
            columns=['agent','episode','cumulative reward']
        )

        train_hist_reward_df = pd.DataFrame(
            train_hist_reward,
            columns=['agent','episode','cumulative reward']
        )

        fig, ax = plt.subplots()
        palette ={"historical": "C0", "q_learner": "C1", "double_q_learner": "C2", "sarsa": "C3","sarsa_lambda":"C4"}
        sns.lineplot(data=train_agent_reward_df, x='episode', y='cumulative reward', hue='agent', palette=palette)
        sns.lineplot(data=train_hist_reward_df, x='episode', y='cumulative reward', hue='agent', palette=palette)
        #plt.legend()
        plt.title("Cumulative Reward (Evaluation Set)")
        plt.grid()
        plt.savefig('cumulative.png')
        #plt.show()

    def plot_reward_distribution(self):
        """ Plot the distribution of rewards on the evaluation set.
        Args:
            eval_results_list (list[Dict]): evaluation results of all the runs we want to show
            agent_list (list[str]): type of agent used in each run 
        """
  
        eval_list = self._get_all_experiments_results()
        n_eval_steps = self.hyperparam_dict['n_eval_steps']
        percentiles = np.linspace(0,100,101)

        rewards_agent, rewards_hist = [], []
        for eval_results in eval_list:
            agent = eval_results['agent']
            rewards_agent_array = [r[0] for r in eval_results['rewards_agent']]
            agent_percentiles = np.percentile(rewards_agent_array, q=percentiles) 
            rewards_hist_array = [r[0] for r in eval_results['rewards_hist']]
            hist_percentiles = np.percentile(rewards_hist_array, q=percentiles) 
            rewards_agent.extend([
                [agent,
                 percentiles[p],
                 agent_percentiles[p]
                ]
                for p in range(percentiles.shape[0])
            ])
            rewards_hist.extend([
                ['historical',
                 percentiles[p],
                 hist_percentiles[p]
                ]
                for p in range(percentiles.shape[0])
            ])                
        rewards_agent_df = pd.DataFrame(
            rewards_agent,
            columns=['agent','percentile','reward']
        )

        rewards_hist_df = pd.DataFrame(
            rewards_hist,
            columns=['agent','percentile','reward']
        )

        fig, ax = plt.subplots()
        palette ={"historical": "C0", "q_learner": "C1", "double_q_learner": "C2", "sarsa": "C3","sarsa_lambda":"C4"}

        sns.lineplot(data=rewards_agent_df, x='percentile', y='reward', hue='agent', palette=palette)

        sns.lineplot(data=rewards_hist_df, x='percentile', y='reward', hue='agent', palette=palette)
        #plt.legend()

        plt.title("Reward Percentiles (Evaluation Set)")
        plt.grid()
        print("saving figure")
        plt.savefig("percentiles.png")
       # plt.show()

