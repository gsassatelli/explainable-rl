from library import *

# Import functions
from src.foundation.engine import Engine
from src.data_handler.data_handler import DataHandler
import pickle
import numpy as np

class Evaluator:
    """Evaluator to perform several experiments and show evaluation graphs."""

    def __init__(self,
                 engine_path):
        """Initialise the Evaluator.

        Args:
            engine_path (str or List[str]): list of trained engine paths
        """

        self.engine_path = engine_path
        self.engines = []
        self.eval_results = []    

        # load engines from path
        self._load_engines()

        # get evaluation data
        self._get_evaluation_results()
    
    def _load_engines(self):
        """ Load engines from the specified paths.
        """
        if isinstance(self.engine_path, str):
            self.engine_path = [self.engine_path]
        
        for path in self.engine_path:
            with open(path, 'rb') as f:
                self.engines.append(pickle.load(f))

    def hist_cum_rewards(self):
        rewards = [r[0] for r in self.eval_results[0]['rewards_hist']]

        return np.sum(rewards)

    def agent_cum_rewards(self):
        rewards = [r[0] for r in self.eval_results[0]['rewards_agent']]

        return np.sum(rewards)

    def hist_array_rewards(self):
        rewards = [r[0] for r in self.eval_results[0]['rewards_hist']]

        return np.array(rewards)

    def agent_array_rewards(self):
        rewards = [r[0] for r in self.eval_results[0]['rewards_agent']]

        return np.array(rewards)
    
    def _get_evaluation_results(self):
        """Evaluate the learned policy for the test states.
        Rewards are calculated using the average reward matrix.

        Args:
            epsilon: value of epsilon in the epsilon-greedy policy
                (default= 0 corresponds to pure exploitation)
        Returns:
            states (list): list of test states
            actions (list): list of test actions (historical)
            rewards_hist (list): list of historical rewards (calculated)
            actions_agent (list): list of recommended actions
            rewards_agent (list): list of rewards obtained by agent (calculated)
        """
        for engine in self.engines:
            eval_dict = {}
            # Save training results
            eval_dict['agent_cumrewards'] = engine.eval_agent_rewards
            eval_dict['hist_cumrewards'] = engine.eval_hist_rewards
            # Get test data from data handler
            states = engine.dh.get_states(split='test').to_numpy().tolist()
            actions = engine.dh.get_actions(split='test').to_numpy().tolist()
            rewards = engine.dh.get_rewards(split='test').to_numpy().tolist()

            # get state and action indexes
            state_dims = list(range(engine.env.state_dim))
            action_dims = list(range(engine.env.state_dim, 
                                    engine.env.state_dim+engine.env.action_dim))
            # Get the binned states
            b_states = engine.env.bin_states(states, idxs=state_dims)
            # Inverse scaling
            states = engine._inverse_scale_feature(states, engine.dh.state_labels)

            # Get the binned actions
            b_actions = engine.env.bin_states(actions, idxs=action_dims)

            # Get actions corresponding to agent's learned policy
            try:
                b_actions_agent = engine.agent.predict_actions(b_states)
            except:
                ipdb.set_trace()

            # De-bin the recommended actions
            actions_agent = engine.env.debin_states(b_actions_agent, idxs=action_dims)
            # Get reward based on agent policy
            try:
                rewards_agent = engine.agent.predict_rewards(b_states, b_actions_agent)
            except:
                ipdb.set_trace()
            # Get reward based on historical policy
            try:
                rewards_hist = engine.agent.predict_rewards(b_states, b_actions)
            except:
                ipdb.set_trace()

            #  Apply inverse scaling to actions, states, and rewards
            eval_dict['states'] = engine._inverse_scale_feature(states,
                                                engine.dh.state_labels)
            eval_dict['actions_hist'] = engine._inverse_scale_feature(actions,
                                                engine.dh.action_labels)
            eval_dict['actions_agent'] = engine._inverse_scale_feature(actions_agent,
                                                engine.dh.action_labels)
            eval_dict['rewards_hist'] = engine._inverse_scale_feature(rewards_hist,
                                                engine.dh.reward_labels)
            eval_dict['rewards_agent'] = engine._inverse_scale_feature(rewards_agent,
                                                engine.dh.reward_labels)
            
            # Save additional arrays
            eval_dict['b_actions'] = b_actions
            eval_dict['b_actions_agent'] = b_actions_agent
            eval_dict['agent_type'] = engine.dh.hyperparam_dict['agent']['agent_type']
            eval_dict['num_eval_steps'] = engine.dh.hyperparam_dict['training']['num_eval_steps']
            self.eval_results.append(eval_dict)

        

    def plot_training_curve(self):
        """Plot the training reward for a list of runs."""
        n_eval_steps = self.eval_results[0]['num_eval_steps']
        train_agent_reward = []
        train_hist_reward = []
        for eval_dict in self.eval_results:
            agent = eval_dict['agent_type']
            train_agent_reward.extend([
                [agent,
                 episode * n_eval_steps,
                 eval_dict['agent_cumrewards'][episode]]
                for episode in range(len(eval_dict['agent_cumrewards']))
            ])

            train_hist_reward.extend([
                ['historical',
                 episode * n_eval_steps,
                 eval_dict['hist_cumrewards']]
                for episode in range(len(eval_dict['agent_cumrewards']))
            ])
        train_agent_reward_df = pd.DataFrame(
            train_agent_reward,
            columns=['agent', 'episode', 'cumulative reward']
        )

        train_hist_reward_df = pd.DataFrame(
            train_hist_reward,
            columns=['agent', 'episode', 'cumulative reward']
        )

        fig, ax = plt.subplots()
        palette = {"historical": "C0", "q_learner": "C1", "double_q_learner": "C2", "sarsa": "C3", "sarsa_lambda": "C4"}
        sns.lineplot(data=train_agent_reward_df, x='episode', y='cumulative reward', hue='agent', palette=palette)
        sns.lineplot(data=train_hist_reward_df, x='episode', y='cumulative reward', hue='agent', palette=palette)
        # plt.legend()
        plt.title("Cumulative Reward (Evaluation Set)")
        plt.grid()
        plt.savefig('cumulative.png')
        # plt.show()

    def plot_reward_distribution(self):
        """Plot the distribution of rewards on the evaluation set."""

        n_eval_steps = self.eval_results[0]['num_eval_steps']
        percentiles = np.linspace(0, 100, 101)

        rewards_agent, rewards_hist = [], []
        for eval_dict in self.eval_results:
            agent = eval_dict['agent_type']
            rewards_agent_array = [r[0] for r in eval_dict['rewards_agent']]
            agent_percentiles = np.percentile(rewards_agent_array, q=percentiles)
            rewards_hist_array = [r[0] for r in eval_dict['rewards_hist']]
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
            columns=['agent', 'percentile', 'reward']
        )

        rewards_hist_df = pd.DataFrame(
            rewards_hist,
            columns=['agent', 'percentile', 'reward']
        )

        fig, ax = plt.subplots()
        palette = {"historical": "C0", "q_learner": "C1", "double_q_learner": "C2", "sarsa": "C3", "sarsa_lambda": "C4"}

        sns.lineplot(data=rewards_agent_df, x='percentile', y='reward', hue='agent', palette=palette)

        sns.lineplot(data=rewards_hist_df, x='percentile', y='reward', hue='agent', palette=palette)
        # plt.legend()

        plt.title("Reward Percentiles (Evaluation Set)")
        plt.grid()
        print("saving figure")
        plt.savefig("percentiles.png")
        # plt.show()
