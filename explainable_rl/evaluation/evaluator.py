from explainable_rl.foundation.library import *

# Import functions
from explainable_rl.foundation.engine import Engine


class Evaluator:
    """Evaluator class which evaluates a list of trained agents
     and produces RL evaluation graphs."""

    def __init__(self, engines):
        """Initialise the Evaluator.

        Args:
            engines (Engine or List[Engine]): one or list of trained engine
        """

        self.engines = engines
        self.eval_results = []

        # get evaluation data
        self._get_evaluation_results()

    def hist_cum_rewards(self):
        """Calculate the cumulative historical rewards on test set.

        Returns:
            hist_cum_rewards (float): total reward on test set using historical policy.
        """
        rewards = self.hist_array_rewards()
        return [np.sum(r) for r in rewards]

    def agent_cum_rewards(self):
        """Calculate the cumulative agent rewards on test set.

        Returns:
            agent_cum_rewards (float): total reward on test set using historical policy.
        """
        rewards = self.agent_array_rewards()
        return [np.sum(r) for r in rewards]

    def hist_array_rewards(self):
        """Calculate the individual historical rewards for each test set sample.
         
        Returns:
            hist_array_rewards (List[float]): array of historical rewards on test set.
        """
        rewards = [
            np.array([r[0] for r in eval_dict["rewards_hist"]])
            for eval_dict in self.eval_results
        ]
        return rewards

    def agent_array_rewards(self):
        """Calculate the individual agent rewards for each test set sample.
         
        Returns:
            agent_array_rewards (List[float]): array of agent rewards on test set.
        """
        rewards = [
            np.array([r[0] for r in eval_dict["rewards_agent"]])
            for eval_dict in self.eval_results
        ]
        return rewards

    def _get_evaluation_results(self):
        """Evaluate the engines on the test set.

        This method fills in self.eval_results, which is a list of dictionaries
        containing all the relevant evaluation metrics.        
        """
        for engine in self.engines:
            eval_dict = {}
            # Save training results
            eval_dict["agent_cumrewards"] = engine.eval_agent_rewards
            eval_dict["hist_cumrewards"] = engine.eval_hist_rewards

            # Get test data from data handler
            states = engine.dh.get_states(split="test").to_numpy().tolist()
            actions = engine.dh.get_actions(split="test").to_numpy().tolist()
            rewards = engine.dh.get_rewards(split="test").to_numpy().tolist()

            # Get state and action indexes
            state_dims = list(range(engine.env.state_dim))
            action_dims = list(
                range(
                    engine.env.state_dim, engine.env.state_dim + engine.env.action_dim
                )
            )
            # Get the binned states
            b_states = engine.env.bin_states(states, idxs=state_dims)
            # Inverse scaling
            states = engine.inverse_scale_feature(states, engine.dh.state_labels)

            # Get the binned actions
            b_actions = engine.env.bin_states(actions, idxs=action_dims)

            # Get actions corresponding to agent's learned policy
            b_actions_agent = engine.agent.predict_actions(b_states)

            # De-bin the recommended actions
            actions_agent = engine.env.debin_states(b_actions_agent, idxs=action_dims)
            # Get reward based on agent policy
            rewards_agent = engine.agent.predict_rewards(b_states, b_actions_agent)
            # Get reward based on historical policy
            rewards_hist = engine.agent.predict_rewards(b_states, b_actions)

            #  Apply inverse scaling to actions, states, and rewards
            eval_dict["states"] = engine.inverse_scale_feature(
                states, engine.dh.state_labels
            )
            eval_dict["actions_hist"] = engine.inverse_scale_feature(
                actions, engine.dh.action_labels
            )
            eval_dict["actions_agent"] = engine.inverse_scale_feature(
                actions_agent, engine.dh.action_labels
            )
            eval_dict["rewards_hist"] = engine.inverse_scale_feature(
                rewards_hist, engine.dh.reward_labels
            )
            eval_dict["rewards_agent"] = engine.inverse_scale_feature(
                rewards_agent, engine.dh.reward_labels
            )

            # Save additional arrays
            eval_dict["b_actions"] = b_actions
            eval_dict["b_actions_agent"] = b_actions_agent
            eval_dict["agent_type"] = engine.dh.hyperparam_dict["agent"]["agent_type"]
            eval_dict["num_eval_steps"] = engine.dh.hyperparam_dict["training"][
                "num_eval_steps"
            ]
            self.eval_results.append(eval_dict)

    def plot_training_curve(self):
        """Plot the training reward for a list of runs.
        """
        n_eval_steps = self.eval_results[0]["num_eval_steps"]
        train_agent_reward = []
        train_hist_reward = []
        for eval_dict in self.eval_results:
            agent = eval_dict["agent_type"]
            train_agent_reward.extend(
                [
                    [
                        agent,
                        episode * n_eval_steps,
                        eval_dict["agent_cumrewards"][episode],
                    ]
                    for episode in range(len(eval_dict["agent_cumrewards"]))
                ]
            )

            train_hist_reward.extend(
                [
                    ["historical", episode * n_eval_steps, eval_dict["hist_cumrewards"]]
                    for episode in range(len(eval_dict["agent_cumrewards"]))
                ]
            )
        train_agent_reward_df = pd.DataFrame(
            train_agent_reward, columns=["agent", "episode", "cumulative reward"]
        )

        train_hist_reward_df = pd.DataFrame(
            train_hist_reward, columns=["agent", "episode", "cumulative reward"]
        )

        palette = {
            "historical": "C0",
            "q_learner": "C1",
            "double_q_learner": "C2",
            "sarsa": "C3",
            "sarsa_lambda": "C4",
        }
        sns.lineplot(
            data=train_agent_reward_df,
            x="episode",
            y="cumulative reward",
            hue="agent",
            palette=palette,
        )
        sns.lineplot(
            data=train_hist_reward_df,
            x="episode",
            y="cumulative reward",
            hue="agent",
            palette=palette,
        )
        plt.title("Cumulative Reward (Evaluation Set)")
        plt.grid()
        plt.savefig("cumulative.png")

    def plot_reward_distribution(self):
        """Plot the distribution of rewards on the evaluation set.
        """
        percentiles = np.linspace(0, 100, 101)

        rewards_agent, rewards_hist = [], []
        for eval_dict in self.eval_results:
            agent = eval_dict["agent_type"]
            rewards_agent_array = [r[0] for r in eval_dict["rewards_agent"]]
            agent_percentiles = np.percentile(rewards_agent_array, q=percentiles)
            rewards_hist_array = [r[0] for r in eval_dict["rewards_hist"]]
            hist_percentiles = np.percentile(rewards_hist_array, q=percentiles)
            rewards_agent.extend(
                [
                    [agent, percentiles[p], agent_percentiles[p]]
                    for p in range(percentiles.shape[0])
                ]
            )
            rewards_hist.extend(
                [
                    ["historical", percentiles[p], hist_percentiles[p]]
                    for p in range(percentiles.shape[0])
                ]
            )
        rewards_agent_df = pd.DataFrame(
            rewards_agent, columns=["agent", "percentile", "reward"]
        )

        rewards_hist_df = pd.DataFrame(
            rewards_hist, columns=["agent", "percentile", "reward"]
        )

        palette = {
            "historical": "C0",
            "q_learner": "C1",
            "double_q_learner": "C2",
            "sarsa": "C3",
            "sarsa_lambda": "C4",
        }
        sns.lineplot(
            data=rewards_agent_df,
            x="percentile",
            y="reward",
            hue="agent",
            palette=palette,
        )
        sns.lineplot(
            data=rewards_hist_df,
            x="percentile",
            y="reward",
            hue="agent",
            palette=palette,
        )
        plt.title("Reward Percentiles (Evaluation Set)")
        plt.grid()
        plt.savefig("percentiles.png")
