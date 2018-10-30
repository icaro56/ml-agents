import logging

import numpy as np
from mlagents.trainers.bc_custom.models import BehavioralCloningCustomModel
from mlagents.trainers.policy import Policy

logger = logging.getLogger("mlagents.trainers")


class BCCustomPolicy(Policy):
    def __init__(self, seed, brain, trainer_parameters, sess):
        """
        :param seed: Random seed.
        :param brain: Assigned Brain object.
        :param trainer_parameters: Defined training parameters.
        :param sess: TensorFlow session.
        """
        super().__init__(seed, brain, trainer_parameters, sess)

        self.model = BehavioralCloningCustomModel(
            h_size=int(trainer_parameters['hidden_units']),
            lr=float(trainer_parameters['learning_rate']),
            n_layers=int(trainer_parameters['num_layers']),
            m_size=self.m_size,
            normalize=False,
            use_recurrent=trainer_parameters['use_recurrent'],
            brain=brain,
            scope=self.variable_scope,
            seed=seed,
            epsilon=float(trainer_parameters['epsilon']),
            beta=float(trainer_parameters['beta']),
            max_step=float(trainer_parameters['max_steps'])
        )

        self.inference_dict = {'action': self.model.sample_action,
                               'value': self.model.value,
                               'entropy': self.model.entropy,
                               'log_probs': self.model.all_log_probs,
                               'learning_rate': self.model.learning_rate}

        self.update_dict = {'loss': self.model.loss,
                            'update_batch': self.model.update,
                            'action_probs': self.model.action_probs,
                            'action_oh': self.model.action_oh,
                            }

        self.update_dict_ppo = {
                            'value_loss': self.model.value_loss,
                            'policy_loss': self.model.policy_loss,
                            'update_batch': self.model.update_batch
                            }

        if self.use_recurrent:
            self.inference_dict['memory_out'] = self.model.memory_out

        self.evaluate_rate = 1.0
        self.update_rate = 0.5

    def evaluate(self, brain_info):
        """
        Evaluates policy for the agent experiences provided.
        :param brain_info: BrainInfo input to network.
        :return: Results of evaluation.
        """
        # feed_dict = {self.model.dropout_rate: self.evaluate_rate,
        #              self.model.sequence_length: 1}
        feed_dict = {self.model.sequence_length: 1}

        feed_dict = self._fill_eval_dict(feed_dict, brain_info)
        if self.use_recurrent:
            if brain_info.memories.shape[1] == 0:
                brain_info.memories = self.make_empty_memory(len(brain_info.agents))
            feed_dict[self.model.memory_in] = brain_info.memories
        run_out = self._execute_model(feed_dict, self.inference_dict)
        return run_out

    def get_value_estimate(self, brain_info, idx):
        """
        Generates value estimates for bootstrapping.
        :param brain_info: BrainInfo to be used for bootstrapping.
        :param idx: Index in BrainInfo of agent.
        :return: Value estimate.
        """
        feed_dict = {self.model.batch_size: 1, self.model.sequence_length: 1}
        for i in range(len(brain_info.visual_observations)):
            feed_dict[self.model.visual_in[i]] = [brain_info.visual_observations[i][idx]]
        if self.use_vec_obs:
            feed_dict[self.model.vector_in] = [brain_info.vector_observations[idx]]
        if self.use_recurrent:
            if brain_info.memories.shape[1] == 0:
                brain_info.memories = self.make_empty_memory(len(brain_info.agents))
            feed_dict[self.model.memory_in] = [brain_info.memories[idx]]
        # nao implementamos ações contínuas
        # if not self.use_continuous_act and self.use_recurrent:
        #     feed_dict[self.model.prev_action] = brain_info.previous_vector_actions[idx].reshape(
        #         [-1, len(self.model.act_size)])
        value_estimate = self.sess.run(self.model.value, feed_dict)
        return value_estimate

    def update(self, mini_batch, num_sequences):
        """
        Performs update on model.
        :param mini_batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """

        feed_dict = {
                     self.model.batch_size: num_sequences,
                     self.model.sequence_length: self.sequence_length}
        if self.use_continuous_act:
            feed_dict[self.model.true_action] = mini_batch['actions']. \
                reshape([-1, self.brain.vector_action_space_size[0]])
        else:
            feed_dict[self.model.true_action] = mini_batch['actions'].reshape(
                [-1, len(self.brain.vector_action_space_size)])
            feed_dict[self.model.action_masks] = np.ones(
                (num_sequences, sum(self.brain.vector_action_space_size)))
        if self.use_vec_obs:
            apparent_obs_size = self.brain.vector_observation_space_size * \
                                self.brain.num_stacked_vector_observations
            feed_dict[self.model.vector_in] = mini_batch['vector_obs'] \
                .reshape([-1,apparent_obs_size])
        for i, _ in enumerate(self.model.visual_in):
            visual_obs = mini_batch['visual_obs%d' % i]
            feed_dict[self.model.visual_in[i]] = visual_obs
        if self.use_recurrent:
            feed_dict[self.model.memory_in] = np.zeros([num_sequences, self.m_size])
        run_out = self._execute_model(feed_dict, self.update_dict)
        return run_out

    def update_ppo(self, mini_batch, num_sequences):
        """
        Performs update on model.
        :param mini_batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """

        # feed_dict = {self.model.dropout_rate: self.update_rate,
        #              self.model.batch_size: num_sequences,
        #              self.model.sequence_length: self.sequence_length}

        feed_dict = {self.model.batch_size: num_sequences,
                     self.model.sequence_length: self.sequence_length,

                     # analisar se mantes essa parte abaixo ou divide o feed_dict em 2
                     self.model.mask_input: mini_batch['masks'].flatten(),
                     self.model.returns_holder: mini_batch['discounted_returns'].flatten(),
                     self.model.old_value: mini_batch['value_estimates'].flatten(),
                     self.model.advantage: mini_batch['advantages'].reshape([-1, 1]),
                     self.model.all_old_log_probs: mini_batch['action_probs'].reshape([-1, sum(self.model.act_size)])
                     }

        if self.use_continuous_act:
            feed_dict[self.model.true_action] = mini_batch['actions']. \
                reshape([-1, self.brain.vector_action_space_size[0]])
        else:
            feed_dict[self.model.true_action] = mini_batch['actions'].reshape([-1, len(self.brain.vector_action_space_size)])
            # feed_dict[self.model.action_masks] = np.ones((num_sequences, sum(self.brain.vector_action_space_size)))

            #parte adicionada
            feed_dict[self.model.action_holder] = mini_batch['actions'].reshape([-1, len(self.model.act_size)])
            if self.use_recurrent:
                feed_dict[self.model.prev_action] = mini_batch['prev_action'].reshape([-1, len(self.model.act_size)])
            feed_dict[self.model.action_masks] = mini_batch['action_mask'].reshape([-1, sum(self.brain.vector_action_space_size)])

        if self.use_vec_obs:
            apparent_obs_size = self.brain.vector_observation_space_size * self.brain.num_stacked_vector_observations
            feed_dict[self.model.vector_in] = mini_batch['vector_obs'].reshape([-1,apparent_obs_size])

        for i, _ in enumerate(self.model.visual_in):
            visual_obs = mini_batch['visual_obs%d' % i]
            feed_dict[self.model.visual_in[i]] = visual_obs
        if self.use_recurrent:
            feed_dict[self.model.memory_in] = np.zeros([num_sequences, self.m_size])

        run_out = self._execute_model(feed_dict, self.update_dict_ppo)
        return run_out

    def get_last_reward(self):
        """
        Returns the last reward the trainer has had
        :return: the new last reward
        """
        return self.sess.run(self.model.last_reward)

    def update_reward(self, new_reward):
        """
        Updates reward value for policy.
        :param new_reward: New reward to save.
        """
        self.sess.run(self.model.update_reward,
                      feed_dict={self.model.new_reward: new_reward})
