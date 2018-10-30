import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as c_layers
from mlagents.trainers.models import LearningModel


class BehavioralCloningCustomModel(LearningModel):
    def __init__(self, brain, h_size=128, lr=1e-4, n_layers=2, m_size=128,
                 normalize=False, use_recurrent=False, scope='PPO', seed=0, epsilon=0.2, beta=1e-3, max_step=5e6):
        with tf.variable_scope(scope):
            LearningModel.__init__(self, m_size, normalize, use_recurrent, brain, seed)
            num_streams = 1
            self.last_reward, self.new_reward, self.update_reward = self.create_reward_encoder()
            hidden_streams = self.create_observation_streams(num_streams, h_size, n_layers)
            hidden = hidden_streams[0]

            # a implementação do ppo não usa dropout, logo iremos comentar essa parte do BC
            # self.dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name="dropout_rate")
            # hidden_reg = tf.layers.dropout(hidden, self.dropout_rate)

            if self.use_recurrent:
                tf.Variable(self.m_size, name="memory_size", trainable=False, dtype=tf.int32)
                self.memory_in = tf.placeholder(shape=[None, self.m_size], dtype=tf.float32, name='recurrent_in')
                hidden, self.memory_out = self.create_recurrent_encoder(hidden, self.memory_in,
                                                                            self.sequence_length)
                self.memory_out = tf.identity(self.memory_out, name='recurrent_out')

            if brain.vector_action_space_type == "discrete":
                policy_branches = []
                for size in self.act_size:
                    policy_branches.append(tf.layers.dense(hidden, size, activation=None, use_bias=False,
                                                kernel_initializer=c_layers.variance_scaling_initializer(factor=0.01)))

                self.action_probs = tf.concat([tf.nn.softmax(branch) for branch in policy_branches], axis=1, name="action_probs_imitation")

                # add Icaro
                self.all_log_probs = tf.concat([branch for branch in policy_branches], axis=1, name="action_probs")
                self.action_masks = tf.placeholder(shape=[None, sum(self.act_size)], dtype=tf.float32, name="action_masks")
                output, normalized_logits = self.create_discrete_action_masking_layer(
                                                                    self.all_log_probs, self.action_masks, self.act_size)
                # output, _ = self.create_discrete_action_masking_layer(tf.concat(policy_branches, axis=1), self.action_masks, self.act_size)
                self.output = tf.identity(output, name="action")

                value = tf.layers.dense(hidden, 1, activation=None)
                self.value = tf.identity(value, name="value_estimate")

                self.action_holder = tf.placeholder(shape=[None, len(policy_branches)], dtype=tf.int32, name="action_holder")
                self.selected_actions = tf.concat([
                    tf.one_hot(self.action_holder[:, i], self.act_size[i]) for i in range(len(self.act_size))], axis=1)
                self.all_old_log_probs = tf.placeholder(shape=[None, sum(self.act_size)], dtype=tf.float32, name='old_probabilities')
                _, old_normalized_logits = self.create_discrete_action_masking_layer(
                                                                self.all_old_log_probs, self.action_masks, self.act_size)
                # --------------------------------------------------------------

                self.sample_action = tf.cast(self.output, tf.int32)
                self.true_action = tf.placeholder(shape=[None, len(policy_branches)], dtype=tf.int32, name="teacher_action")
                self.action_oh = tf.concat([
                    tf.one_hot(self.true_action[:, i], self.act_size[i]) for i in range(len(self.act_size))], axis=1)
                self.loss = tf.reduce_sum(-tf.log(self.action_probs + 1e-10) * self.action_oh)
                self.action_percent = tf.reduce_mean(tf.cast(
                    tf.equal(tf.cast(tf.argmax(self.action_probs, axis=1), tf.int32), self.sample_action), tf.float32))

                # codigo adicionado
                action_idx = [0] + list(np.cumsum(self.act_size))

                self.entropy = tf.reduce_sum((tf.stack([
                    tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=tf.nn.softmax(self.all_log_probs[:, action_idx[i]:action_idx[i + 1]]),
                        logits=self.all_log_probs[:, action_idx[i]:action_idx[i + 1]])
                    for i in range(len(self.act_size))], axis=1)), axis=1)

                self.log_probs = tf.reduce_sum((tf.stack([-tf.nn.softmax_cross_entropy_with_logits_v2(
                                                        labels=self.selected_actions[:, action_idx[i]:action_idx[i + 1]],
                                                        logits=normalized_logits[:, action_idx[i]:action_idx[i + 1]]
                                                    )for i in range(len(self.act_size))], axis=1)), axis=1, keepdims=True)

                self.old_log_probs = tf.reduce_sum((tf.stack([-tf.nn.softmax_cross_entropy_with_logits_v2(
                                                        labels=self.selected_actions[:, action_idx[i]:action_idx[i + 1]],
                                                        logits=old_normalized_logits[:, action_idx[i]:action_idx[i + 1]]
                                                    )for i in range(len(self.act_size))], axis=1)), axis=1, keepdims=True)
                #----------------
            else:
                self.policy = tf.layers.dense(hidden, self.act_size[0], activation=None, use_bias=False, name='pre_action',
                                              kernel_initializer=c_layers.variance_scaling_initializer(factor=0.01))
                self.clipped_sample_action = tf.clip_by_value(self.policy, -1, 1)
                self.sample_action = tf.identity(self.clipped_sample_action, name="action")
                self.true_action = tf.placeholder(shape=[None, self.act_size[0]], dtype=tf.float32, name="teacher_action")
                self.clipped_true_action = tf.clip_by_value(self.true_action, -1, 1)
                self.loss = tf.reduce_sum(tf.squared_difference(self.clipped_true_action, self.sample_action))

            self.create_ppo_optimizer(self.log_probs, self.old_log_probs, self.value, self.entropy, beta, epsilon, lr, max_step)

            optimizer1 = tf.train.AdamOptimizer(learning_rate=lr)
            self.update = optimizer1.minimize(self.loss)

    @staticmethod
    def create_reward_encoder():
        """Creates TF ops to track and increment recent average cumulative reward."""
        last_reward = tf.Variable(0, name="last_reward", trainable=False, dtype=tf.float32)
        new_reward = tf.placeholder(shape=[], dtype=tf.float32, name='new_reward')
        update_reward = tf.assign(last_reward, new_reward)
        return last_reward, new_reward, update_reward

    def create_ppo_optimizer(self, probs, old_probs, value, entropy, beta, epsilon, lr, max_step):
        """
        Creates training-specific Tensorflow ops for PPO models.
        :param probs: Current policy probabilities
        :param old_probs: Past policy probabilities
        :param value: Current value estimate
        :param beta: Entropy regularization strength
        :param entropy: Current policy entropy
        :param epsilon: Value for policy-divergence threshold
        :param lr: Learning rate
        :param max_step: Total number of training steps.
        """
        self.returns_holder = tf.placeholder(shape=[None], dtype=tf.float32, name='discounted_rewards')
        self.advantage = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='advantages')

        self.learning_rate = tf.train.polynomial_decay(lr, self.global_step, max_step, 1e-10, power=1.0)

        self.old_value = tf.placeholder(shape=[None], dtype=tf.float32, name='old_value_estimates')

        decay_epsilon = tf.train.polynomial_decay(epsilon, self.global_step, max_step, 0.1, power=1.0)
        decay_beta = tf.train.polynomial_decay(beta, self.global_step, max_step, 1e-5, power=1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        clipped_value_estimate = self.old_value + tf.clip_by_value(tf.reduce_sum(value, axis=1) - self.old_value,
                                                                   - decay_epsilon, decay_epsilon)

        v_opt_a = tf.squared_difference(self.returns_holder, tf.reduce_sum(value, axis=1))
        v_opt_b = tf.squared_difference(self.returns_holder, clipped_value_estimate)
        self.value_loss = tf.reduce_mean(tf.dynamic_partition(tf.maximum(v_opt_a, v_opt_b), self.mask, 2)[1])

        # Here we calculate PPO policy loss. In continuous control this is done independently for each action gaussian
        # and then averaged together. This provides significantly better performance than treating the probability
        # as an average of probabilities, or as a joint probability.
        r_theta = tf.exp(probs - old_probs)
        p_opt_a = r_theta * self.advantage
        p_opt_b = tf.clip_by_value(r_theta, 1.0 - decay_epsilon, 1.0 + decay_epsilon) * self.advantage
        self.policy_loss = -tf.reduce_mean(tf.dynamic_partition(tf.minimum(p_opt_a, p_opt_b), self.mask, 2)[1])

        # self.zero_const = tf.Variable(0.0, name="zero_const", dtype=tf.float32)
        # def f2():
        #     return self.zero_const
        #
        # def f1():
        #     return self.loss
        # self.is_enabled_imitation_loss = tf.Variable(True, name="is_enabled_imitation_loss", dtype=tf.bool)
        # cond_loss = tf.cond(self.is_enabled_imitation_loss, f1, f2, name="cond_loss")

        self.loss_total = self.policy_loss + 0.5 * self.value_loss - decay_beta * tf.reduce_mean(
            tf.dynamic_partition(entropy, self.mask, 2)[1])

        self.update_batch = optimizer.minimize(self.loss_total)