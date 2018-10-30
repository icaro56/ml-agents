# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (Imitation)
# Contains an implementation of Behavioral Cloning Algorithm

import logging
import os

import numpy as np
import tensorflow as tf
import time
from collections import deque

from mlagents.envs import AllBrainInfo, BrainInfo
from mlagents.trainers.bc_custom.policy import BCCustomPolicy
from mlagents.trainers.buffer import Buffer
from mlagents.trainers.trainer import UnityTrainerException, Trainer

logger = logging.getLogger("mlagents.envs")


class BehavioralCloningCustomTrainer(Trainer):
    """The ImitationTrainer is an implementation of the imitation learning."""

    def __init__(self, sess, brain, trainer_parameters, training, seed, run_id):
        """
        Responsible for collecting experiences and training PPO model.
        :param sess: Tensorflow session.
        :param  trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        """
        super(BehavioralCloningCustomTrainer, self).__init__(sess, brain, trainer_parameters, training, run_id)

        self.param_keys = ['brain_to_imitate', 'batch_size', 'time_horizon',
                           'graph_scope', 'summary_freq', 'max_steps',
                           'batches_per_epoch', 'use_recurrent',
                           'hidden_units','learning_rate', 'num_layers',
                           'sequence_length', 'memory_size', 'epsilon', 'beta']

        for k in self.param_keys:
            if k not in trainer_parameters:
                raise UnityTrainerException("The hyperparameter {0} could not be found for the Imitation trainer of "
                                            "brain {1}.".format(k, brain.brain_name))

        self.step = 0
        self.policy = BCCustomPolicy(seed, brain, trainer_parameters, sess)
        self.brain_name = brain.brain_name
        self.brain_to_imitate = trainer_parameters['brain_to_imitate']
        self.batches_per_epoch = trainer_parameters['batches_per_epoch']
        self.n_sequences = max(int(trainer_parameters['batch_size'] / self.policy.sequence_length), 1)
        self.cumulative_rewards = {}
        # usado no curriculo learning
        self._reward_buffer = deque(maxlen=1000)
        self.episode_steps = {}
        self.stats = {'losses': [], 'episode_length': [], 'cumulative_reward': [], 'value_estimate': [], 'entropy': [],
                      'value_loss': [], 'policy_loss': [], 'learning_rate': []}

        self.training_buffer = Buffer()
        self.training_buffer_ppo = Buffer()
        self.summary_path = trainer_parameters['summary_path']
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)

        self.summary_writer = tf.summary.FileWriter(self.summary_path)
        #criando arquivo
        self.actionProbList = []
        self.trueActionList = []
        self.use_curiosity = False

    def createTempFolder(self, foldername):
        try:
            if not os.path.exists(foldername):
                os.makedirs(foldername)
        except Exception:
            print("Arquivo temporario nao foi criado: " + foldername)

    def saveTemps(self):

        timestr = time.strftime("%Y%m%d-%H%M%S")
        folderName1 = './tmp/' + timestr + "/"
        self.createTempFolder(folderName1)

        arq = open(folderName1 + 'actionProb_' + self.brain_name + ".txt", 'w')
        arq.writelines(self.actionProbList)
        arq.close()

        arq = open(folderName1 + 'trueAction_' + self.brain_name + ".txt", 'w')
        arq.writelines(self.trueActionList)
        arq.close()

    def addStepInTempList(self, gStep):
        self.actionProbList.append("Step: " + str(gStep) + "\n");
        self.trueActionList.append("Step: " + str(gStep) + "\n");

    def __str__(self):
        return '''Hyperparameters for the Imitation Trainer of brain {0}: \n{1}'''.format(
            self.brain_name, '\n'.join(['\t{0}:\t{1}'.format(x, self.trainer_parameters[x]) for x in self.param_keys]))

    @property
    def parameters(self):
        """
        Returns the trainer parameters of the trainer.
        """
        return self.trainer_parameters

    @property
    def get_max_steps(self):
        """
        Returns the maximum number of steps. Is used to know when the trainer should be stopped.
        :return: The maximum number of steps of the trainer
        """
        return float(self.trainer_parameters['max_steps'])

    @property
    def get_step(self):
        """
        Returns the number of steps the trainer has performed
        :return: the step count of the trainer
        """
        # return self.policy.get_current_step()
        return self.step

    @property
    def reward_buffer(self):
        """
        Returns the reward buffer. The reward buffer contains the cumulative
        rewards of the most recent episodes completed by agents using this
        trainer.
        :return: the reward buffer.
        """
        return self._reward_buffer

    @property
    def get_last_reward(self):
        """
        Returns the last reward the trainer has had
        :return: the new last reward
        """
        if len(self.stats['cumulative_reward']) > 0:
            return np.mean(self.stats['cumulative_reward'])
        else:
            return 0

    def increment_step_and_update_last_reward(self):
        """
        Increment the step count of the trainer and Updates the last reward
        """
        if len(self.stats['cumulative_reward']) > 0:
            mean_reward = np.mean(self.stats['cumulative_reward'])
            self.policy.update_reward(mean_reward)
        self.policy.increment_step()
        self.step = self.policy.get_current_step()

    def take_action(self, all_brain_info: AllBrainInfo):
        """
        Decides actions using policy given current brain info.
        :param all_brain_info: AllBrainInfo from environment.
        :return: a tuple containing action, memories, values and an object
        to be passed to add experiences
        """
        if len(all_brain_info[self.brain_name].agents) == 0:
            return [], [], [], None, None

        agent_brain = all_brain_info[self.brain_name]
        run_out = self.policy.evaluate(agent_brain)
        self.stats['value_estimate'].append(run_out['value'].mean())
        self.stats['entropy'].append(run_out['entropy'].mean())
        self.stats['learning_rate'].append(run_out['learning_rate'])
        if self.policy.use_recurrent:
            return run_out['action'], run_out['memory_out'], None, None, None
        else:
            return run_out['action'], None, None, run_out['value'], run_out

    def add_experiences(self, curr_info: AllBrainInfo, next_info: AllBrainInfo, take_action_outputs):
        """
        Adds experiences to each agent's experience history.
        :param curr_info: Current AllBrainInfo (Dictionary of all current brains and corresponding BrainInfo).
        :param next_info: Next AllBrainInfo (Dictionary of all current brains and corresponding BrainInfo).
        :param take_action_outputs: The outputs of the take action method.
        """

        # Used to collect teacher experience into training buffer
        info_teacher = curr_info[self.brain_to_imitate]
        next_info_teacher = next_info[self.brain_to_imitate]

        for agent_id in info_teacher.agents:
            self.training_buffer[agent_id].last_brain_info = info_teacher

        for agent_id in next_info_teacher.agents:
            stored_info_teacher = self.training_buffer[agent_id].last_brain_info
            if stored_info_teacher is None:
                continue
            else:
                idx = stored_info_teacher.agents.index(agent_id)
                next_idx = next_info_teacher.agents.index(agent_id)
                if stored_info_teacher.text_observations[idx] != "":
                    info_teacher_record, info_teacher_reset = \
                        stored_info_teacher.text_observations[idx].lower().split(",")
                    next_info_teacher_record, next_info_teacher_reset = next_info_teacher.text_observations[idx].\
                        lower().split(",")
                    if next_info_teacher_reset == "true":
                        self.training_buffer.reset_update_buffer()
                else:
                    info_teacher_record, next_info_teacher_record = "true", "true"
                if info_teacher_record == "true" and next_info_teacher_record == "true":
                    if not stored_info_teacher.local_done[idx]:
                        for i in range(self.policy.vis_obs_size):
                            self.training_buffer[agent_id]['visual_obs%d' % i]\
                                .append(stored_info_teacher.visual_observations[i][idx])
                        if self.policy.use_vec_obs:
                            self.training_buffer[agent_id]['vector_obs']\
                                .append(stored_info_teacher.vector_observations[idx])
                        if self.policy.use_recurrent:
                            if stored_info_teacher.memories.shape[1] == 0:
                                stored_info_teacher.memories = np.zeros((len(stored_info_teacher.agents),
                                                                         self.policy.m_size))
                            self.training_buffer[agent_id]['memory'].append(stored_info_teacher.memories[idx])
                        self.training_buffer[agent_id]['actions'].append(next_info_teacher.
                                                                         previous_vector_actions[next_idx])
        info_student = curr_info[self.brain_name]
        next_info_student = next_info[self.brain_name]
        for agent_id in info_student.agents:
            self.training_buffer_ppo[agent_id].last_brain_info = info_student
            self.training_buffer_ppo[agent_id].last_take_action_outputs = take_action_outputs

        # apenas modulo curiosity usa isso. Logo nao preciso adicionar isso aqui
        # if info_student.agents != next_info_student.agents:
        #     curr_to_use = self.construct_curr_info(next_info_student)
        # else:
        #     curr_to_use = info_student
        # intrinsic_rewards = self.policy.get_intrinsic_rewards(curr_to_use, next_info)

        for agent_id in next_info_student.agents:
            stored_info = self.training_buffer_ppo[agent_id].last_brain_info
            stored_take_action_outputs = self.training_buffer_ppo[agent_id].last_take_action_outputs
            if stored_info is not None:
                idx = stored_info.agents.index(agent_id)
                next_idx = next_info_student.agents.index(agent_id)
                if not stored_info.local_done[idx]:
                    for i, _ in enumerate(stored_info.visual_observations):
                        self.training_buffer_ppo[agent_id]['visual_obs%d' % i].append(
                            stored_info.visual_observations[i][idx])
                        self.training_buffer_ppo[agent_id]['next_visual_obs%d' % i].append(
                            next_info_student.visual_observations[i][next_idx])
                    if self.policy.use_vec_obs:
                        self.training_buffer_ppo[agent_id]['vector_obs'].append(stored_info.vector_observations[idx])
                        self.training_buffer_ppo[agent_id]['next_vector_in'].append(
                            next_info_student.vector_observations[next_idx])
                    if self.policy.use_recurrent:
                        if stored_info.memories.shape[1] == 0:
                            stored_info.memories = np.zeros((len(stored_info.agents), self.policy.m_size))
                        self.training_buffer_ppo[agent_id]['memory'].append(stored_info.memories[idx])
                    actions = stored_take_action_outputs['action']

                    # não trabalhamos com ação continua
                    if not self.policy.use_continuous_act:
                        self.training_buffer_ppo[agent_id]['action_mask'].append(stored_info.action_masks[idx])

                    a_dist = stored_take_action_outputs['log_probs']
                    value = stored_take_action_outputs['value']
                    self.training_buffer_ppo[agent_id]['actions'].append(actions[idx])
                    self.training_buffer_ppo[agent_id]['prev_action'].append(stored_info.previous_vector_actions[idx])
                    self.training_buffer_ppo[agent_id]['masks'].append(1.0)

                    if not self.use_curiosity:
                        self.training_buffer_ppo[agent_id]['rewards'].append(next_info_student.rewards[next_idx])

                    self.training_buffer_ppo[agent_id]['action_probs'].append(a_dist[idx])
                    self.training_buffer_ppo[agent_id]['value_estimates'].append(value[idx][0])

                    if agent_id not in self.cumulative_rewards:
                        self.cumulative_rewards[agent_id] = 0
                    self.cumulative_rewards[agent_id] += next_info_student.rewards[next_idx]
                if not next_info_student.local_done[next_idx]:
                    if agent_id not in self.episode_steps:
                        self.episode_steps[agent_id] = 0
                    self.episode_steps[agent_id] += 1

    def process_experiences(self, current_info: AllBrainInfo, next_info: AllBrainInfo):
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param current_info: Current AllBrainInfo
        :param next_info: Next AllBrainInfo
        """
        info_teacher = next_info[self.brain_to_imitate]
        for l in range(len(info_teacher.agents)):
            teacher_action_list = len(self.training_buffer[info_teacher.agents[l]]['actions'])
            horizon_reached = teacher_action_list > self.trainer_parameters['time_horizon']
            teacher_filled = len(self.training_buffer[info_teacher.agents[l]]['actions']) > 0
            if ((info_teacher.local_done[l] or horizon_reached) and teacher_filled):
                agent_id = info_teacher.agents[l]
                self.training_buffer.append_update_buffer(agent_id, batch_size=None, training_length=self.policy.sequence_length)
                self.training_buffer[agent_id].reset_agent()

        info_student = next_info[self.brain_name]
        for l in range(len(info_student.agents)):
            agent_actions = self.training_buffer_ppo[info_student.agents[l]]['actions']
            if ((info_student.local_done[l] or len(agent_actions) > self.trainer_parameters['time_horizon']) and len(agent_actions) > 0):
                agent_id = info_student.agents[l]
                if info_student.local_done[l] and not info_student.max_reached[l]:
                    value_next = 0.0
                else:
                    if info_student.max_reached[l]:
                        bootstrapping_info = self.training_buffer_ppo[agent_id].last_brain_info
                        idx = bootstrapping_info.agents.index(agent_id)
                    else:
                        bootstrapping_info = info_student
                        idx = l
                    value_next = self.policy.get_value_estimate(bootstrapping_info, idx)
                self.training_buffer_ppo[agent_id]['advantages'].set(
                    get_gae(
                        rewards=self.training_buffer_ppo[agent_id]['rewards'].get_batch(),
                        value_estimates=self.training_buffer_ppo[agent_id]['value_estimates'].get_batch(),
                        value_next=value_next,
                        gamma=self.trainer_parameters['gamma'],
                        lambd=self.trainer_parameters['lambd']))
                self.training_buffer_ppo[agent_id]['discounted_returns'].set(
                    self.training_buffer_ppo[agent_id]['advantages'].get_batch()
                    + self.training_buffer_ppo[agent_id]['value_estimates'].get_batch())

                self.training_buffer_ppo.append_update_buffer(agent_id, batch_size=None, training_length=self.policy.sequence_length)
                self.training_buffer_ppo[agent_id].reset_agent()

                if info_student.local_done[l]:
                    self.stats['cumulative_reward'].append(
                        self.cumulative_rewards.get(agent_id, 0))
                    self.reward_buffer.appendleft(self.cumulative_rewards.get(agent_id, 0))
                    self.stats['episode_length'].append(
                        self.episode_steps.get(agent_id, 0))
                    self.cumulative_rewards[agent_id] = 0
                    self.episode_steps[agent_id] = 0

    def end_episode(self):
        """
        A signal that the Episode has ended. The buffer must be reset. 
        Get only called when the academy resets.
        """
        self.training_buffer.reset_all()
        self.training_buffer_ppo.reset_all()
        for agent_id in self.cumulative_rewards:
            self.cumulative_rewards[agent_id] = 0
        for agent_id in self.episode_steps:
            self.episode_steps[agent_id] = 0

    def is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to whether or not update_model() can be run
        """
        cond_1 = len(self.training_buffer.update_buffer['actions']) > self.n_sequences
        cond_2 = len(self.training_buffer_ppo.update_buffer['actions']) > self.n_sequences

        return cond_1 or cond_2

    def is_ready_update_bc(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to whether or not update_model() can be run
        """
        cond_1 = len(self.training_buffer.update_buffer['actions']) > self.n_sequences

        return cond_1

    def is_ready_update_ppo(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to whether or not update_model() can be run
        """
        cond_2 = len(self.training_buffer_ppo.update_buffer['actions']) > self.n_sequences

        return cond_2

    def update_policy(self):
        """
        Updates the policy.
        """
        self.n_sequences = max(int(self.trainer_parameters['batch_size'] / self.policy.sequence_length), 1)
        num_epoch = self.trainer_parameters['num_epoch']

        if self.is_ready_update_bc():
            batch_losses = []
            for k in range(num_epoch):
                self.training_buffer.update_buffer.shuffle()
                buffer = self.training_buffer.update_buffer
                num_batches = min(len(self.training_buffer.update_buffer['actions']) // self.n_sequences,
                                  self.batches_per_epoch)
                for i in range(num_batches):
                    start = i * self.n_sequences
                    end = (i + 1) * self.n_sequences
                    mini_batch = buffer.make_mini_batch(start, end)
                    run_out = self.policy.update(mini_batch, self.n_sequences)

                    self.actionProbList.append(
                        "epoca: " + str(i) + "\n" + np.array2string(run_out['action_probs'], precision=2) + "\n")
                    self.trueActionList.append("epoca: " + str(i) + "\n" + str(run_out['action_oh']) + "\n")

                    loss = run_out['loss']
                    batch_losses.append(loss)

            if len(batch_losses) > 0:
                self.stats['losses'].append(np.mean(batch_losses))
            else:
                self.stats['losses'].append(0)
        if self.is_ready_update_ppo():
            value_total, policy_total, forward_total, inverse_total = [], [], [], []
            advantages = self.training_buffer_ppo.update_buffer['advantages'].get_batch()
            self.training_buffer_ppo.update_buffer['advantages'].set(
                (advantages - advantages.mean()) / (advantages.std() + 1e-10))

            for k in range(num_epoch):
                self.training_buffer_ppo.update_buffer.shuffle()
                buffer_ppo = self.training_buffer_ppo.update_buffer
                num_batches_ppo = min(len(self.training_buffer_ppo.update_buffer['actions']) // self.n_sequences,
                                      self.batches_per_epoch)

                for i in range(num_batches_ppo):
                    start = i * self.n_sequences
                    end = (i + 1) * self.n_sequences
                    mini_batch = buffer_ppo.make_mini_batch(start, end)
                    run_out = self.policy.update_ppo(mini_batch, self.n_sequences)

                    value_total.append(run_out['value_loss'])
                    policy_total.append(np.abs(run_out['policy_loss']))

            self.stats['value_loss'].append(np.mean(value_total))
            self.stats['policy_loss'].append(np.mean(policy_total))
            self.training_buffer_ppo.reset_update_buffer()

def discount_rewards(r, gamma=0.99, value_next=0.0):
    """
    Computes discounted sum of future rewards for use in updating value estimate.
    :param r: List of rewards.
    :param gamma: Discount factor.
    :param value_next: T+1 value estimate for returns calculation.
    :return: discounted sum of future rewards as list.
    """
    discounted_r = np.zeros_like(r)
    running_add = value_next
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def get_gae(rewards, value_estimates, value_next=0.0, gamma=0.99, lambd=0.95):
    """
    Computes generalized advantage estimate for use in updating policy.
    :param rewards: list of rewards for time-steps t to T.
    :param value_next: Value estimate for time-step T+1.
    :param value_estimates: list of value estimates for time-steps t to T.
    :param gamma: Discount factor.
    :param lambd: GAE weighing factor.
    :return: list of advantage estimates for time-steps t to T.
    """
    value_estimates = np.asarray(value_estimates.tolist() + [value_next])
    delta_t = rewards + gamma * value_estimates[1:] - value_estimates[:-1]
    advantage = discount_rewards(r=delta_t, gamma=gamma * lambd)
    return advantage
