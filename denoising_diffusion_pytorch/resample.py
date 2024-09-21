from abc import ABC, abstractmethod

import numpy as np
import torch as th

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 오버플로우를 피하기 위해 최대값을 빼줍니다.
    return exp_x / exp_x.sum()

def dramatic(p):
    exponent = 2  # 더 큰 값으로 설정하면 차이가 더 커집니다.
    dramatic_p = p ** exponent
    dramatic_p /= dramatic_p.sum()  # 정규화하여 합이 1이 되도록 만듭니다.
    return dramatic_p

def create_named_schedule_sampler(name, diffuse_time_step, total_epochs=None, init_bias=0.4, final_bias=0.7):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffuse_time_step)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffuse_time_step)
    elif name == "train-step":
        return TrainStepScheduler(diffuse_time_step, total_epochs, initial_bias=init_bias, final_bias=final_bias)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")

class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights

class TrainStepScheduler(ScheduleSampler):
    def __init__(self, max_timestep, total_epochs,initial_bias=0.4, final_bias=0.7):
        self.max_timestep = max_timestep
        self.initial_bias = initial_bias
        self.final_bias = final_bias
        self.current_epoch = 0
        self.total_epochs = total_epochs

    def set_epoch(self, current_epoch):
        self.current_epoch = current_epoch

    def weights(self):
        bias = np.linspace(self.initial_bias, self.final_bias, self.total_epochs)[self.current_epoch]
        weights = np.linspace(1 - bias, bias, self.max_timestep)
        weights = dramatic(weights)
        return weights
    

class UniformSampler(ScheduleSampler):
    def __init__(self, diffuse_time_step):
        self._weights = np.ones([diffuse_time_step])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        self.update_with_all_losses(local_ts, local_losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffuse_time_step, history_per_term=10, uniform_prob=0.001):
        self.diffuse_time_step = diffuse_time_step
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffuse_time_step, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffuse_time_step], dtype=np.int64)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffuse_time_step], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss.sum()
            else:
                self._loss_history[t, self._loss_counts[t]] = loss.sum()
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()
