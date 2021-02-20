"""
- does curr_frames = frames[idxes] actually work
"""

import torch
import torch.nn as nn
import numpy as np


def get_counts(model: nn.Module,
               frames: torch.Tensor,
               strides: list,
               batch_size: int,
               threshold: float,
               within_period_threshold: float,
               constant_speed: bool = False,
               median_filter:bool = False,
               fully_periodic = False):

    seq_len = len(frames)
    raw_scores_list = []
    scores = []
    within_period_scores_list = []

    if fully_periodic:
        within_period_threshold = 0.0

    frames = model.preprocess(imgs = frames)

    for stride in strides:
        num_batches = int(np.ceil(seq_len / model.num_frames / stride / batch_size))
        raw_scores_per_stride = []
        within_period_score_stride = []
        for batch_idx in range(num_batches):
            idxes = torch.arange(start = batch_idx * batch_size * model.num_frames * stride,
                                 end = (batch_idx + 1) * batch_size * model.num_frames * stride,
                                 step = stride)
            idxes = torch.clamp(idxes, 0, seq_len - 1)
            curr_frames = frames[idxes]
            curr_frames = torch.reshape(
                curr_frames,
                [batch_size, model.num_frames, 3, model.image_size, model.image_size])
            raw_scores, within_period_scores, _ = model(curr_frames)
            raw_scores_per_stride.append(np.reshape(raw_scores.numpy(),
                                                    [-1, model.num_frames // 2]))
            within_period_score_stride.append(np.reshape(within_period_scores.numpy(),
                                                         [-1, 1]))
        raw_scores_per_stride = np.concatenate(raw_scores_per_stride, axis = 0)
        raw_scores_list.append(raw_scores_per_stride)
        within_period_score_stride = np.concatenate(
            within_period_score_stride, axis = 0)
        pred_score, within_period_score_stride = get_score(
            raw_scores_per_stride, within_period_score_stride)
        scores.append(pred_score)
        within_period_scores_list.append(within_period_score_stride)


if __name__ == "__main__":
    from RepNet.RepNetModel import RepNetPeriodEstimator

    model = RepNetPeriodEstimator()
    imgs = torch.randn(66, 3, 224, 224)
    THRESHOLD = 0.2
    WITHIN_PERIOD_THRESHOLD = 0.5
    CONSTANT_SPEED = False
    MEDIAN_FILTER = True
    FULLY_PERIODIC = False

    get_counts(
        model,
        imgs,
        # strides = [1, 2, 3, 4],
        strides = [3],
        batch_size = 20,
        threshold = THRESHOLD,
        within_period_threshold = WITHIN_PERIOD_THRESHOLD,
        constant_speed = CONSTANT_SPEED,
        median_filter = MEDIAN_FILTER,
        fully_periodic = FULLY_PERIODIC)
