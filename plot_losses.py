#!/usr/bin/env python3

import os
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from util.io import load_json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('-p', '--pause', type=int, default=60)
    parser.add_argument('-e', '--max_epoch', type=int)
    return parser.parse_args()


def collect_dataset_losses(losses, key):
    datasets = defaultdict(list)
    for l in losses:
        if key in l:
            for d, v in l[key]:
                datasets[d].append((l['epoch'], v))
    return datasets


def smooth(x, window):
    result = []
    for i in range(len(x)):
        result.append(np.mean(x[max(i - window, 0): i + 1 + window]))
    return result


def main(model_dir, pause, max_epoch):
    losses = load_json(os.path.join(model_dir, 'loss.json'))

    best_val_loss = float('inf')
    best_epoch = None
    for l in losses:
        if l['val'] < best_val_loss:
            best_epoch = l['epoch']
            best_val_loss = l['val']

    print('Best epoch:', best_epoch)
    print('Best val loss:', best_val_loss)

    print()
    for i in range(3, 11, 2):
        print('Val loss (smooth: {}):'.format(i),
              min(smooth([l['val'] for l in losses], i)))

    dataset_train_losses = collect_dataset_losses(losses, 'dataset_train')
    dataset_val_losses = collect_dataset_losses(losses, 'dataset_val')
    has_subplots = max(len(dataset_train_losses), len(dataset_val_losses)) > 1

    if has_subplots:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 8))
        main_ax, sub_ax = axes
    else:
        fig = plt.figure(figsize=(7, 4))
        main_ax, sub_ax = plt.gca(), None
        axes = [main_ax]

    timer = fig.canvas.new_timer(interval=60000 * pause)
    timer.add_callback(lambda: plt.close())
    epochs, val_losses, train_losses = zip(
        *[(l['epoch'], l['val'], l['train']) for l in losses
          if max_epoch is None or l['epoch'] <= max_epoch])
    main_ax.plot(epochs, train_losses,
                 label='train', lw=1, alpha=0.5)
    main_ax.plot(epochs, val_losses, label='val', lw=1, alpha=0.5)
    main_ax.plot(epochs, smooth(train_losses, 3),
                 label='train (smooth +/-3)', lw=2, linestyle=':')
    main_ax.plot(epochs, smooth(val_losses, 3),
                 label='val (smooth +/-3)', lw=2, linestyle=':')
    main_ax.hlines(best_val_loss, min(epochs), max(epochs), colors='grey',
                   alpha=0.5)
    main_ax.set_title('Losses: {}'.format(model_dir))
    main_ax.set_ylim(
        max(0, min(train_losses + val_losses)),
        min(np.nanmedian(val_losses) * 2, max(val_losses)))

    if sub_ax is not None:
        min_dataset_train = float('inf')
        for dataset_name, train_losses in sorted(dataset_train_losses.items()):
            x, y = zip(*train_losses)
            sub_ax.plot(x, y, linestyle=':',
                        label='train ({})'.format(dataset_name))
            min_dataset_train = min(min(y), min_dataset_train)

        max_median_val = 0
        for dataset_name, val_losses in sorted(dataset_val_losses.items()):
            x, y = zip(*val_losses)
            sub_ax.plot(x, y, label='val ({})'.format(dataset_name))
            max_median_val = max(max_median_val, np.nanmedian(y))
        sub_ax.set_title('Loss breakdown by dataset')
        sub_ax.set_ylim(max(0, min_dataset_train) * 0.8, max_median_val * 2)

    for ax in axes:
        ax.set_xlabel('epoch')
        ax.set_ylabel('avg_loss')
        ax.legend(loc='upper right')

        if max_epoch is not None:
            ax.set_xlim(0, max_epoch)

    plt.tight_layout()
    timer.start()
    plt.show()
    print('Exited')


if __name__ == '__main__':
    main(**vars(get_args()))
