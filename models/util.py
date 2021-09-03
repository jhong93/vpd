import random
import numpy as np


def batch_mulitplexer(data_loaders):
    """Drain all loaders in a weighted way"""

    data_iterators = []
    for a, b in data_loaders:
        data_iterators.append({'name': a, 'n': len(b), 'it': iter(b)})
    while len(data_iterators) > 0:
        i = random.choices(
            list(range(len(data_iterators))), k=1,
            weights=[x['n'] for x in data_iterators])[0]

        sel = data_iterators[i]
        try:
            yield sel['name'], next(sel['it'])
        except StopIteration:
            raise Exception('Uh oh... something went horribly wrong! :(')
        sel['n'] -= 1
        if sel['n'] == 0:
            data_iterators.pop(i)


def batch_zipper(data_loaders):
    """Drain all loaders simultaneously"""

    data_iterators = []
    for a, b in data_loaders:
        data_iterators.append((a, len(b), iter(b)))
    num_batches = max(len(b) for a, b in data_loaders)

    skip_idxs = {}
    for a, b in data_loaders:
        deficit = num_batches - len(b)
        if deficit > 0:
            skip_idxs[a] = set(np.random.choice(
                np.arange(num_batches), deficit, replace=False).tolist())

    for i in range(num_batches):
        batch = []
        for a, b, c in data_iterators:
            if a in skip_idxs and i in skip_idxs[a]:
                continue
            batch.append((a, next(c)))
        yield batch


def step(optimizer, scaler, loss):
    if scaler is None:
        loss.backward()
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    optimizer.zero_grad()
