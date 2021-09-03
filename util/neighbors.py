from collections import Counter
import heapq
import numpy as np
from multiprocessing import Pool
from sklearn.metrics import pairwise_distances
from dtw import dtw


def build_dtw_distance_fn(step_pattern='symmetricP2'):
    def dtw_distance(a, b):
        pd = pairwise_distances(a, b).astype(np.double)
        try:
            align = dtw(pd, distance_only=True, step_pattern=step_pattern)
            return align.normalizedDistance
        except ValueError:
            return float('inf')
    return dtw_distance


# Hack to transfer dist function to worker before forking
_WORKER_DIST_FN = {}
_WORKER_PROCESSES = 4


def _worker_helper(dist_fn_id, x, x_train, i):
    return _WORKER_DIST_FN[dist_fn_id](x, x_train), i


class KNearestNeighbors:

    def __init__(self, X, y, distance_fn, k=1, use_processes=False):
        self.X = X
        self.y = y
        self.k = k
        self.distance_fn = distance_fn

        self._dist_fn_id = -1
        if use_processes:
            self._dist_fn_id = len(_WORKER_DIST_FN)
            _WORKER_DIST_FN[self._dist_fn_id] = distance_fn
            self._pool = Pool(_WORKER_PROCESSES)

    def predict(self, x):
        return self.predict_n(x)

    def predict_n(self, *xs):
        if self._dist_fn_id < 0:
            top_k = []
            for x in xs:
                for i, x_train in enumerate(self.X):
                    d = self.distance_fn(x, x_train)
                    (heapq.heappush if len(top_k) < self.k else heapq.heappushpop
                    )(top_k, (-d, i))
            top_k = [(-d, i) for d, i in top_k]
        else:
            args = []
            for x in xs:
                for i, x_train in enumerate(self.X):
                    args.append((self._dist_fn_id, x, x_train, i))
            results = self._pool.starmap(_worker_helper, args)
            top_k = sorted(results)[:self.k]

        cls_count = Counter(self.y[i] for _, i in top_k)
        max_count = cls_count.most_common(1)[0][1]

        best_i = None
        best_cls_dist = float('inf')
        for d, i in top_k:
            cls_ = self.y[i]
            if cls_count[cls_] == max_count and d < best_cls_dist:
                best_cls_dist = d
                best_i = i
        return self.y[best_i], best_i


class Neighbors:

    def __init__(self, X, distance_fn):
        self.X = X
        self.distance_fn = distance_fn

    def find(self, x, k, min_len):
        knn_pq = []
        for i, x_train in enumerate(self.X):
            if x_train is not None and x_train.shape[0] >= min_len:
                d = self.distance_fn(x, x_train)
                (heapq.heappush if len(knn_pq) < k else heapq.heappushpop
                )(knn_pq, (-d, i))
        return [(i, -nd) for nd, i in sorted(knn_pq, key=lambda x: -x[0])]

    def dist(self, x, i):
        return self.distance_fn(x, self.X[i])
