#encoding: utf-8
#
# 10回コインを投げて出た実験結果を元に確率モデルを推定するトイプログラム
# 最初にコインの裏表同じ確率ででるモデルを最初に構築し、次に上記の実験結果のデータを与え推論する
# 最後にモデルを検証するのに、モデルからキャプチャしたコイン投げを解析する。
# Edwardを使うと、コードの行数をかなり削減できる。

import edward as ed
import numpy as np

from edward.models import PythonModel, Variational, Beta
from scipy.stats import beta, bernoulli


class BetaBernoulli(PythonModel):
    def _py_log_prob(self, xs, zs):
        n_samples = zs.shape[0]
        lp = np.zeros(n_samples, dtype=np.float32)
        for s in range(n_samples):
            lp[s] = beta.logpdf(zs[s, :], a=1.0, b=1.0)
            for n in range(len(xs)):
                lp[s] += bernoulli.logpmf(xs[n], p=zs[s, :])
        return lp


def main():
    data = ed.Data(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    model = BetaBernoulli()
    variational = Variational()
    variational.add(Beta())

    # mean-field variational inference.
    inference = ed.MFVI(model, variational, data)
    
    inference.run(n_iter=10000)

if __name__ == '__main__':
    main()
