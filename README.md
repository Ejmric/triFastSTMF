[![DOI](https://img.shields.io/badge/DOI-arXiv-red)](https://arxiv.org/abs/2305.06624)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Ejmric/triFastSTMF/HEAD)
[![DOI](https://zenodo.org/badge/613025677.svg)](https://zenodo.org/badge/latestdoi/613025677)

# triFastSTMF: Matrix tri-factorization over the tropical semiring

triFastSTMF is a tri-factorization approach for matrix approximation and prediction based on Fast Sparse Tropical Matrix Factorization (FastSTMF).

For details, please refer to Amra Omanović, Polona Oblak, and Tomaž Curk (2023). Matrix tri-factorization over the tropical
semiring. The preprint is available in [arXiv:2305.06624](https://arxiv.org/abs/2305.06624). If you use this work, please cite:
```
@misc{omanovic2023triFastSTMF,
      title={Matrix tri-factorization over the tropical semiring}, 
      author={Amra Omanović and Polona Oblak and Tomaž Curk},
      year={2023},
      eprint={2305.06624},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Real data
We downloaded the real-world interaction dataset of an ant colony named "insecta-ant-colony3" [[1]](#1) from "Animal Social Networks" data collection on http://networkrepository.com [[2]](#2). Additional preprocessing before running our experiments is explained in the paper.

### Jupyter notebooks
The notebooks are independent and can be run in any order.

- [preprocessing_real_data.ipynb](https://github.com/Ejmric/triFastSTMF/blob/main/preprocessing_real_data.ipynb): Presents the preprocessing of the real-world interaction dataset of an ant colony.
- [heatmaps.ipynb](https://github.com/Ejmric/triFastSTMF/blob/main/heatmaps.ipynb): Presents the analysis of ants' behavioral patterns over 41 days.
- [real_exps.ipynb](https://github.com/Ejmric/triFastSTMF/blob/main/real_exps.ipynb): Presents the experiments on real data.
- [synthetic_network.ipynb](https://github.com/Ejmric/triFastSTMF/blob/main/synthetic_network.ipynb): Presents the analysis of four-partition network construction.


### Use
```
import numpy.ma as ma
import numpy as np
from triFastSTMF import triFastSTMF

data = ma.array(np.random.rand(100,100), mask=np.zeros((100,100)))
model = triFastSTMF(rank_1 = 5, rank_2 = 3, initialization="random_vcol", threshold=100)
model.fit(data)
approx = model.predict_all()
```

### References

<a id="1">[1]</a> 
D. P. Mersch, A. Crespi, and L. Keller (2013). [Tracking individuals shows spatial
fidelity is a key regulator of ant social organization](https://www.science.org/doi/10.1126/science.1234316). Science, vol. 340, no.
6136, pp. 1090–1093.

<a id="2">[2]</a> 
R. A. Rossi and N. K. Ahmed (2015) [The network data repository with
interactive graph analytics and visualization](http://networkrepository.com). AAAI. [Online].
Available: http://networkrepository.com
