"""Reuse exactly the v23 held-domain selection rule with paired REVE features."""
from threadpoolctl import threadpool_limits
from .mi_covariance_fit_v23 import main
from .mi_reve_scaling_v24 import ROOT, SOURCE, ARMS

if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main(root=ROOT, source=SOURCE, arms=ARMS, dimension=1536)
