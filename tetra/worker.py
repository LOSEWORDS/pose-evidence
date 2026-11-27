import sys
from tetra.runtime.store import Store
from tetra.prediction.backend import run_job

if __name__=='__main__':run_job(Store(sys.argv[1]),sys.argv[2])
