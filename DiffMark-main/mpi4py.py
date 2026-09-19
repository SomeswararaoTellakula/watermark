"""Lightweight stub of mpi4py for local dry-runs.

This provides a minimal `MPI.COMM_WORLD` object with the methods
used by the codebase so imports succeed when a real MPI install
isn't available. Only intended for local development/dry-run.
"""

class _Comm:
    def __init__(self):
        self.rank = 0
        self.size = 1

    def bcast(self, value, root=0):
        return value

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size


class _MPI:
    COMM_WORLD = _Comm()


MPI = _MPI()
