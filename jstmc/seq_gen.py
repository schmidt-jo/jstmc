from jstmc import options
import numpy as np


class GenSequence:
    def __init__(self, seq_opts: options.Sequence):
        self.seq = seq_opts
        self.params = seq_opts.params
        self.system = seq_opts.ppSys

        # phase grads
        self.phase_areas: np.ndarray = (- np.arange(self.params.resolutionNPhase) +
                                        self.params.resolutionNPhase / 2) * self.params.deltaK_phase
        # slice loop
        self.num_slices = self.params.resolutionNumSlices
        self.z = np.zeros((2, int(np.ceil(self.num_slices / 2))))

        self.trueSliceNum = np.zeros(self.num_slices)
        # k space
        self.k_indexes: np.ndarray = np.zeros(
            (self.params.ETL, self.params.numberOfCentralLines + self.params.numberOfOuterLines),
            dtype=int
        )

        self.sampling_pattern: list = []

    def get_pypulseq_seq(self):
        return self.seq.ppSeq

    def get_seq(self):
        return self.seq

    def _set_grad_for_emc(self, grad):
        return 1e3 / self.seq.specs.gamma * grad

    def get_emc_info(self) -> dict:
        return NotImplemented

    def get_sampling_pattern(self) -> list:
        return self.sampling_pattern

    def get_z(self):
        # get slice extend
        return self.z
