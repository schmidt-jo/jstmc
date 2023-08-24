import pandas as pd
from jstmc import options, kernels
import numpy as np
import logging
log_module = logging.getLogger(__name__)


class GenSequence:
    def __init__(self, seq_opts: options.Sequence):
        self.seq = seq_opts
        self.params = seq_opts.params.pypulseq
        self.system = seq_opts.pp_sys

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

        self.sampling_pattern_constr: list = []
        self.sampling_pattern: pd.DataFrame = pd.DataFrame()
        self.sequence_info: dict = {}
        self.block_excitation: kernels.Kernel = kernels.Kernel()
        self.sampling_pattern_set: bool = False

        # use random state for reproducibiltiy of eg sampling patterns
        # (for that matter any random or semi random sampling used)
        self.prng = np.random.RandomState(0)

    def get_pypulseq_seq(self):
        return self.seq.pp_seq

    def get_seq(self):
        return self.seq

    def _set_grad_for_emc(self, grad):
        return 1e3 / self.seq.params.specs.gamma * grad


    def get_z(self):
        # get slice extend
        return self.z

    def _set_k_space(self):
        if self.params.accelerationFactor > 1.1:
            # calculate center of k space and indexes for full sampling band
            k_central_phase = round(self.params.resolutionNPhase / 2)
            k_half_central_lines = round(self.params.numberOfCentralLines / 2)
            # set indexes for start and end of full k space center sampling
            k_start = k_central_phase - k_half_central_lines
            k_end = k_central_phase + k_half_central_lines

            # The rest of the lines we will use tse style phase step blip between the echoes of one echo train
            # Trying random sampling, ie. pick random line numbers for remaining indices,
            # we dont want to pick the same positive as negative phase encodes to account
            # for conjugate symmetry in k-space.
            # Hence, we pick from the positive indexes twice (thinking of the center as 0) without allowing for duplexes
            # and negate half the picks
            # calculate indexes
            k_remaining = np.arange(0, k_start)
            # build array with dim [num_slices, num_outer_lines] to sample different random scheme per slice
            weighting_factor = np.clip(self.params.sampleWeighting, 0.01, 1)
            if weighting_factor > 0.05:
                log_module.info(f"\t\t-weighted random sampling of k-space phase encodes, factor: {weighting_factor}")
            # random encodes for different echoes - random choice weighted towards center
            weighting = np.clip(np.power(np.linspace(0, 1, k_start), weighting_factor), 1e-5, 1)
            weighting /= np.sum(weighting)
            for idx_echo in range(self.params.ETL):
                # same encode for all echoes -> central lines
                self.k_indexes[idx_echo, :self.params.numberOfCentralLines] = np.arange(k_start, k_end)

                k_indices = self.prng.choice(
                    k_remaining,
                    size=self.params.numberOfOuterLines,
                    replace=False,
                    p=weighting

                )
                k_indices[::2] = self.params.resolutionNPhase - 1 - k_indices[::2]
                self.k_indexes[idx_echo, self.params.numberOfCentralLines:] = np.sort(k_indices)
        else:
            self.k_indexes[:, :] = np.arange(self.params.numberOfCentralLines + self.params.numberOfOuterLines)

    def _calculate_scan_time(self):
        t_total = self.params.TR * 1e-3 * (self.params.numberOfCentralLines + self.params.numberOfOuterLines)
        log_module.info(f"\t\t-total scan time: {t_total / 60:.1f} min ({t_total:.1f} s)")

    def _set_delta_slices(self):
        # multi-slice
        numSlices = self.params.resolutionNumSlices
        # cast from mm
        delta_z = self.params.z_extend * 1e-3
        if self.params.interleavedAcquisition:
            log_module.info("\t\t-set interleaved acquisition")
            # want to go through the slices alternating from beginning and middle
            self.z.flat[:numSlices] = np.linspace((-delta_z / 2), (delta_z / 2), numSlices)
            # reshuffle slices mid+1, 1, mid+2, 2, ...
            self.z = self.z.transpose().flatten()[:numSlices]
        else:
            log_module.info("\t\t-set sequential acquisition")
            self.z = np.linspace((-delta_z / 2), (delta_z / 2), numSlices)
        # find reshuffled slice numbers
        for idx_slice_num in range(numSlices):
            z_val = self.z[idx_slice_num]
            z_pos = np.where(np.unique(self.z) == z_val)[0][0]
            self.trueSliceNum[idx_slice_num] = z_pos

