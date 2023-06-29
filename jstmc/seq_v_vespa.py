from jstmc import events, options, seq_gen
from jstmc.kernels import Kernel
import numpy as np
import logging
import tqdm

logModule = logging.getLogger(__name__)


class VespaGerdSequence(seq_gen.GenSequence):
    def __init__(self, seq_opts: options.Sequence):
        super().__init__(seq_opts)

        logModule.info(f"init vespa-gerd algorithm")

        # timing
        self.te: np.ndarray = np.zeros(self.params.ETL)
        self.t_delay_e0_e1: events.DELAY = events.DELAY()

        # echo settings
        self.num_succ_echoes = int(self.params.ETL / 2) - 1
        self.num_gres_echoes = self.params.ETL - self.num_succ_echoes - 1

        # sbbs
        self.block_pf_acquisition, grad_pre_area = Kernel.acquisition_pf_undersampled(
            params=self.params, system=self.system
        )
        self.block_acq = Kernel.acquisition_sym_undersampled(
            params=self.params, system=self.system
        )
        self.block_se1_acq = self.block_acq.copy()
        self._mod_se1_acq()
        self.block_gre_acq = Kernel.acquisition_sym_undersampled(
            params=self.params, system=self.system, invert_grad_dir=True
        )
        self.block_spoil_end: Kernel = Kernel.spoil_all_grads(
            params=self.params, system=self.system
        )
        self._mod_spoil_end()
        self.block_excitation: Kernel = Kernel.excitation_slice_sel(
            params=self.params, system=self.system, use_slice_spoiling=False
        )
        self._mod_excitation(grad_pre_area=grad_pre_area)

        self.block_refocus, self.t_spoiling_pe = Kernel.refocus_slice_sel_spoil(
            params=self.params, system=self.system, pulse_num=1, return_pe_time=True
        )
        # for the first we need a different gradient rewind
        self.block_refocus_first: Kernel = Kernel.copy(self.block_refocus)

        self._mod_first_refocus_rewind_0_echo(grad_pre_area=grad_pre_area)
        # need to adapt echo read prewinder and rewinder
        self._mod_block_prewind_echo_read(self.block_refocus_first)
        self._mod_block_rewind_echo_read(self.block_refocus)
        self._mod_block_prewind_echo_read(self.block_refocus)

        # for gradient echo samplings we need to turn around gradient read pre and rewinding
        self.block_refocus_gesse: Kernel = Kernel.copy(self.block_refocus)
        self._mod_gesse_ref()

        if self.seq.config.visualize:
            self.block_excitation.plot()
            self.block_refocus_first.plot()
            self.block_refocus.plot()
            self.block_refocus_gesse.plot()

    def _mod_excitation(self, grad_pre_area):
        # need to prewind for the pf readout of 0th echo
        rephasing_time = self.block_excitation.get_duration() - self.block_excitation.rf.get_duration()
        # set it at the start of the rephasing slice gradient
        grad_pre = events.GRAD.make_trapezoid(
            channel=self.params.read_dir, system=self.system, area=-grad_pre_area,
            duration_s=rephasing_time, delay_s=self.block_excitation.rf.get_duration()
        )
        grad_phase = events.GRAD.make_trapezoid(
            channel=self.params.phase_dir, system=self.system, area=-np.max(self.phase_areas),
            duration_s=rephasing_time, delay_s=self.block_excitation.rf.get_duration()
        )
        self.block_excitation.grad_read = grad_pre
        self.block_excitation.grad_phase = grad_phase

    def _mod_first_refocus_rewind_0_echo(self, grad_pre_area):
        # need to rewind 0 read gradient
        area_0_read_grad = self.block_pf_acquisition.grad_read.area
        area_to_rewind = area_0_read_grad - grad_pre_area
        delta_times_first_grad_part = np.diff(self.block_refocus_first.grad_read.t_array_s[:4])
        amplitude = - area_to_rewind / np.sum(np.array([0.5, 1.0, 0.5]) * delta_times_first_grad_part)
        if np.abs(amplitude) > self.system.max_grad:
            err = f"amplitude violation when rewinding 0 echo readout gradient"
            logModule.error(err)
            raise ValueError(err)
        self.block_refocus_first.grad_read.amplitude[1:3] = amplitude

    def _mod_block_prewind_echo_read(self, sbb: Kernel):
        # need to prewind readout echo gradient
        area_read = np.sum(self.block_acq.grad_read.area)
        area_prewind = - 0.5 * area_read
        delta_times_last_grad_part = np.diff(sbb.grad_read.t_array_s[-4:])
        amplitude = area_prewind / np.sum(np.array([0.5, 1.0, 0.5]) * delta_times_last_grad_part)
        if np.abs(amplitude) > self.system.max_grad:
            err = f"amplitude violation when prewinding first echo readout gradient"
            logModule.error(err)
            raise ValueError(err)
        sbb.grad_read.amplitude[-3:-1] = amplitude

    def _mod_block_rewind_echo_read(self, sbb: Kernel):
        # need to rewind readout echo gradient
        area_read = np.sum(self.block_acq.grad_read.area)
        area_rewind = - 0.5 * area_read
        delta_t_first_grad_part = np.diff(sbb.grad_read.t_array_s[:4])
        amplitude = area_rewind / np.sum(np.array([0.5, 1.0, 0.5]) * delta_t_first_grad_part)
        if np.abs(amplitude) > self.system.max_grad:
            err = f"amplitude violation when prewinding first echo readout gradient"
            logModule.error(err)
            raise ValueError(err)
        sbb.grad_read.amplitude[1:3] = amplitude

    def _mod_spoil_end(self):
        self.block_spoil_end.grad_read.amplitude = - self.block_spoil_end.grad_read.amplitude

    def _mod_se1_acq(self):
        # we dont need to accelerate first part of the readout since we have time to spare
        pass

    def _mod_gesse_ref(self):
        self.block_refocus_gesse.grad_read.amplitude = - self.block_refocus_gesse.grad_read.amplitude
        self.block_refocus_gesse.grad_read.area = - self.block_refocus_gesse.grad_read.area
        self.block_refocus_gesse.grad_read.flat_area = - self.block_refocus_gesse.grad_read.flat_area

    def build(self):
        logModule.info(f"__Build Sequence__")
        logModule.info(f"build -- calculate minimum ESP")
        self._calculate_echo_timings()
        self._loop_lines()
        self.seq.ppSeq.plot()

    def _calculate_echo_timings(self):
        # have etl echoes including 0th echo
        t_start = self.block_excitation.rf.t_delay_s + self.block_excitation.rf.t_duration_s / 2
        t_exci_0e = self.block_excitation.get_duration() - t_start + self.block_pf_acquisition.t_mid
        t_0e_1ref = self.block_pf_acquisition.get_duration() - self.block_pf_acquisition.t_mid + \
                    self.block_refocus_first.get_duration() / 2
        t_1ref_adc = self.block_refocus_first.get_duration() / 2 + self.block_acq.get_duration() / 2

        te_1 = 2 * (t_exci_0e + t_0e_1ref)
        self.te[0] = t_exci_0e
        self.te[1] = te_1

        if t_1ref_adc < te_1 / 2:
            self.t_delay_e0_e1 = events.DELAY.make_delay(te_1 / 2 - t_1ref_adc, system=self.system)

        for k in np.arange(2, self.num_succ_echoes + 1):
            self.te[k] = self.te[k-1] + 2 * t_1ref_adc
        for k in np.arange(self.num_succ_echoes + 1, self.params.ETL):
            self.te[k] = self.te[k-1] + 2 * t_1ref_adc + 2 * self.block_acq.get_duration()
        logModule.info(f"echo times: {1000*self.te} ms")

    def _loop_lines(self):
        # through phase encodes
        # line_bar = tqdm.trange(
        #     self.params.numberOfCentralLines + self.params.numberOfOuterLines, desc="phase encodes"
        # )
        line_bar = tqdm.trange(
            1, desc="phase encodes"
        )
        for idx_n in line_bar:  # We have N phase encodes for all ETL contrasts
            # for idx_slice in range(self.params.resolutionNumSlices):
            for idx_slice in range(1):
                # looping through slices per phase encode
                # self._set_fa(echo_idx=0)
                # self._set_phase_grad(phase_idx=idx_n, echo_idx=0)
                # apply slice offset
                # self._apply_slice_offset(idx_slice=idx_slice)

                # excitation
                # add block
                self.seq.ppSeq.add_block(*self.block_excitation.list_events_to_ns())

                # 0th echo sampling
                self.seq.ppSeq.add_block(*self.block_pf_acquisition.list_events_to_ns())
                # first refocus
                # add block
                self.seq.ppSeq.add_block(*self.block_refocus_first.list_events_to_ns())

                # delay if necessary
                if self.t_delay_e0_e1.get_duration() > 1e-7:
                    self.seq.ppSeq.add_block(self.t_delay_e0_e1.to_simple_ns())

                # adc
                self.seq.ppSeq.add_block(*self.block_acq.list_events_to_ns())
                # # write sampling pattern
                # self._write_sampling_pattern(phase_idx=idx_n, echo_idx=0, slice_idx=idx_slice)

                # short successive mese
                for echo_idx in np.arange(1, self.num_succ_echoes):
                    # refocus
                    self.seq.ppSeq.add_block(*self.block_refocus.list_events_to_ns())
                    # adc
                    self.seq.ppSeq.add_block(*self.block_acq.list_events_to_ns())

                # add gre sampling
                self.seq.ppSeq.add_block(*self.block_gre_acq.list_events_to_ns())

                # successive double gre + mese in center
                for echo_idx in np.arange(self.num_succ_echoes, self.params.ETL-1):
                    # refocus
                    self.seq.ppSeq.add_block(*self.block_refocus_gesse.list_events_to_ns())
                    # add gre sampling
                    self.seq.ppSeq.add_block(*self.block_gre_acq.list_events_to_ns())
                    # add se sampling
                    self.seq.ppSeq.add_block(*self.block_acq.list_events_to_ns())
                    # add gre sampling
                    self.seq.ppSeq.add_block(*self.block_gre_acq.list_events_to_ns())

                # end with spoiling
                self.seq.ppSeq.add_block(*self.block_spoil_end.list_events_to_ns())


if __name__ == '__main__':
    seq_gv = VespaGerdSequence(seq_opts=options.Sequence())
    seq_gv.build()
    te = np.zeros(seq_gv.te.shape[0]+1)
    te[1:] = seq_gv.te * 1e3
    print(te)
    print(np.diff(te))
