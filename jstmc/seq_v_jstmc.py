from jstmc import events, options, kernels, seq_gen
import numpy as np
import logging
import tqdm

logModule = logging.getLogger(__name__)


class JsTmcSequence(seq_gen.GenSequence):
    def __init__(self, seq_opts: options.Sequence):
        super().__init__(seq_opts)
        logModule.info(f"init jstmc algorithm")

        # timing
        self.esp: float = 0.0
        self.delay_exci_ref1: events.DELAY = events.DELAY()
        self.delay_ref_adc: events.DELAY = events.DELAY()
        self.phase_enc_time: float = 0.0
        self.delay_slice: events.DELAY = events.DELAY()

        # sbbs
        self.block_refocus_1: kernels.EventBlock = kernels.EventBlock.build_jstmc_refocus(params=self.params,
                                                                                          system=self.system,
                                                                                          pulse_num=0)
        ramp_area_ref_1 = self.block_refocus_1.grad_slice.t_array_s[1] * self.block_refocus_1.grad_slice.amplitude[
            1] / 2.0
        self.block_excitation: kernels.EventBlock = kernels.EventBlock.build_excitation(params=self.params,
                                                                                        system=self.system,
                                                                                        adjust_ramp_area=ramp_area_ref_1)

        self.block_acquisition: kernels.EventBlock = kernels.EventBlock.build_jstmc_acquisition(params=self.params,
                                                                                                system=self.system)
        self.block_refocus, self.phase_enc_time = kernels.EventBlock.build_jstmc_refocus(
            params=self.params, system=self.system, pulse_num=1, return_pe_time=True
        )
        self.block_spoil_end: kernels.EventBlock = kernels.EventBlock.build_jstmc_spoiler_end(params=self.params,
                                                                                              system=self.system)
        if self.seq.config.visualize:
            self.block_excitation.plot()
            self.block_refocus_1.plot()
            self.block_refocus.plot()

    def calculate_min_esp(self):
        # calculate time between midpoints
        t_exci_ref = self.block_refocus_1.rf.t_delay_s + self.block_refocus_1.rf.t_duration_s / 2 + \
                     self.block_excitation.get_duration() - self.block_excitation.rf.t_delay_s - \
                     self.block_excitation.rf.t_duration_s / 2
        t_ref_1_adc = self.block_refocus_1.get_duration() - self.block_refocus_1.rf.t_delay_s - \
                      self.block_refocus_1.rf.t_duration_s / 2 + self.block_acquisition.get_duration() / 2
        t_ref_2_adc = self.block_acquisition.get_duration() / 2 + self.block_refocus.get_duration() / 2

        self.params.ESP = 2 * np.max([t_exci_ref, t_ref_1_adc, t_ref_2_adc]) * 1e3
        logModule.info(f"\t\t-found minimum ESP: {self.params.ESP:.2f} ms")

        if np.abs(t_ref_1_adc - t_ref_2_adc) > 1e-6:
            logModule.error(f"refocus to adc timing different from adc to refocus. Systematic error in seq. creation")
        t_half_esp = self.params.ESP * 1e-3 / 2
        # add delays
        if t_exci_ref < t_half_esp:
            self.delay_exci_ref1 = events.DELAY.make_delay(t_half_esp - t_exci_ref, system=self.system)
            if not self.delay_exci_ref1.check_on_block_raster():
                err = f"exci ref delay not on block raster"
                logModule.error(err)
        if t_ref_1_adc < t_half_esp:
            self.delay_ref_adc = events.DELAY.make_delay(t_half_esp - t_ref_1_adc, system=self.system)
            if not self.delay_ref_adc.check_on_block_raster():
                err = f"adc ref delay not on block raster"
                logModule.error(err)

    def _calculate_slice_delay(self):
        # time per echo train
        t_pre_etl = self.block_excitation.rf.t_delay_s + self.block_excitation.rf.t_duration_s / 2
        t_etl = self.params.ETL * self.params.ESP * 1e-3  # esp in ms
        t_post_etl = self.block_acquisition.get_duration() / 2 + self.phase_enc_time

        t_total_etl = t_pre_etl + t_etl + t_post_etl

        max_num_slices = int(np.floor(self.params.TR * 1e-3 / t_total_etl))
        logModule.info(f"\t\t-total echo train length: {t_total_etl * 1e3:.2f} ms")
        logModule.info(f"\t\t-desired number of slices: {self.params.resolutionNumSlices}")
        logModule.info(f"\t\t-possible number of slices within TR: {max_num_slices}")
        if self.params.resolutionNumSlices > max_num_slices:
            logModule.info(f"increase TR or Concatenation needed")

        self.delay_slice = events.DELAY.make_delay(
            self.params.TR * 1e-3 / self.params.resolutionNumSlices - t_total_etl, system=self.system
        )
        logModule.info(f"\t\t-time between slices: {self.delay_slice.get_duration() * 1e3:.2f} ms")
        if not self.delay_slice.check_on_block_raster():
            self.delay_slice.set_on_block_raster()
            logModule.info(f"\t\t-adjusting TR delay to raster time: {self.delay_slice.get_duration() * 1e3:.2f} ms")

    def _calculate_scan_time(self):
        t_total = self.params.TR * 1e-3 * (self.params.numberOfCentralLines + self.params.numberOfOuterLines)
        logModule.info(f"\t\t-total scan time: {t_total / 60:.1f} min ({t_total:.1f} s)")

    def build(self):
        logModule.info(f"__Build Sequence__")
        logModule.info(f"build -- calculate minimum ESP")
        self.calculate_min_esp()
        logModule.info(f"build -- calculate slice delay")
        self._calculate_slice_delay()
        logModule.info(f"build -- calculate total scan time")
        self._calculate_scan_time()
        logModule.info(f"build -- set up k-space")
        self._set_k_space()
        logModule.info(f"build -- set up slices")
        self._set_delta_slices()
        logModule.info(f"build -- loop lines")
        self._loop_lines()

    def _write_sampling_pattern(self, echo_idx: int, phase_idx: int, slice_idx: int):
        pe_num = self.k_indexes[echo_idx, phase_idx]
        sampling_index = {"pe_num": pe_num, "slice_num": int(self.trueSliceNum[slice_idx]), "echo_num": echo_idx}
        self.sampling_pattern.append(sampling_index)

    def _loop_lines(self):
        # through phase encodes
        line_bar = tqdm.trange(
            self.params.numberOfCentralLines + self.params.numberOfOuterLines, desc="phase encodes"
        )
        for idx_n in line_bar:  # We have N phase encodes for all ETL contrasts
            for idx_slice in range(self.params.resolutionNumSlices):
                # looping through slices per phase encode
                self._set_fa(echo_idx=0)
                self._set_phase_grad(phase_idx=idx_n, echo_idx=0)
                # apply slice offset
                self._apply_slice_offset(idx_slice=idx_slice)

                # excitation
                # add block
                self.seq.ppSeq.add_block(*self.block_excitation.list_events_to_ns())

                # delay if necessary
                if self.delay_exci_ref1.get_duration() > 1e-7:
                    self.seq.ppSeq.add_block(self.delay_exci_ref1.to_simple_ns())

                # first refocus
                # add block
                self.seq.ppSeq.add_block(*self.block_refocus_1.list_events_to_ns())

                # delay if necessary
                if self.delay_ref_adc.get_duration() > 1e-7:
                    self.seq.ppSeq.add_block(self.delay_ref_adc.to_simple_ns())

                # adc
                self.seq.ppSeq.add_block(*self.block_acquisition.list_events_to_ns())
                # write sampling pattern
                self._write_sampling_pattern(phase_idx=idx_n, echo_idx=0, slice_idx=idx_slice)
                # delay if necessary
                if self.delay_ref_adc.get_duration() > 1e-7:
                    self.seq.ppSeq.add_block(self.delay_ref_adc.to_simple_ns())

                # loop
                for echo_idx in np.arange(1, self.params.ETL):
                    # set fa
                    self._set_fa(echo_idx=echo_idx)
                    # set phase
                    self._set_phase_grad(echo_idx=echo_idx, phase_idx=idx_n)
                    # set slice offset
                    self._apply_slice_offset(idx_slice=idx_slice)
                    # add block
                    self.seq.ppSeq.add_block(*self.block_refocus.list_events_to_ns())
                    # delay if necessary
                    if self.delay_ref_adc.get_duration() > 1e-7:
                        self.seq.ppSeq.add_block(self.delay_ref_adc.to_simple_ns())

                    # adc
                    self.seq.ppSeq.add_block(*self.block_acquisition.list_events_to_ns())
                    # write sampling pattern
                    self._write_sampling_pattern(echo_idx=echo_idx, phase_idx=idx_n, slice_idx=idx_slice)

                    # delay if necessary
                    if self.delay_ref_adc.get_duration() > 1e-7:
                        self.seq.ppSeq.add_block(self.delay_ref_adc.to_simple_ns())
                # spoil end
                self._set_end_spoil_phase_grad()
                self.seq.ppSeq.add_block(*self.block_spoil_end.list_events_to_ns())
                # insert TR
                self.seq.ppSeq.add_block(self.delay_slice.to_simple_ns())
        logModule.info(f"sequence built!")

    def _set_fa(self, echo_idx: int):
        if echo_idx == 0:
            sbb = self.block_refocus_1
        else:
            sbb = self.block_refocus
        flip = sbb.rf.t_duration_s / sbb.rf.signal.shape[0] * np.sum(np.abs(sbb.rf.signal)) * 2 * np.pi
        fa_rad = self.params.refocusingRadFA[echo_idx]
        phase_rad = self.params.refocusingRadRfPhase[echo_idx]
        sbb.rf.signal *= fa_rad / flip
        sbb.rf.phase_rad = phase_rad

    def _set_phase_grad(self, echo_idx: int, phase_idx: int):
        idx_phase = self.k_indexes[echo_idx, phase_idx]
        if echo_idx == 0:
            sbb = self.block_refocus_1
        else:
            sbb = self.block_refocus
            last_idx_phase = self.k_indexes[echo_idx - 1, phase_idx]
            sbb.grad_phase.amplitude[1:3] = self.phase_areas[last_idx_phase] / self.phase_enc_time
        if np.abs(self.phase_areas[idx_phase]) > 1:
            sbb.grad_phase.amplitude[-3:-1] = - self.phase_areas[idx_phase] / self.phase_enc_time
        else:
            sbb.grad_phase.amplitude = np.zeros_like(sbb.grad_phase.amplitude)

    def _set_end_spoil_phase_grad(self):
        factor = np.array([0.5, 1.0, 0.5])

        # get phase moment of last phase encode
        pe_grad_amp = self.block_refocus.grad_phase.amplitude[-2]
        pe_grad_times = self.block_refocus.grad_phase.t_array_s[-4:]
        delta_times = np.diff(pe_grad_times)
        area = np.sum(delta_times * pe_grad_amp * factor)

        # adopt last grad to inverse area
        pe_end_times = self.block_spoil_end.grad_phase.t_array_s[-4:]
        delta_end_times = np.diff(pe_end_times)
        pe_end_amp = area / np.sum(factor * delta_end_times)
        if np.abs(pe_end_amp) > self.system.max_grad:
            err = f"amplitude violation upon last pe grad setting"
            logModule.error(err)
            raise AttributeError(err)
        self.block_spoil_end.grad_phase.amplitude[1:3] = - pe_end_amp

    def _apply_slice_offset(self, idx_slice: int):
        for sbb in [self.block_excitation, self.block_refocus_1, self.block_refocus]:
            grad_slice_amplitude_hz = sbb.grad_slice.amplitude[sbb.grad_slice.t_array_s >= sbb.rf.t_delay_s][0]
            sbb.rf.freq_offset_hz = grad_slice_amplitude_hz * self.z[idx_slice]
            # we are setting the phase of a pulse here into its phase offset var.
            # To merge both: given phase parameter and any complex signal array data
            sbb.rf.phase_offset_rad = sbb.rf.phase_rad - 2 * np.pi * sbb.rf.freq_offset_hz * sbb.rf.calculate_center()

    def _set_delta_slices(self):
        # multi-slice
        numSlices = self.params.resolutionNumSlices
        # cast from mm
        delta_z = self.params.z_extend * 1e-3
        if self.params.interleavedAcquisition:
            logModule.info("\t\t-set interleaved acquisition")
            # want to go through the slices alternating from beginning and middle
            self.z.flat[:numSlices] = np.linspace((-delta_z / 2), (delta_z / 2), numSlices)
            # reshuffle slices mid+1, 1, mid+2, 2, ...
            self.z = self.z.transpose().flatten()[:numSlices]
        else:
            logModule.info("\t\t-set sequential acquisition")
            self.z = np.linspace((-delta_z / 2), (delta_z / 2), numSlices)
        # find reshuffled slice numbers
        for idx_slice_num in range(numSlices):
            z_val = self.z[idx_slice_num]
            z_pos = np.where(np.unique(self.z) == z_val)[0][0]
            self.trueSliceNum[idx_slice_num] = z_pos

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
                logModule.info(f"\t\t-weighted random sampling of k-space phase encodes, factor: {weighting_factor}")
            # random encodes for different echoes - random choice weighted towards center
            weighting = np.clip(np.power(np.linspace(0, 1, k_start), weighting_factor), 1e-5, 1)
            weighting /= np.sum(weighting)
            for idx_echo in range(self.params.ETL):
                # same encode for all echoes -> central lines
                self.k_indexes[idx_echo, :self.params.numberOfCentralLines] = np.arange(k_start, k_end)

                k_indices = np.random.choice(
                    k_remaining,
                    size=self.params.numberOfOuterLines,
                    replace=False,
                    p=weighting

                )
                k_indices[::2] = self.params.resolutionNPhase - 1 - k_indices[::2]
                self.k_indexes[idx_echo, self.params.numberOfCentralLines:] = np.sort(k_indices)
        else:
            self.k_indexes[:, :] = np.arange(self.params.numberOfCentralLines + self.params.numberOfOuterLines)

    def get_emc_info(self) -> dict:
        emc_dict = {
            "gammaHz": self.seq.specs.gamma,
            "ETL": self.params.ETL,
            "ESP": self.params.ESP,
            "bw": self.params.bandwidth,
            "gradMode": "Normal",
            "excitationAngle": self.params.excitationRadFA / np.pi * 180.0,
            "excitationPhase": self.params.excitationRfPhase,
            "gradientExcitation": self._set_grad_for_emc(self.block_excitation.grad_slice.slice_select_amplitude),
            "durationExcitation": self.params.excitationDuration,
            "gradientExcitationRephase": self._set_grad_for_emc(self.block_excitation.grad_slice.amplitude[-2]),
            "durationExcitationRephase": np.sum(np.diff(self.block_excitation.grad_slice.t_array_s[-4:])) * 1e6,
            "gradientExcitationVerse1": 0.0,
            "gradientExcitationVerse2": 0.0,
            "durationExcitationVerse1": 0.0,
            "durationExcitationVerse2": 0.0,
            "refocusAngle": self.params.refocusingFA,
            "refocusPhase": self.params.refocusingRfPhase,
            "gradientRefocus": self._set_grad_for_emc(self.block_refocus.grad_slice.slice_select_amplitude),
            "durationRefocus": self.params.refocusingDuration,
            "gradientCrush": self._set_grad_for_emc(self.block_refocus.grad_slice.amplitude[1]),
            "durationCrush": self.phase_enc_time * 1e6,
            "gradientRefocusVerse1": 0.0,
            "gradientRefocusVerse2": 0.0,
            "durationRefocusVerse1": 0.0,
            "durationRefocusVerse2": 0.0
        }
        return emc_dict

    def get_pulse_amplitudes(self) -> np.ndarray:
        exc_pulse = self.block_excitation.rf.signal
        return exc_pulse



if __name__ == '__main__':
    seq = JsTmcSequence(options.Sequence())
    seq.build()
    emc_dict = seq.get_emc_info()
    pp_seq = seq.get_pypulseq_seq()
    pp_seq.plot()
