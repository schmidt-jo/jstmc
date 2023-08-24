from jstmc import events, options, seq_gen, plotting
from jstmc.kernels import Kernel
import numpy as np
import logging
import tqdm
import pandas as pd

log_module = logging.getLogger(__name__)


class VespaGerdSequence(seq_gen.GenSequence):
    def __init__(self, seq_opts: options.Sequence):
        super().__init__(seq_opts)

        log_module.info(f"init vespa-gerd algorithm")
        self.num_gre_lobes: int = 1

        # timing
        self.te: np.ndarray = np.zeros(1 + 3 * self.params.ETL)
        self.t_delay_e0_ref1: events.DELAY = events.DELAY()
        self.t_delay_ref1_se1: events.DELAY = events.DELAY()

        # sbbs
        # partial fourier acquisition -> 0th echo
        self.block_pf_acquisition, grad_pre_area = Kernel.acquisition_pf_undersampled(
            params=self.params, system=self.system
        )
        # undersampled readout with symmetrical accelerated sidelobes
        self.block_se_acq, self.acc_factor_us_read = Kernel.acquisition_sym_undersampled(
            params=self.params, system=self.system
        )

        # gradient echo readouts, always have inverted gradient direction wrt to se readouts,
        # - inverted gradient, k-space from right to left
        self.block_gre_acq, _ = Kernel.acquisition_sym_undersampled(
            params=self.params, system=self.system, invert_grad_dir=True
        )
        # spoiling at end of echo train
        self.block_spoil_end: Kernel = Kernel.spoil_all_grads(
            params=self.params, system=self.system
        )
        self._mod_spoiling_end()

        # excitation pulse
        self.block_excitation: Kernel = Kernel.excitation_slice_sel(
            params=self.params, system=self.system, use_slice_spoiling=False
        )
        self._mod_excitation(grad_pre_area=grad_pre_area)

        # refocusing
        self.block_refocus, self.t_spoiling_pe = Kernel.refocus_slice_sel_spoil(
            params=self.params, system=self.system, pulse_num=1, return_pe_time=True
        )
        # sanity check
        if np.abs(np.sum(self.block_se_acq.grad_read.area)) - np.abs(np.sum(self.block_gre_acq.grad_read.area)) > 1e-8:
            err = f"readout areas of gradient and spin echo readouts differ"
            log_module.error(err)
            raise ValueError(err)

        # for the first we need a different gradient rewind to rephase the partial fourier readout k-space travel
        self.block_refocus_first: Kernel = Kernel.copy(self.block_refocus)
        self._mod_first_refocus_rewind_0_echo(grad_pre_area=grad_pre_area)
        # need to adapt echo read prewinder and rewinder
        self._mod_block_prewind_echo_read(self.block_refocus_first)

        self._mod_block_rewind_echo_read(self.block_refocus)
        self._mod_block_prewind_echo_read(self.block_refocus)

        # plot files for visualization
        if self.seq.config.visualize:
            self.block_excitation.plot(path=self.seq.config.outputPath, name="excitation")
            self.block_refocus_first.plot(path=self.seq.config.outputPath, name="refocus-first")
            self.block_refocus.plot(path=self.seq.config.outputPath, name="refocus")
            self.block_pf_acquisition.plot(path=self.seq.config.outputPath, name="partial-fourier-acqusisition")
            self.block_se_acq.plot(path=self.seq.config.outputPath, name="undersampled-acquisition")
            self.block_gre_acq.plot(path=self.seq.config.outputPath, name="undersampled-gre-acq")

        # ToDo:
        # as is now all gesse readouts sample the same phase encode lines as the spin echoes.
        # this would allow joint recon of t2 and t2* contrasts independently
        # but we could also benefit even more from joint recon of all echoes and
        # hence switch up the phase encode scheme even further also in between gesse samplings

    def build_save_k_traj(self, k_trajs: list, labels: list, n_samples_ref: int) -> pd.DataFrame:
        # sanity check
        if len(k_trajs) != len(labels):
            err = "provide same number of labels as trajectories"
            log_module.error(err)
            raise AttributeError(err)
        # build k_traj df
        # add fully sampled reference
        k_labels = ["fs_ref"] * n_samples_ref
        traj_data: list = np.linspace(-0.5, 0.5, n_samples_ref).tolist()
        traj_pts: list = np.arange(n_samples_ref).tolist()
        for traj_idx in range(len(k_trajs)):
            k_labels.extend([labels[traj_idx]] * k_trajs[traj_idx].shape[0])
            traj_data.extend(k_trajs[traj_idx].tolist())
            traj_pts.extend(np.arange(k_trajs[traj_idx].shape[0]).tolist())

        k_traj_df = pd.DataFrame({
            "acquisition": k_labels, "k_read_position": traj_data, "adc_sampling_num": traj_pts
        })
        self.seq.write_k_traj(k_traj_df)
        return k_traj_df

    def _mod_spoiling_end(self):
        # want to enable complete refocusing of read gradient when spoiling factor -0.5 is chosen in opts
        readout_area = np.trapz(
            x=self.block_gre_acq.grad_read.t_array_s,
            y=self.block_gre_acq.grad_read.amplitude
        )
        spoil_area = self.params.readSpoilingFactor * readout_area
        # now we need to plug in new amplitude into spoiling read gradient
        t_sr = np.sum(
            np.diff(
                self.block_spoil_end.grad_read.t_array_s[-4:]
            ) * np.array([0.5, 1.0, 0.5])
        )
        self.block_spoil_end.grad_read.amplitude[-3:-1] = spoil_area / t_sr

    def _mod_excitation(self, grad_pre_area):
        # need to prewind for the pf readout of 0th echo
        rephasing_time = self.block_excitation.get_duration() - self.block_excitation.rf.get_duration() + \
                         self.block_excitation.rf.t_ringdown_s
        # set it at the start of the rephasing slice gradient
        grad_pre = events.GRAD.make_trapezoid(
            channel=self.params.read_dir, system=self.system, area=-grad_pre_area,
            duration_s=rephasing_time,
            delay_s=self.block_excitation.rf.t_delay_s + self.block_excitation.rf.t_duration_s
        )
        grad_phase = events.GRAD.make_trapezoid(
            channel=self.params.phase_dir, system=self.system, area=-np.max(self.phase_areas),
            duration_s=rephasing_time,
            delay_s=self.block_excitation.rf.t_delay_s + self.block_excitation.rf.t_duration_s
        )
        self.block_excitation.grad_read = grad_pre
        self.block_excitation.grad_phase = grad_phase

    def _mod_first_refocus_rewind_0_echo(self, grad_pre_area):
        # need to rewind 0 read gradient
        # get whole read area
        area_0_read_grad = self.block_pf_acquisition.grad_read.area
        # substract prewound area (dependent on partial fourier factor)
        area_to_rewind = area_0_read_grad - grad_pre_area
        # get times of the gradient to adopt - read gradient, and calculate deltas
        delta_times_first_grad_part = np.diff(self.block_refocus_first.grad_read.t_array_s[:4])
        # amplitude at trapezoid points is middle rectangle plus 2 ramp triangles
        amplitude = - area_to_rewind / np.sum(np.array([0.5, 1.0, 0.5]) * delta_times_first_grad_part)
        # check max grad violation
        if np.abs(amplitude) > self.system.max_grad:
            err = f"amplitude violation when rewinding 0 echo readout gradient"
            log_module.error(err)
            raise ValueError(err)
        # assign
        self.block_refocus_first.grad_read.amplitude[1:3] = amplitude
        self.block_refocus_first.grad_read.area[0] = np.trapz(
            x=self.block_refocus_first.grad_read.t_array_s[:4],
            y=self.block_refocus_first.grad_read.amplitude[:4]
        )

    def _mod_block_prewind_echo_read(self, sbb: Kernel):
        # need to prewind readout echo gradient
        area_read = np.sum(self.block_gre_acq.grad_read.area)
        area_prewind = - 0.5 * area_read
        delta_times_last_grad_part = np.diff(sbb.grad_read.t_array_s[-4:])
        amplitude = area_prewind / np.sum(np.array([0.5, 1.0, 0.5]) * delta_times_last_grad_part)
        if np.abs(amplitude) > self.system.max_grad:
            err = f"amplitude violation when prewinding first echo readout gradient"
            log_module.error(err)
            raise ValueError(err)
        sbb.grad_read.amplitude[-3:-1] = amplitude
        sbb.grad_read.area[-1] = area_prewind

    def _mod_block_rewind_echo_read(self, sbb: Kernel):
        # need to rewind readout echo gradient
        area_read = np.sum(self.block_gre_acq.grad_read.area)
        area_rewind = - 0.5 * area_read
        delta_t_first_grad_part = np.diff(sbb.grad_read.t_array_s[:4])
        amplitude = area_rewind / np.sum(np.array([0.5, 1.0, 0.5]) * delta_t_first_grad_part)
        if np.abs(amplitude) > self.system.max_grad:
            err = f"amplitude violation when prewinding first echo readout gradient"
            log_module.error(err)
            raise ValueError(err)
        sbb.grad_read.amplitude[1:3] = amplitude
        sbb.grad_read.area[0] = area_rewind

    def _set_seq_info(self):
        self.sampling_pattern = pd.DataFrame(self.sampling_pattern_constr)
        self.sequence_info = {
            "IMG_N_read": self.params.resolutionNRead,
            "IMG_N_phase": self.params.resolutionNPhase,
            "IMG_N_slice": self.params.resolutionNumSlices,
            "IMG_resolution_read": self.params.resolutionVoxelSizeRead,
            "IMG_resolution_phase": self.params.resolutionVoxelSizePhase,
            "IMG_resolution_slice": self.params.resolutionSliceThickness,
            "ETL": self.params.ETL,
            "OS_factor": self.params.oversampling,
            "READ_dir": self.params.read_dir,
            "ACC_factor_phase": self.params.accelerationFactor,
            "ACC_factor_read": self.acc_factor_us_read,
            "TE": self.te.tolist()
        }

    def get_emc_info(self) -> dict:
        t_rephase = (self.block_excitation.get_duration() -
                     (self.block_excitation.rf.t_duration_s + self.block_excitation.rf.t_delay_s))
        amp_rephase = self.block_excitation.grad_slice.area[-1] / t_rephase
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
            "gradientExcitationRephase": self._set_grad_for_emc(amp_rephase),
            "durationExcitationRephase": t_rephase * 1e6,
            "gradientExcitationVerse1": 0.0,
            "gradientExcitationVerse2": 0.0,
            "durationExcitationVerse1": 0.0,
            "durationExcitationVerse2": 0.0,
            "refocusAngle": self.params.refocusingFA,
            "refocusPhase": self.params.refocusingRfPhase,
            "gradientRefocus": self._set_grad_for_emc(self.block_refocus.grad_slice.slice_select_amplitude),
            "durationRefocus": self.params.refocusingDuration,
            "gradientCrush": self._set_grad_for_emc(self.block_refocus.grad_slice.amplitude[1]),
            "durationCrush": np.sum(np.diff(self.block_refocus.grad_slice.t_array_s[-4:])) * 1e6,
            "gradientRefocusVerse1": 0.0,
            "gradientRefocusVerse2": 0.0,
            "durationRefocusVerse1": 0.0,
            "durationRefocusVerse2": 0.0
        }
        return emc_dict

    def build(self):
        log_module.info(f"__Build Sequence__")
        log_module.info(f"build -- calculate minimum ESP")
        self._calculate_echo_timings()
        log_module.info(f"build -- calculate slice delay")
        self._calculate_slice_delay()
        log_module.info(f"build -- calculate total scan time")
        self._calculate_scan_time()
        log_module.info(f"build -- set up k-space")
        self._set_k_space()
        log_module.info(f"build -- set up slices")
        self._set_delta_slices()
        log_module.info(f"build -- loop lines")
        self._loop_lines()
        self.simulate_grad_moments()
        self._set_seq_info()
        self._set_k_space_sampling_readout_patterns()

    def _calculate_slice_delay(self):
        # time per echo train
        # time to mid excitation
        t_pre_etl = self.block_excitation.rf.t_delay_s + self.block_excitation.rf.t_duration_s / 2
        # time of etl
        t_etl = self.te[-1]
        # time from mid last gre til end
        t_post_etl = self.block_gre_acq.get_duration() / 2 + self.block_spoil_end.get_duration()
        # total echo train length
        t_total_etl = (t_pre_etl + t_etl + t_post_etl) * 1e3  # esp in ms
        max_num_slices = int(np.floor(self.params.TR / t_total_etl))
        log_module.info(f"\t\t-total echo train length: {t_total_etl:.2f} ms")
        log_module.info(f"\t\t-desired number of slices: {self.params.resolutionNumSlices}")
        log_module.info(f"\t\t-possible number of slices within TR: {max_num_slices}")
        if self.params.resolutionNumSlices > max_num_slices:
            time_missing = (self.params.resolutionNumSlices - max_num_slices) * t_total_etl
            log_module.info(f"increase TR or Concatenation needed. - need {time_missing:.2f} ms more")
        self.delay_slice = events.DELAY.make_delay(
            1e-3 * (self.params.TR - self.params.resolutionNumSlices * t_total_etl) /
            self.params.resolutionNumSlices,
            system=self.system
        )
        log_module.info(f"\t\t-time between slices: {self.delay_slice.get_duration() * 1e3:.2f} ms")
        if not self.delay_slice.check_on_block_raster():
            self.delay_slice.set_on_block_raster()
            log_module.info(f"\t\t-adjusting TR delay to raster time: {self.delay_slice.get_duration() * 1e3:.2f} ms")

    def _calculate_echo_timings(self):
        # have etl echoes including 0th echo
        # find midpoint of rf
        t_start = self.block_excitation.rf.t_delay_s + self.block_excitation.rf.t_duration_s / 2
        # find time between mid rf and mid 0th echo
        t_exci_0e = self.block_excitation.get_duration() - t_start + self.block_pf_acquisition.t_mid
        # find time between mid 0th echo and mid first refocus
        t_0e_1ref = self.block_pf_acquisition.get_duration() - self.block_pf_acquisition.t_mid + \
                    self.block_refocus_first.get_duration() / 2

        # find time between mid refocus and first gre and first se
        t_ref1_gre1 = self.block_refocus_first.get_duration() / 2 + self.block_gre_acq.get_duration() / 2
        t_gre1_se1 = self.block_gre_acq.get_duration() / 2 + self.block_se_acq.get_duration() / 2

        # echo time of first se is twice the bigger time of 1) between excitation and first ref
        # 2) between first ref and se
        te_1 = 2 * np.max([t_exci_0e + t_0e_1ref, t_ref1_gre1 + t_gre1_se1])

        # time to either side between exxcitation - ref - se needs to be equal, calculate appropriate delays
        if t_exci_0e + t_0e_1ref < te_1 / 2:
            self.t_delay_e0_ref1 = events.DELAY.make_delay(te_1 / 2 - t_exci_0e - t_0e_1ref, system=self.system)
        if t_ref1_gre1 + t_gre1_se1 < te_1 / 2:
            self.t_delay_ref1_se1 = events.DELAY.make_delay(te_1 / 2 - t_ref1_gre1 - t_gre1_se1, system=self.system)

        # write echo times to array
        self.te[0] = t_exci_0e
        self.te[1] = t_exci_0e + t_0e_1ref + self.t_delay_e0_ref1.get_duration() + \
                     self.t_delay_ref1_se1.get_duration() + t_ref1_gre1
        self.te[2] = te_1
        self.te[3] = te_1 + t_gre1_se1
        for k in np.arange(4, self.te.shape[0], 3):
            # take last echo time (gre sampling after se) need to add time from gre to rf and from rf to gre (equal)
            self.te[k] = self.te[k - 1] + 2 * t_ref1_gre1
            # take this time and add time between gre and se
            self.te[k + 1] = self.te[k] + t_gre1_se1
            # and same amount again to arrive at gre sampling
            self.te[k + 2] = self.te[k + 1] + t_gre1_se1
        log_module.info(f"echo times: {1000 * self.te} ms")

    def _set_fa(self, rf_idx: int):
        # we take same kernels for different refocusing pulses when going through the sequence
        # want to adopt rf flip angle and phase based on given input parameters via options
        block = self._get_refocus_block_from_echo_idx(rf_idx=rf_idx)
        # calculate flip angle as given
        flip = block.rf.t_duration_s / block.rf.signal.shape[0] * np.sum(np.abs(block.rf.signal)) * 2 * np.pi
        # take flip angle in radiants from options
        fa_rad = self.params.refocusingRadFA[rf_idx]
        # take phase as given in options
        phase_rad = self.params.refocusingRadRfPhase[rf_idx]
        # set block values
        block.rf.signal *= fa_rad / flip
        block.rf.phase_rad = phase_rad

    def _get_refocus_block_from_echo_idx(self, rf_idx: int) -> Kernel:
        # want to choose the rf based on position in echo train
        if rf_idx == 0:
            # first refocusing is different kernel
            block = self.block_refocus_first
        else:
            # we are on usual gesse echoes, past the first refocus
            block = self.block_refocus
        return block

    def _set_phase_grad(self, echo_idx: int, phase_idx: int, excitation: bool = False):
        # caution we assume trapezoidal phase encode gradients
        area_factors = np.array([0.5, 1.0, 0.5])
        # we get the actual line index from the sampling pattern, dependent on echo number and phase index in the loop
        idx_phase = self.k_indexes[echo_idx, phase_idx]
        # additionally we need the last blocks phase encode for rephasing
        if echo_idx > 0:
            # if we are not on the first readout:
            # we need the last phase encode value to reset before refocusing
            last_idx_phase = self.k_indexes[echo_idx - 1, phase_idx]
        else:
            # we need the phase encode from the 0th echo, as is now it is also encoded like the refocused se readout
            last_idx_phase = self.k_indexes[echo_idx, phase_idx]

        # we choose the block based on position in the echo train
        if excitation:
            # upon excitation, we don't need to do any re-phasing
            block = self.block_excitation
        else:
            block = self._get_refocus_block_from_echo_idx(rf_idx=echo_idx)
            # if not on excitation we set the re-phase phase encode gradient
            phase_enc_time_pre_pulse = np.sum(np.diff(block.grad_phase.t_array_s[:4]) * area_factors)
            block.grad_phase.amplitude[1:3] = self.phase_areas[last_idx_phase] / phase_enc_time_pre_pulse
        # we get the time of the phase encode after pulse for every event
        phase_enc_time_post_pulse = np.sum(np.diff(block.grad_phase.t_array_s[-4:]) * area_factors)

        # we set the post pulse phase encode gradient that sets up the next readout
        if np.abs(self.phase_areas[idx_phase]) > 1:
            block.grad_phase.amplitude[-3:-1] = - self.phase_areas[idx_phase] / phase_enc_time_post_pulse
        else:
            block.grad_phase.amplitude = np.zeros_like(block.grad_phase.amplitude)

    def _apply_slice_offset(self, idx_slice: int):
        for sbb in [self.block_excitation, self.block_refocus_first, self.block_refocus]:
            grad_slice_amplitude_hz = sbb.grad_slice.amplitude[sbb.grad_slice.t_array_s >= sbb.rf.t_delay_s][0]
            sbb.rf.freq_offset_hz = grad_slice_amplitude_hz * self.z[idx_slice]
            # we are setting the phase of a pulse here into its phase offset var.
            # To merge both: given phase parameter and any complex signal array data
            sbb.rf.phase_offset_rad = sbb.rf.phase_rad - 2 * np.pi * sbb.rf.freq_offset_hz * sbb.rf.calculate_center()

    def _write_sampling_pattern(self, echo_idx: int, phase_idx: int, slice_idx: int,
                                se_gre_type: str, num_scan: int = -1, se_gre_num: int = -1):
        if se_gre_type == "gre":
            if se_gre_num == 0:
                acq_type = "pf_acq"
            else:
                acq_type = "gre_us_sym"
        else:
            acq_type = "se_us_sym"
        sampling_index = {
            "num_scan": num_scan, "type": se_gre_type, "se_gre_num": se_gre_num,
            "slice_num": slice_idx, "pe_num": phase_idx, "echo_num": echo_idx, "acq_type": acq_type
        }
        self.sampling_pattern_constr.append(sampling_index)
        self.sampling_pattern_set = True
        return num_scan + 1, se_gre_num + 1

    def _add_gesse_readouts(self, idx_pe_loop: int, idx_slice_loop: int,
                            scan_idx: int, echo_se_idx: int, echo_gre_idx: int):
        # add gre sampling
        self.seq.ppSeq.add_block(*self.block_gre_acq.list_events_to_ns())
        # write sampling pattern
        scan_idx, echo_gre_idx = self._write_sampling_pattern(
            phase_idx=self.k_indexes[echo_se_idx, idx_pe_loop], echo_idx=echo_gre_idx + echo_se_idx,
            slice_idx=self.trueSliceNum[idx_slice_loop], num_scan=scan_idx,
            se_gre_type="gre", se_gre_num=echo_gre_idx
        )

        # add se sampling
        self.seq.ppSeq.add_block(*self.block_se_acq.list_events_to_ns())
        # write sampling pattern
        scan_idx, echo_se_idx = self._write_sampling_pattern(
            phase_idx=self.k_indexes[echo_se_idx, idx_pe_loop], echo_idx=echo_gre_idx + echo_se_idx,
            slice_idx=self.trueSliceNum[idx_slice_loop], num_scan=scan_idx,
            se_gre_type="se", se_gre_num=echo_se_idx
        )

        # add gre sampling
        self.seq.ppSeq.add_block(*self.block_gre_acq.list_events_to_ns())
        # write sampling pattern
        scan_idx, echo_gre_idx = self._write_sampling_pattern(
            phase_idx=self.k_indexes[echo_se_idx, idx_pe_loop], echo_idx=echo_gre_idx + echo_se_idx,
            slice_idx=self.trueSliceNum[idx_slice_loop], num_scan=scan_idx,
            se_gre_type="gre", se_gre_num=echo_gre_idx
        )
        return scan_idx, echo_se_idx, echo_gre_idx

    def _loop_lines(self):
        # through phase encodes
        line_bar = tqdm.trange(
            self.params.numberOfCentralLines + self.params.numberOfOuterLines, desc="phase encodes"
        )
        scan_idx = 0
        for idx_n in line_bar:  # We have N phase encodes for all ETL contrasts
            for idx_slice in range(self.params.resolutionNumSlices):
                echo_se_idx = 0
                echo_gre_idx = 0
                # apply slice offset for all kernels
                self._apply_slice_offset(idx_slice=idx_slice)

                # -- excitation --
                # looping through slices per phase encode, set phase encode for excitation
                self._set_phase_grad(phase_idx=idx_n, echo_idx=0, excitation=True)
                # add block
                self.seq.ppSeq.add_block(*self.block_excitation.list_events_to_ns())

                # 0th echo sampling
                self.seq.ppSeq.add_block(*self.block_pf_acquisition.list_events_to_ns())
                # write sampling pattern
                scan_idx, echo_gre_idx = self._write_sampling_pattern(
                    phase_idx=self.k_indexes[0, idx_n], echo_idx=echo_gre_idx + echo_se_idx,
                    slice_idx=self.trueSliceNum[idx_slice], num_scan=scan_idx,
                    se_gre_type="gre", se_gre_num=echo_gre_idx
                )
                # delay if necessary
                if self.t_delay_e0_ref1.get_duration() > 1e-7:
                    self.seq.ppSeq.add_block(self.t_delay_e0_ref1.to_simple_ns())

                # -- first refocus --
                # set flip angle from param list
                self._set_fa(rf_idx=0)
                # looping through slices per phase encode, set phase encode for ref 1
                self._set_phase_grad(phase_idx=idx_n, echo_idx=0)
                # add block
                self.seq.ppSeq.add_block(*self.block_refocus_first.list_events_to_ns())

                # delay if necessary
                if self.t_delay_ref1_se1.get_duration() > 1e-7:
                    self.seq.ppSeq.add_block(self.t_delay_ref1_se1.to_simple_ns())

                scan_idx, echo_se_idx, echo_gre_idx = self._add_gesse_readouts(
                    idx_pe_loop=idx_n, idx_slice_loop=idx_slice,
                    scan_idx=scan_idx, echo_se_idx=echo_se_idx, echo_gre_idx=echo_gre_idx)

                # successive double gre + mese in center
                for echo_idx in np.arange(1, self.params.ETL):
                    # set flip angle from param list
                    self._set_fa(rf_idx=echo_idx)
                    # looping through slices per phase encode, set phase encode for ref 1
                    self._set_phase_grad(phase_idx=idx_n, echo_idx=echo_idx)
                    # refocus
                    self.seq.ppSeq.add_block(*self.block_refocus.list_events_to_ns())

                    scan_idx, echo_se_idx, echo_gre_idx = self._add_gesse_readouts(
                        idx_pe_loop=idx_n, idx_slice_loop=idx_slice,
                        scan_idx=scan_idx, echo_se_idx=echo_se_idx, echo_gre_idx=echo_gre_idx
                    )

                # set phase encode of final spoiling grad
                self._set_end_spoil_phase_grad()
                # end with spoiling
                self.seq.ppSeq.add_block(*self.block_spoil_end.list_events_to_ns())
                # set slice delay
                self.seq.ppSeq.add_block(self.delay_slice.to_simple_ns())

    def simulate_grad_moments(self):
        log_module.info(f"simulating gradient moments")
        # set for how long to run through
        t_lim_ms = 110
        dt_steps = 10  # steps of us
        # build axis of length TR in steps of us
        ax = np.arange(t_lim_ms * 1e3 / dt_steps)
        t = 0
        # get gradient shapes
        grads = np.zeros((4, ax.shape[0]))  # grads [read, phase, slice, adc]
        # get seq data until defined length
        block_times = np.cumsum(self.seq.ppSeq.block_durations)
        end_id = np.where(block_times >= t_lim_ms * 1e-3)[0][0]
        for block_counter in range(end_id):
            block = self.seq.ppSeq.get_block(block_counter + 1)
            if getattr(block, "adc", None) is not None:  # ADC
                b_adc = block.adc
                # From Pulseq: According to the information from Klaus Scheffler and indirectly from Siemens this
                # is the present convention - the samples are shifted by 0.5 dwell
                t_start = t + int(1e6 * b_adc.delay / dt_steps)
                t_end = t_start + int(1e6 * b_adc.num_samples * b_adc.dwell / dt_steps)
                grads[3, t_start:t_end] = 1

            grad_channels = ["gx", "gy", "gz"]
            for x in range(len(grad_channels)):  # Gradients
                if getattr(block, grad_channels[x], None) is not None:
                    grad = getattr(block, grad_channels[x])
                    t_start = t + int(1e6 * grad.delay / dt_steps)
                    t_end = t_start + int(1e6 * grad.shape_dur / dt_steps)
                    grad_shape = np.interp(np.arange(t_end - t_start), 1e6 * grad.tt / dt_steps, grad.waveform)
                    grads[x, t_start:t_end] = grad_shape

            t += int(1e6 * self.seq.ppSeq.block_durations[block_counter] / dt_steps)

        # want to get the moments, basically just cumsum over the grads, multiplied by delta t = 5us
        grad_moments = np.copy(grads)
        grad_moments[:3] = np.cumsum(grads[:3], axis=1) * dt_steps * 1e-6
        # do lazy maximization to 2 for visual purpose, we are only interested in visualizing the drift
        grad_moments[:3] = 2 * grad_moments[:3] / np.max(np.abs(grad_moments[:3]), axis=1, keepdims=True)
        # want to plot the moments
        if self.seq.config.visualize:
            self._plot_grad_moments(grad_moments, dt_in_us=dt_steps)

    def _plot_grad_moments(self, grad_moments: np.ndarray, dt_in_us: int):
        ids = ["gx"] * grad_moments.shape[1] + ["gy"] * grad_moments.shape[1] + ["gz"] * grad_moments.shape[1] + \
              ["adc"] * grad_moments.shape[1]
        ax_time = np.tile(np.arange(grad_moments.shape[1]) * dt_in_us, 4)
        df = pd.DataFrame({
            "moments": grad_moments.flatten(), "id": ids,
            "time": ax_time
        })
        plotting.plot_grad_moments(mom_df=df, out_path=self.seq.config.outputPath, name="sim_moments")

    def _set_end_spoil_phase_grad(self):
        factor = np.array([0.5, 1.0, 0.5])

        # get phase moment of last phase encode
        pe_last_area = np.trapz(
            x=self.block_refocus.grad_phase.t_array_s[-4:],
            y=self.block_refocus.grad_phase.amplitude[-4:]
        )
        # adopt last grad to inverse area
        pe_end_times = self.block_spoil_end.grad_phase.t_array_s[-4:]
        delta_end_times = np.diff(pe_end_times)
        pe_end_amp = pe_last_area / np.sum(factor * delta_end_times)
        if np.abs(pe_end_amp) > self.system.max_grad:
            err = f"amplitude violation upon last pe grad setting"
            log_module.error(err)
            raise AttributeError(err)
        self.block_spoil_end.grad_phase.amplitude[1:3] = - pe_end_amp

    def _set_k_space_sampling_readout_patterns(self):
        # get all read - k - trajectories
        grad_pre_area = np.sum(self.block_excitation.grad_read.area)
        k_traj_pf = self.block_pf_acquisition.get_k_space_trajectory(
            pre_read_area=grad_pre_area, fs_grad_area=self.params.resolutionNRead * self.params.deltaK_read
        )
        k_traj_gre_us_sym = self.block_gre_acq.get_k_space_trajectory(
            pre_read_area=np.sum(self.block_refocus.grad_read.area) / 2,
            fs_grad_area=self.params.resolutionNRead * self.params.deltaK_read
        )
        pre_area_se = np.sum(self.block_refocus.grad_read.area) / 2 + np.trapz(
            x=self.block_gre_acq.grad_read.t_array_s, y=self.block_gre_acq.grad_read.amplitude
        )
        k_traj_se_us_sym = self.block_se_acq.get_k_space_trajectory(
            pre_read_area=pre_area_se,
            fs_grad_area=self.params.resolutionNRead * self.params.deltaK_read
        )

        k_traj = self.build_save_k_traj(
            k_trajs=[k_traj_pf, k_traj_se_us_sym, k_traj_gre_us_sym], labels=["pf_acq", "se_us_sym", "gre_us_sym"],
            n_samples_ref=int(self.params.resolutionNRead * self.params.oversampling)
        )
        # plot files for visualization
        if self.seq.config.visualize:
            plotting.plot_k_traj_pd(
                k_trajs_df=k_traj,
                out_path=self.seq.config.outputPath, name="k_space_trajectories"
            )


if __name__ == '__main__':
    seq_gv = VespaGerdSequence(seq_opts=options.Sequence())
    seq_gv.build()
    te = np.zeros(seq_gv.te.shape[0] + 1)
    te[1:] = seq_gv.te * 1e3
    print(te)
    print(np.diff(te))
