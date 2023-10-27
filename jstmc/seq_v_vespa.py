import pypsi
from jstmc import events, seq_baseclass
from jstmc.kernels import Kernel
import numpy as np
import logging
import tqdm

log_module = logging.getLogger(__name__)


class SeqVespaGerd(seq_baseclass.Sequence):
    def __init__(self, pypsi_params: pypsi.Params = pypsi.Params()):
        super().__init__(pypsi_params=pypsi_params)

        log_module.info(f"init vespa-gerd algorithm")
        self.num_gre_lobes: int = 1

        # timing
        self.t_delay_e0_ref1: events.DELAY = events.DELAY()
        self.t_delay_ref1_se1: events.DELAY = events.DELAY()

        # sbbs
        # partial fourier acquisition -> 0th echo
        self.block_pf_acquisition, grad_pre_area = Kernel.acquisition_pf_undersampled(
            pyp_interface=self.params, system=self.pp_sys
        )
        # add id
        self.id_pf_acq: str = "gre_us_pf"

        # # undersampled readout with symmetrical accelerated sidelobes
        # self.block_se_acq, self.acc_factor_us_read = Kernel.acquisition_sym_undersampled(
        #     pyp_interface=self.params, system=self.pp_sys
        # )
        # for now lets go with a fs readout, takes more time but for proof of concept easier
        self.block_se_acq = Kernel.acquisition_fs(
            pyp_params=self.params, system=self.pp_sys
        )
        # add id
        self.id_se_acq: str = "se_fs"

        # gradient echo readouts, always have inverted gradient direction wrt to se readouts,
        # - inverted gradient, k-space from right to left
        # self.block_gre_acq, _ = Kernel.acquisition_sym_undersampled(
        #     pyp_interface=self.params, system=self.pp_sys, invert_grad_dir=True
        # )
        self.block_gre_acq = Kernel.acquisition_fs(
            pyp_params=self.params, system=self.pp_sys
        )
        # invert gradient
        self.block_gre_acq.grad_read.area = - self.block_gre_acq.grad_read.area
        self.block_gre_acq.grad_read.amplitude = - self.block_gre_acq.grad_read.amplitude
        # add id
        self.id_gre_acq: str = "gre_fs"

        # spoiling at end of echo train
        self.block_spoil_end: Kernel = Kernel.spoil_all_grads(
            pyp_interface=self.params, system=self.pp_sys
        )
        self._mod_spoiling_end()

        # excitation pulse
        self.block_excitation = Kernel.excitation_slice_sel(
            pyp_interface=self.params, system=self.pp_sys, use_slice_spoiling=False
        )
        self._mod_excitation(grad_pre_area=grad_pre_area)

        # refocusing
        self.block_refocus, self.t_spoiling_pe = Kernel.refocus_slice_sel_spoil(
            pyp_interface=self.params, system=self.pp_sys, pulse_num=1, return_pe_time=True
        )
        # sanity check
        if np.abs(np.sum(self.block_se_acq.grad_read.area)) - np.abs(np.sum(self.block_gre_acq.grad_read.area)) > 1e-8:
            err = f"readout areas of gradient and spin echo readouts differ"
            log_module.error(err)
            raise ValueError(err)

        # for the first we need a different gradient rewind to rephase the partial fourier readout k-space travel
        self.block_refocus_1: Kernel = Kernel.copy(self.block_refocus)
        self._mod_first_refocus_rewind_0_echo(grad_pre_area=grad_pre_area)
        # need to adapt echo read prewinder and rewinder
        self._mod_block_prewind_echo_read(self.block_refocus_1)

        self._mod_block_rewind_echo_read(self.block_refocus)
        self._mod_block_prewind_echo_read(self.block_refocus)

        # plot files for visualization
        if self.params.visualize:
            self.block_excitation.plot(path=self.interface.config.output_path, name="excitation")
            self.block_refocus_1.plot(path=self.interface.config.output_path, name="refocus-1")
            self.block_refocus.plot(path=self.interface.config.output_path, name="refocus")
            self.block_pf_acquisition.plot(path=self.interface.config.output_path, name="partial-fourier-acqusisition")
            self.block_se_acq.plot(path=self.interface.config.output_path, name="se-acquisition")
            self.block_gre_acq.plot(path=self.interface.config.output_path, name="gre-acquisition")

        # register all slice select kernel pulse gradients
        self.kernel_pulses_slice_select = [self.block_excitation, self.block_refocus_1, self.block_refocus]

        # ToDo:
        # as is now all gesse readouts sample the same phase encode lines as the spin echoes.
        # this would allow joint recon of t2 and t2* contrasts independently
        # but we could also benefit even more from joint recon of all echoes and
        # hence switch up the phase encode scheme even further also in between gesse samplings

    # __ pypsi __
    # sampling + k-traj
    def _set_k_trajectories(self):
        # get all read - k - trajectories
        # get prephasing gradient area
        grad_pre_area = np.sum(self.block_excitation.grad_read.area)
        # calculate trajectory for pf readout
        self._register_k_trajectory(
            self.block_pf_acquisition.get_k_space_trajectory(
                pre_read_area=grad_pre_area, fs_grad_area=self.params.resolution_n_read * self.params.delta_k_read
            ),
            identifier=self.id_pf_acq
        )

        # calculate trajectory for gre readout, prephasing area = to refocus block read area half
        self._register_k_trajectory(
            self.block_gre_acq.get_k_space_trajectory(
                pre_read_area=np.sum(self.block_refocus.grad_read.area) / 2,
                fs_grad_area=self.params.resolution_n_read * self.params.delta_k_read
            ),
            identifier=self.id_gre_acq
        )
        # calculate trajectory for se readouts, prephasing is the prephase gre area + whole gre area
        pre_area_se = np.sum(self.block_refocus.grad_read.area) / 2 + np.sum(self.block_gre_acq.grad_read.area)
        self._register_k_trajectory(
            self.block_se_acq.get_k_space_trajectory(
                pre_read_area=pre_area_se,
                fs_grad_area=self.params.resolution_n_read * self.params.delta_k_read
            ),
            identifier=self.id_se_acq
        )

    # recon
    def _set_nav_parameters(self):
        # no navigators used
        pass

    # emc
    def _fill_emc_info(self):
        t_rephase = (self.block_excitation.get_duration() -
                     (self.block_excitation.rf.t_duration_s + self.block_excitation.rf.t_delay_s))
        amp_rephase = self.block_excitation.grad_slice.area[-1] / t_rephase
        self.interface.emc.gradientExcitationRephase = self._set_grad_for_emc(amp_rephase)
        self.interface.emc.durationExcitationRephase = t_rephase * 1e6

    def _mod_spoiling_end(self):
        # want to enable complete refocusing of read gradient when spoiling factor -0.5 is chosen in opts
        readout_area = np.trapz(
            x=self.block_gre_acq.grad_read.t_array_s,
            y=self.block_gre_acq.grad_read.amplitude
        )
        spoil_area = self.params.read_grad_spoiling_factor * readout_area
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
            channel=self.params.read_dir, system=self.pp_sys, area=-grad_pre_area,
            duration_s=rephasing_time,
            delay_s=self.block_excitation.rf.t_delay_s + self.block_excitation.rf.t_duration_s
        )
        grad_phase = events.GRAD.make_trapezoid(
            channel=self.params.phase_dir, system=self.pp_sys, area=-np.max(self.phase_areas),
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
        delta_times_first_grad_part = np.diff(self.block_refocus_1.grad_read.t_array_s[:4])
        # amplitude at trapezoid points is middle rectangle plus 2 ramp triangles
        amplitude = - area_to_rewind / np.sum(np.array([0.5, 1.0, 0.5]) * delta_times_first_grad_part)
        # check max grad violation
        if np.abs(amplitude) > self.pp_sys.max_grad:
            err = f"amplitude violation when rewinding 0 echo readout gradient"
            log_module.error(err)
            raise ValueError(err)
        # assign
        self.block_refocus_1.grad_read.amplitude[1:3] = amplitude
        self.block_refocus_1.grad_read.area[0] = np.trapz(
            x=self.block_refocus_1.grad_read.t_array_s[:4],
            y=self.block_refocus_1.grad_read.amplitude[:4]
        )

    def _mod_block_prewind_echo_read(self, sbb: Kernel):
        # need to prewind readout echo gradient
        area_read = np.sum(self.block_gre_acq.grad_read.area)
        area_prewind = - 0.5 * area_read
        delta_times_last_grad_part = np.diff(sbb.grad_read.t_array_s[-4:])
        amplitude = area_prewind / np.sum(np.array([0.5, 1.0, 0.5]) * delta_times_last_grad_part)
        if np.abs(amplitude) > self.pp_sys.max_grad:
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
        if np.abs(amplitude) > self.pp_sys.max_grad:
            err = f"amplitude violation when prewinding first echo readout gradient"
            log_module.error(err)
            raise ValueError(err)
        sbb.grad_read.amplitude[1:3] = amplitude
        sbb.grad_read.area[0] = area_rewind

    def _build_variant(self):
        log_module.info(f"build -- calculate minimum ESP")
        self._calculate_echo_timings()
        log_module.info(f"build -- calculate slice delay")
        self._calculate_slice_delay()

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
        max_num_slices = int(np.floor(self.params.tr / t_total_etl))
        log_module.info(f"\t\t-total echo train length: {t_total_etl:.2f} ms")
        log_module.info(f"\t\t-desired number of slices: {self.params.resolution_slice_num}")
        log_module.info(f"\t\t-possible number of slices within TR: {max_num_slices}")
        if self.params.resolution_slice_num > max_num_slices:
            time_missing = (self.params.resolution_slice_num - max_num_slices) * t_total_etl
            log_module.info(f"increase TR or Concatenation needed. - need {time_missing:.2f} ms more")
        self.delay_slice = events.DELAY.make_delay(
            1e-3 * (self.params.tr - self.params.resolution_slice_num * t_total_etl) /
            self.params.resolution_slice_num,
            system=self.pp_sys
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
                    self.block_refocus_1.get_duration() / 2

        # find time between mid refocus and first gre and first se
        t_ref1_gre1 = self.block_refocus_1.get_duration() / 2 + self.block_gre_acq.get_duration() / 2
        t_gre1_se1 = self.block_gre_acq.get_duration() / 2 + self.block_se_acq.get_duration() / 2

        # echo time of first se is twice the bigger time of 1) between excitation and first ref
        # 2) between first ref and se
        te_1 = 2 * np.max([t_exci_0e + t_0e_1ref, t_ref1_gre1 + t_gre1_se1])

        # time to either side between exxcitation - ref - se needs to be equal, calculate appropriate delays
        if t_exci_0e + t_0e_1ref < te_1 / 2:
            self.t_delay_e0_ref1 = events.DELAY.make_delay(te_1 / 2 - t_exci_0e - t_0e_1ref, system=self.pp_sys)
        if t_ref1_gre1 + t_gre1_se1 < te_1 / 2:
            self.t_delay_ref1_se1 = events.DELAY.make_delay(te_1 / 2 - t_ref1_gre1 - t_gre1_se1, system=self.pp_sys)

        # write echo times to array
        self.te.append(t_exci_0e)
        self.te.append(t_exci_0e + t_0e_1ref + self.t_delay_e0_ref1.get_duration() + \
                     self.t_delay_ref1_se1.get_duration() + t_ref1_gre1)
        self.te.append(te_1)
        self.te.append(te_1 + t_gre1_se1)
        for k in np.arange(4, self.params.etl * 3 + 1, 3):
            # take last echo time (gre sampling after se) need to add time from gre to rf and from rf to gre (equal)
            self.te.append(self.te[k - 1] + 2 * t_ref1_gre1)
            # take this time and add time between gre and se
            self.te.append(self.te[k] + t_gre1_se1)
            # and same amount again to arrive at gre sampling
            self.te.append(self.te[k + 1] + t_gre1_se1)
        te_print = [f'{1000*t:.2f}' for t in self.te]
        log_module.info(f"echo times: {te_print} ms")

    def _set_fa(self, rf_idx: int):
        # we take same kernels for different refocusing pulses when going through the sequence
        # want to adopt rf flip angle and phase based on given input parameters via options
        block = self._get_refocus_block_from_echo_idx(rf_idx=rf_idx)
        # calculate flip angle as given
        flip = block.rf.t_duration_s / block.rf.signal.shape[0] * np.sum(np.abs(block.rf.signal)) * 2 * np.pi
        # take flip angle in radiants from options
        fa_rad = self.params.refocusing_rf_rad_fa[rf_idx]
        # take phase as given in options
        phase_rad = self.params.refocusing_rf_rad_phase[rf_idx]
        # set block values
        block.rf.signal *= fa_rad / flip
        block.rf.phase_rad = phase_rad

    def _get_refocus_block_from_echo_idx(self, rf_idx: int) -> Kernel:
        # want to choose the rf based on position in echo train
        if rf_idx == 0:
            # first refocusing is different kernel
            block = self.block_refocus_1
        else:
            # we are on usual gesse echoes, past the first refocus
            block = self.block_refocus
        return block

    def _set_phase_grad(self, echo_idx: int, phase_idx: int, excitation: bool = False):
        # caution we assume trapezoidal phase encode gradients
        area_factors = np.array([0.5, 1.0, 0.5])
        # we get the actual line index from the sampling pattern, dependent on echo number and phase index in the loop
        idx_phase = self.k_pe_indexes[echo_idx, phase_idx]
        # additionally we need the last blocks phase encode for rephasing
        if echo_idx > 0:
            # if we are not on the first readout:
            # we need the last phase encode value to reset before refocusing
            last_idx_phase = self.k_pe_indexes[echo_idx - 1, phase_idx]
        else:
            # we need the phase encode from the 0th echo, as is now it is also encoded like the refocused se readout
            last_idx_phase = self.k_pe_indexes[echo_idx, phase_idx]

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

    def _add_gesse_readouts(self, idx_pe_loop: int, idx_slice_loop: int,
                            scan_idx: int, echo_se_idx: int, echo_gre_idx: int):
        # phase encodes are set equal for all 3 readouts and follow the central se echo idx
        phase_encode_echo = echo_se_idx
        # add gre sampling
        self.pp_seq.add_block(*self.block_gre_acq.list_events_to_ns())
        # write sampling pattern
        scan_idx, echo_gre_idx = self._write_sampling_pattern_entry(scan_num=scan_idx,
                                                                    slice_num=self.trueSliceNum[idx_slice_loop],
                                                                    pe_num=int(self.k_pe_indexes[
                                                                        phase_encode_echo, idx_pe_loop]),
                                                                    echo_num=echo_gre_idx + echo_se_idx,
                                                                    acq_type=self.id_gre_acq, echo_type="gre",
                                                                    echo_type_num=echo_gre_idx)

        # add se sampling
        self.pp_seq.add_block(*self.block_se_acq.list_events_to_ns())
        # write sampling pattern
        scan_idx, echo_se_idx = self._write_sampling_pattern_entry(scan_num=scan_idx,
                                                                   slice_num=self.trueSliceNum[idx_slice_loop],
                                                                   pe_num=int(self.k_pe_indexes[
                                                                       phase_encode_echo, idx_pe_loop]),
                                                                   echo_num=echo_gre_idx + echo_se_idx,
                                                                   acq_type=self.id_se_acq, echo_type="se",
                                                                   echo_type_num=echo_se_idx)

        # add gre sampling
        self.pp_seq.add_block(*self.block_gre_acq.list_events_to_ns())
        # write sampling pattern
        scan_idx, echo_gre_idx = self._write_sampling_pattern_entry(scan_num=scan_idx,
                                                                    slice_num=self.trueSliceNum[idx_slice_loop],
                                                                    pe_num=int(self.k_pe_indexes[
                                                                        phase_encode_echo, idx_pe_loop]),
                                                                    echo_num=echo_gre_idx + echo_se_idx,
                                                                    acq_type=self.id_gre_acq, echo_type="gre",
                                                                    echo_type_num=echo_gre_idx)
        return scan_idx, echo_se_idx, echo_gre_idx

    def _loop_lines(self):
        # through phase encodes
        line_bar = tqdm.trange(
            self.params.number_central_lines + self.params.number_outer_lines, desc="phase encodes"
        )
        scan_idx = 0
        for idx_n in line_bar:  # We have N phase encodes for all ETL contrasts
            for idx_slice in range(self.params.resolution_slice_num):
                echo_se_idx = 0
                echo_gre_idx = 0
                # apply slice offset for all kernels
                self._apply_slice_offset(idx_slice=idx_slice)

                # -- excitation --
                # looping through slices per phase encode, set phase encode for excitation
                self._set_phase_grad(phase_idx=idx_n, echo_idx=0, excitation=True)
                # add block
                self.pp_seq.add_block(*self.block_excitation.list_events_to_ns())

                # 0th echo sampling
                self.pp_seq.add_block(*self.block_pf_acquisition.list_events_to_ns())
                # write sampling pattern
                scan_idx, echo_gre_idx = self._write_sampling_pattern_entry(scan_num=scan_idx,
                                                                            slice_num=self.trueSliceNum[idx_slice],
                                                                            pe_num=int(self.k_pe_indexes[0, idx_n]),
                                                                            echo_num=echo_gre_idx + echo_se_idx,
                                                                            acq_type=self.id_pf_acq, echo_type="gre",
                                                                            echo_type_num=echo_gre_idx)
                # delay if necessary
                if self.t_delay_e0_ref1.get_duration() > 1e-7:
                    self.pp_seq.add_block(self.t_delay_e0_ref1.to_simple_ns())

                # -- first refocus --
                # set flip angle from param list
                self._set_fa(rf_idx=0)
                # looping through slices per phase encode, set phase encode for ref 1
                self._set_phase_grad(phase_idx=idx_n, echo_idx=0)
                # add block
                self.pp_seq.add_block(*self.block_refocus_1.list_events_to_ns())

                # delay if necessary
                if self.t_delay_ref1_se1.get_duration() > 1e-7:
                    self.pp_seq.add_block(self.t_delay_ref1_se1.to_simple_ns())

                scan_idx, echo_se_idx, echo_gre_idx = self._add_gesse_readouts(
                    idx_pe_loop=idx_n, idx_slice_loop=idx_slice,
                    scan_idx=scan_idx, echo_se_idx=echo_se_idx, echo_gre_idx=echo_gre_idx)

                # successive double gre + mese in center
                for echo_idx in np.arange(1, self.params.etl):
                    # set flip angle from param list
                    self._set_fa(rf_idx=echo_idx)
                    # looping through slices per phase encode, set phase encode for ref 1
                    self._set_phase_grad(phase_idx=idx_n, echo_idx=echo_idx)
                    # refocus
                    self.pp_seq.add_block(*self.block_refocus.list_events_to_ns())

                    scan_idx, echo_se_idx, echo_gre_idx = self._add_gesse_readouts(
                        idx_pe_loop=idx_n, idx_slice_loop=idx_slice,
                        scan_idx=scan_idx, echo_se_idx=echo_se_idx, echo_gre_idx=echo_gre_idx
                    )

                # set phase encode of final spoiling grad
                self._set_end_spoil_phase_grad()
                # end with spoiling
                self.pp_seq.add_block(*self.block_spoil_end.list_events_to_ns())
                # set slice delay
                self.pp_seq.add_block(self.delay_slice.to_simple_ns())

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
        if np.abs(pe_end_amp) > self.pp_sys.max_grad:
            err = f"amplitude violation upon last pe grad setting"
            log_module.error(err)
            raise AttributeError(err)
        self.block_spoil_end.grad_phase.amplitude[1:3] = - pe_end_amp


if __name__ == '__main__':
    pass
