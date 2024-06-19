import pypsi
from jstmc import events, kernels, seq_baseclass
import numpy as np
import logging
import tqdm

log_module = logging.getLogger(__name__)


class SeqJstmc(seq_baseclass.Sequence):
    def __init__(self, pypsi_params: pypsi.Params = pypsi.Params()):
        super().__init__(pypsi_params=pypsi_params)
        log_module.info(f"init jstmc algorithm")

        # timing
        self.esp: float = 0.0
        self.delay_exci_ref1: events.DELAY = events.DELAY()
        self.delay_ref_adc: events.DELAY = events.DELAY()
        self.phase_enc_time: float = 0.0
        self.delay_slice: events.DELAY = events.DELAY()

        # sbbs
        self.block_acquisition: kernels.Kernel = kernels.Kernel.acquisition_fs(
            pyp_params=self.params,
            system=self.pp_sys
        )
        self.id_acq_se = "fs_acq"

        self.block_refocus, self.phase_enc_time = kernels.Kernel.refocus_slice_sel_spoil(
            pyp_interface=self.params,
            system=self.pp_sys,
            pulse_num=1,
            return_pe_time=True,
            read_gradient_to_prephase=self.block_acquisition.grad_read.area / 2
        )

        self.block_refocus_1: kernels.Kernel = kernels.Kernel.refocus_slice_sel_spoil(
            pyp_interface=self.params,
            system=self.pp_sys,
            pulse_num=0,
            read_gradient_to_prephase=self.block_acquisition.grad_read.area / 2
        )
        # calculate ramp area to slice select upon refocusing 1
        ramp_area_ref_1 = np.trapz(
            x=self.block_refocus_1.grad_slice.t_array_s[:2],
            y=self.block_refocus_1.grad_slice.amplitude[:2]
        )

        self.block_excitation = kernels.Kernel.excitation_slice_sel(
            pyp_interface=self.params,
            system=self.pp_sys,
            adjust_ramp_area=ramp_area_ref_1
        )
        self.block_spoil_end: kernels.Kernel = kernels.Kernel.spoil_all_grads(
            pyp_interface=self.params,
            system=self.pp_sys
        )

        # set resolution downgrading for fid navs
        self.nav_num = 2
        # set so the in plane reso is 3 mm
        self.nav_resolution_defactor: float = self.params.resolution_slice_thickness / 3

        self.block_excitation_nav: kernels.Kernel = self._set_excitation_fid_nav()
        self.block_list_fid_nav_acq: list = self._set_acquisition_fid_nav()
        self.id_acq_nav = "nav_acq"

        if self.params.visualize:
            self.block_excitation.plot(path=self.interface.config.output_path, name="excitation")
            self.block_refocus_1.plot(path=self.interface.config.output_path, name="refocus_1")
            self.block_refocus.plot(path=self.interface.config.output_path, name="refocus")
            self.block_excitation_nav.plot(path=self.interface.config.output_path, name="nav_excitation")
            self.block_acquisition.plot(path=self.interface.config.output_path, name="fs_acquisition")
            for k in range(3):
                self.block_list_fid_nav_acq[k].plot(path=self.interface.config.output_path, name=f"nav_acq_{k}")

        # register slice select pulse grad kernels
        self.kernel_pulses_slice_select = [self.block_excitation, self.block_refocus_1, self.block_refocus]

    # __ pypsi __
    # sampling + k traj
    def _set_k_trajectories(self):
        # read direction is always fully oversampled, no trajectories to register
        grad_pre_area = float(np.sum(self.block_refocus.grad_read.area) / 2)
        # calculate trajectory for pf readout
        self._register_k_trajectory(
            self.block_acquisition.get_k_space_trajectory(
                pre_read_area=grad_pre_area, fs_grad_area=self.params.resolution_n_read * self.params.delta_k_read
            ),
            identifier=self.id_acq_se
        )
        # need 2 trajectory lines for navigators: plus + minus directions
        # sanity check that pre-phasing for odd and even read lines are same, i.e. cycling correct
        grad_read_exc_pre = np.sum(self.block_excitation_nav.grad_read.area)
        grad_read_2nd_pre = grad_read_exc_pre + np.sum(
            self.block_list_fid_nav_acq[0].grad_read.area
        )
        grad_read_3rd_pre = grad_read_2nd_pre + np.sum(self.block_list_fid_nav_acq[1].grad_read.area)
        grad_read_4th_pre = grad_read_3rd_pre + np.sum(
            self.block_list_fid_nav_acq[2].grad_read.area
        )
        if np.abs(grad_read_exc_pre - grad_read_3rd_pre) > 1e-9:
            err = f"navigator readout prephasing gradients of odd echoes do not coincide"
            log_module.error(err)
            raise ValueError(err)
        if np.abs(grad_read_2nd_pre - grad_read_4th_pre) > 1e-9:
            err = f"navigator readout prephasing gradients of even echoes do not coincide"
            log_module.error(err)
            raise ValueError(err)
        # register trajectories
        # odd
        acq_nav_block = self.block_list_fid_nav_acq[0]
        self._register_k_trajectory(
            acq_nav_block.get_k_space_trajectory(
                pre_read_area=grad_read_exc_pre,
                fs_grad_area=self.params.resolution_n_read * self.params.delta_k_read * self.nav_resolution_defactor
            ),
            identifier=f"{self.id_acq_nav}_odd"
        )
        # even
        acq_nav_block = self.block_list_fid_nav_acq[1]
        self._register_k_trajectory(
            acq_nav_block.get_k_space_trajectory(
                pre_read_area=grad_read_2nd_pre,
                fs_grad_area=self.params.resolution_n_read * self.params.delta_k_read * self.nav_resolution_defactor
            ),
            identifier=f"{self.id_acq_nav}_even"
        )

    # recon
    def _set_nav_parameters(self):
        self.interface.recon.set_navigator_params(
            lines_per_nav=int(self.params.resolution_n_phase * self.nav_resolution_defactor / 2),
            num_of_nav=self.params.number_central_lines + self.params.number_outer_lines,
            nav_acc_factor=2, nav_resolution_scaling=self.nav_resolution_defactor,
            num_of_navs_per_tr=2
        )

    # emc
    def _fill_emc_info(self):
        t_rephase = (self.block_excitation.get_duration() -
                     (self.block_excitation.rf.t_duration_s + self.block_excitation.rf.t_delay_s))
        amp_rephase = self.block_excitation.grad_slice.area[-1] / t_rephase
        self.interface.emc.gradient_excitation_rephase = self._set_grad_for_emc(amp_rephase)
        self.interface.emc.duration_excitation_rephase = t_rephase * 1e6
        self.interface.emc.duration_crush = self.phase_enc_time * 1e6
        # etl left unchanged

    # __ private __
    def _set_acquisition_fid_nav(self) -> list:
        # want to use an EPI style readout with acceleration. i.e. skipping of every other line.
        acceleration_factor = 2
        # want to go center out. i.e:
        # acquire line [0, 1, -2, 3, -4, 5 ...] etc i.e. acc_factor_th of the lines + 1,
        pe_increments = np.arange(
            1, int(self.params.resolution_n_phase * self.nav_resolution_defactor), acceleration_factor
        )
        pe_increments *= np.power(-1, np.arange(pe_increments.shape[0]))
        # in general only nth of resolution
        block_fid_nav = [kernels.Kernel.acquisition_fid_nav(
            pyp_interface=self.params,
            system=self.pp_sys,
            line_num=k,
            reso_degrading=self.nav_resolution_defactor
        ) for k in range(int(self.params.resolution_n_phase * self.nav_resolution_defactor / 2))]
        # add spoiling
        block_fid_nav.append(self.block_spoil_end)
        # add delay
        block_fid_nav.append(kernels.Kernel(system=self.pp_sys, delay=events.DELAY.make_delay(delay_s=10e-3)))
        return block_fid_nav

    def _set_excitation_fid_nav(self) -> kernels.Kernel:
        # use excitation kernel without spoiling
        k_ex = kernels.Kernel.excitation_slice_sel(
            pyp_interface=self.params,
            system=self.pp_sys,
            use_slice_spoiling=False
        )
        # set up prephasing gradient for fid readouts
        # get timings
        t_spoiling = np.sum(np.diff(k_ex.grad_slice.t_array_s[-4:]))
        t_spoiling_start = k_ex.grad_slice.t_array_s[-4]
        # get area
        num_samples_per_read = int(self.params.resolution_n_read * self.nav_resolution_defactor)
        grad_read_area = events.GRAD.make_trapezoid(
            channel=self.params.read_dir, system=self.pp_sys,
            flat_area=num_samples_per_read * self.params.delta_k_read,
            flat_time=self.params.dwell * num_samples_per_read * self.params.oversampling
        ).area
        # need half of this area (includes ramps etc)
        grad_read_pre = events.GRAD.make_trapezoid(
            channel=self.params.read_dir, system=self.pp_sys, area=-grad_read_area / 2,
            duration_s=float(t_spoiling), delay_s=t_spoiling_start
        )
        k_ex.grad_read = grad_read_pre
        return k_ex

    def _calculate_min_esp(self):
        # calculate time between midpoints
        t_exci_ref = self.block_refocus_1.rf.t_delay_s + self.block_refocus_1.rf.t_duration_s / 2 + \
                     self.block_excitation.get_duration() - self.block_excitation.rf.t_delay_s - \
                     self.block_excitation.rf.t_duration_s / 2
        t_ref_1_adc = self.block_refocus_1.get_duration() - self.block_refocus_1.rf.t_delay_s - \
                      self.block_refocus_1.rf.t_duration_s / 2 + self.block_acquisition.get_duration() / 2
        t_ref_2_adc = self.block_acquisition.get_duration() / 2 + self.block_refocus.get_duration() / 2

        self.params.esp = 2 * np.max([t_exci_ref, t_ref_1_adc, t_ref_2_adc]) * 1e3
        log_module.info(f"\t\t-found minimum ESP: {self.params.esp:.2f} ms")

        if np.abs(t_ref_1_adc - t_ref_2_adc) > 1e-6:
            log_module.error(f"refocus to adc timing different from adc to refocus. Systematic error in seq. creation")
        t_half_esp = self.params.esp * 1e-3 / 2
        # add delays
        if t_exci_ref < t_half_esp:
            self.delay_exci_ref1 = events.DELAY.make_delay(t_half_esp - t_exci_ref, system=self.pp_sys)
            if not self.delay_exci_ref1.check_on_block_raster():
                err = f"exci ref delay not on block raster"
                log_module.error(err)
        if t_ref_1_adc < t_half_esp:
            self.delay_ref_adc = events.DELAY.make_delay(t_half_esp - t_ref_1_adc, system=self.pp_sys)
            if not self.delay_ref_adc.check_on_block_raster():
                err = f"adc ref delay not on block raster"
                log_module.error(err)
        tes = np.arange(1, self.params.etl + 1) * self.params.esp
        self.te = tes.tolist()

    def _calculate_slice_delay(self):
        # time per echo train
        t_pre_etl = self.block_excitation.rf.t_delay_s + self.block_excitation.rf.t_duration_s / 2
        t_etl = self.params.etl * self.params.esp * 1e-3  # esp in ms
        t_post_etl = self.block_acquisition.get_duration() / 2 + self.block_spoil_end.get_duration()

        t_total_etl = t_pre_etl + t_etl + t_post_etl
        # time for fid navs - one delay in between
        t_total_fid_nav = np.sum(
            [b.get_duration() for b in self.block_list_fid_nav_acq]
        ) + np.sum(
            [b.get_duration() for b in self.block_list_fid_nav_acq[:-1]]
        )
        log_module.info(f"\t\t-total fid-nav time (2 navs + 1 delay of 10ms): {t_total_fid_nav * 1e3:.2f} ms")
        # deminish TR by FIDnavs
        tr_eff = self.params.tr * 1e-3 - t_total_fid_nav
        max_num_slices = int(np.floor(tr_eff / t_total_etl))
        log_module.info(f"\t\t-total echo train length: {t_total_etl * 1e3:.2f} ms")
        log_module.info(f"\t\t-desired number of slices: {self.params.resolution_slice_num}")
        log_module.info(f"\t\t-possible number of slices within TR: {max_num_slices}")
        if self.params.resolution_slice_num > max_num_slices:
            log_module.info(f"increase TR or Concatenation needed")
        # we want to add a delay additionally after fid nav block
        self.delay_slice = events.DELAY.make_delay(
            (tr_eff - self.params.resolution_slice_num * t_total_etl) / (self.params.resolution_slice_num + 1),
            system=self.pp_sys
        )
        log_module.info(f"\t\t-time between slices: {self.delay_slice.get_duration() * 1e3:.2f} ms")
        if not self.delay_slice.check_on_block_raster():
            self.delay_slice.set_on_block_raster()
            log_module.info(f"\t\t-adjusting TR delay to raster time: {self.delay_slice.get_duration() * 1e3:.2f} ms")

    def _build_variant(self):
        log_module.info(f"build -- calculate minimum ESP")
        self._calculate_min_esp()
        log_module.info(f"build -- calculate slice delay")
        self._calculate_slice_delay()

    def _loop_slices(self, idx_pe_n: int, no_adc: bool = False):
        # adc
        if no_adc:
            aq_block = self.block_acquisition.copy()
            aq_block.adc = events.ADC()
        else:
            aq_block = self.block_acquisition
        for idx_slice in np.arange(-1, self.params.resolution_slice_num):
            # one loop as intro (-1)
            self._set_fa(echo_idx=0, slice_idx=idx_slice, excitation=True)
            self._set_fa(echo_idx=0, slice_idx=idx_slice)
            # looping through slices per phase encode
            self._set_phase_grad(phase_idx=idx_pe_n, echo_idx=0)
            # apply slice offset
            self._apply_slice_offset(idx_slice=idx_slice)

            # excitation
            # add block
            self.pp_seq.add_block(*self.block_excitation.list_events_to_ns())

            # delay if necessary
            if self.delay_exci_ref1.get_duration() > 1e-7:
                self.pp_seq.add_block(self.delay_exci_ref1.to_simple_ns())

            # first refocus
            # add block
            self.pp_seq.add_block(*self.block_refocus_1.list_events_to_ns())

            # delay if necessary
            if self.delay_ref_adc.get_duration() > 1e-7:
                self.pp_seq.add_block(self.delay_ref_adc.to_simple_ns())

            # adc
            self.pp_seq.add_block(*aq_block.list_events_to_ns())
            if not no_adc:
                # write sampling pattern
                _ = self._write_sampling_pattern_entry(
                    slice_num=self.trueSliceNum[idx_slice],
                    pe_num=int(self.k_pe_indexes[0, idx_pe_n]), echo_num=0,
                    acq_type=self.id_acq_se
                )

            # delay if necessary
            if self.delay_ref_adc.get_duration() > 1e-7:
                self.pp_seq.add_block(self.delay_ref_adc.to_simple_ns())

            # loop
            for echo_idx in np.arange(1, self.params.etl):
                # set fa
                self._set_fa(echo_idx=echo_idx, slice_idx=idx_slice)
                # set phase
                self._set_phase_grad(echo_idx=echo_idx, phase_idx=idx_pe_n)
                # add block
                self.pp_seq.add_block(*self.block_refocus.list_events_to_ns())
                # delay if necessary
                if self.delay_ref_adc.get_duration() > 1e-7:
                    self.pp_seq.add_block(self.delay_ref_adc.to_simple_ns())

                # adc
                self.pp_seq.add_block(*aq_block.list_events_to_ns())
                if not no_adc:
                    # write sampling pattern
                    _ = self._write_sampling_pattern_entry(
                        slice_num=self.trueSliceNum[idx_slice],
                        pe_num=int(self.k_pe_indexes[echo_idx, idx_pe_n]),
                        echo_num=echo_idx, acq_type=self.id_acq_se
                    )

                # delay if necessary
                if self.delay_ref_adc.get_duration() > 1e-7:
                    self.pp_seq.add_block(self.delay_ref_adc.to_simple_ns())
            # spoil end
            self._set_end_spoil_phase_grad()
            self.pp_seq.add_block(*self.block_spoil_end.list_events_to_ns())
            # insert slice delay
            self.pp_seq.add_block(self.delay_slice.to_simple_ns())

    def _loop_navs(self):
        # navigators
        for nav_idx in range(self.nav_num):
            self._apply_slice_offset_fid_nav(idx_nav=nav_idx)
            # excitation
            # add block
            self.pp_seq.add_block(*self.block_excitation_nav.list_events_to_ns())
            # epi style nav read
            # we set up a counter to track the phase encode line, k-space center is half of num lines
            line_counter = 0
            central_line = int(self.params.resolution_n_phase * self.nav_resolution_defactor / 2) - 1
            # we set up the pahse encode increments
            pe_increments = np.arange(1, int(self.params.resolution_n_phase * self.nav_resolution_defactor), 2)
            pe_increments *= np.power(-1, np.arange(pe_increments.shape[0]))
            # we loop through all fid nav blocks (whole readout)
            for b_idx in range(self.block_list_fid_nav_acq.__len__()):
                # get the block
                b = self.block_list_fid_nav_acq[b_idx]
                # if at the end we add a delay
                if (nav_idx == 1) & (b_idx == self.block_list_fid_nav_acq.__len__() - 1):
                    self.pp_seq.add_block(self.delay_slice.to_simple_ns())
                # otherwise we add the block
                else:
                    self.pp_seq.add_block(*b.list_events_to_ns())
                # if we have a readout we write to sampling pattern file
                # for navigators we want the 0th to have identifier 0, all minus directions have 1, all plus have 2
                if b_idx % 2:
                    nav_ident = "odd"
                else:
                    nav_ident = "even"
                if b.adc.get_duration() > 0:
                    # track which line we are writing from the incremental steps
                    nav_line_pe = np.sum(pe_increments[:line_counter]) + central_line
                    _ = self._write_sampling_pattern_entry(
                        slice_num=nav_idx, pe_num=nav_line_pe, echo_num=0,
                        acq_type=f"{self.id_acq_nav}_{nav_ident}",
                        echo_type="gre-fid", nav_acq=True
                    )
                    line_counter += 1

    def _loop_lines(self):
        # through phase encodes
        line_bar = tqdm.trange(
            self.params.number_central_lines + self.params.number_outer_lines, desc="phase encodes"
        )
        # one slice loop for introduction
        _ = self._loop_slices(idx_pe_n=0, no_adc=True)
        # counter for number of scan
        for idx_n in line_bar:  # We have N phase encodes for all ETL contrasts
            self._loop_slices(idx_pe_n=idx_n)
            self._loop_navs()

        log_module.info(f"sequence built!")

    def _set_fa(self, echo_idx: int, slice_idx: int, excitation: bool = False):
        if excitation:
            sbb = self.block_excitation
            fa_rad = self.params.excitation_rf_rad_fa
            phase_rad = self.params.excitation_rf_rad_phase
        else:
            if echo_idx == 0:
                sbb = self.block_refocus_1
            else:
                sbb = self.block_refocus
            fa_rad = self.params.refocusing_rf_rad_fa[echo_idx]
            phase_rad = self.params.refocusing_rf_rad_phase[echo_idx]
        flip = sbb.rf.t_duration_s / sbb.rf.signal.shape[0] * np.sum(np.abs(sbb.rf.signal)) * 2 * np.pi
        # slice adaptive fa scaling - we need true slice position here!
        sbb.rf.signal *= fa_rad / flip * self.rf_slice_adaptive_scaling[self.trueSliceNum[slice_idx]]
        sbb.rf.phase_rad = phase_rad

    def _set_phase_grad(self, echo_idx: int, phase_idx: int):
        # phase gradient placement maximum time was calculated for whole area, this is reflected in self.phase_enc_time
        # however this includes ramps. Hence we need to calculate the amplitudes with the ramps in mind
        area_factors = np.array([0.5, 1.0, 0.5])
        # first get the phase index from the sampling scheme
        idx_phase = self.k_pe_indexes[echo_idx, phase_idx]
        # thens et the block we need to change dependent on the echo index
        if echo_idx == 0:
            sbb = self.block_refocus_1
        else:
            sbb = self.block_refocus
            # if we are further in the echo train we also need to rewind the previous phase encode.
            # get the last phase encode index
            last_idx_phase = self.k_pe_indexes[echo_idx - 1, phase_idx]
            # get the effective phase encode time
            t_pe = np.sum(np.diff(sbb.grad_phase.t_array_s[:4]) * area_factors)
            sbb.grad_phase.amplitude[1:3] = self.phase_areas[last_idx_phase] / t_pe
            
        # set the phase encode into the gradient of the refocussing kernel while slice spoil takes place,
        # default phase before read to -, after read to +
        if np.abs(self.phase_areas[idx_phase]) > 1:
            t_pe = np.sum(np.diff(sbb.grad_phase.t_array_s[-4:]) * area_factors)
            sbb.grad_phase.amplitude[-3:-1] = - self.phase_areas[idx_phase] / t_pe
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
        if np.abs(pe_end_amp) > self.pp_sys.max_grad:
            err = f"amplitude violation upon last pe grad setting"
            log_module.error(err)
            raise AttributeError(err)
        self.block_spoil_end.grad_phase.amplitude[1:3] = - pe_end_amp

    def _apply_slice_offset_fid_nav(self, idx_nav: int):
        sbb = self.block_excitation_nav
        grad_slice_amplitude_hz = sbb.grad_slice.amplitude[sbb.grad_slice.t_array_s >= sbb.rf.t_delay_s][0]
        # want to set the navs outside of the slice profile with equal distance to the rest of slices
        if idx_nav == 0:
            # first nav below slice slab
            z = np.min(self.z) - np.abs(np.diff(self.z)[0])
        elif idx_nav == 1:
            # second nav above slice slab
            z = np.max(self.z) + np.abs(np.diff(self.z)[0])
        else:
            err = f"sequence setup for only 2 navigators outside slice slab, " \
                  f"index {idx_nav} was given (should be 0 or 1)"
            log_module.error(err)
            raise ValueError(err)
        sbb.rf.freq_offset_hz = grad_slice_amplitude_hz * z
        # we are setting the phase of a pulse here into its phase offset var.
        # To merge both: given phase parameter and any complex signal array data
        sbb.rf.phase_offset_rad = sbb.rf.phase_rad - 2 * np.pi * sbb.rf.freq_offset_hz * sbb.rf.t_mid


if __name__ == '__main__':
    pass
