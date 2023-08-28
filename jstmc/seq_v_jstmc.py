import pypsi
from jstmc import events, kernels, seq_baseclass
import numpy as np
import logging
import tqdm

logModule = logging.getLogger(__name__)


class SeqJstmc(seq_baseclass.Sequence):
    def __init__(self, pypsi_params: pypsi.Params = pypsi.Params()):
        super().__init__(pypsi_params=pypsi_params)
        logModule.info(f"init jstmc algorithm")

        # timing
        self.esp: float = 0.0
        self.delay_exci_ref1: events.DELAY = events.DELAY()
        self.delay_ref_adc: events.DELAY = events.DELAY()
        self.phase_enc_time: float = 0.0
        self.delay_slice: events.DELAY = events.DELAY()

        # sbbs
        self.block_refocus_1: kernels.Kernel = kernels.Kernel.refocus_slice_sel_spoil(
            pyp_interface=self.params,
            system=self.pp_sys,
            pulse_num=0
        )
        ramp_area_ref_1 = self.block_refocus_1.grad_slice.t_array_s[1] * self.block_refocus_1.grad_slice.amplitude[
            1] / 2.0

        self.block_excitation = kernels.Kernel.excitation_slice_sel(
            pyp_interface=self.params,
            system=self.pp_sys,
            adjust_ramp_area=ramp_area_ref_1
        )

        self.block_acquisition: kernels.Kernel = kernels.Kernel.acquisition_fs(
            pyp_interface=self.params,
            system=self.pp_sys
        )

        self.block_refocus, self.phase_enc_time = kernels.Kernel.refocus_slice_sel_spoil(
            pyp_interface=self.params,
            system=self.pp_sys,
            pulse_num=1,
            return_pe_time=True
        )

        self.block_spoil_end: kernels.Kernel = kernels.Kernel.spoil_all_grads(
            pyp_interface=self.params,
            system=self.pp_sys
        )

        # set resolution downgrading for fid navs
        self.nav_num = 2
        # set so the in plane reso is 3 mm
        self.nav_resolution_defactor: float = self.params.resolutionSliceThickness / 3

        self.block_excitation_nav: kernels.Kernel = self._set_excitation_fid_nav()
        self.block_list_fid_nav_acq: list = self._set_acquisition_fid_nav()

        if self.params.visualize:
            self.block_excitation.plot(path=self.interface.config.output_path, name="excitation")
            self.block_refocus_1.plot(path=self.interface.config.output_path, name="refocus_1")
            self.block_refocus.plot(path=self.interface.config.output_path, name="refocus")
            self.block_excitation_nav.plot(path=self.interface.config.output_path, name="nav_excitation")
            for k in range(5):
                self.block_list_fid_nav_acq[k].plot(path=self.interface.config.output_path, name=f"nav_acq_{k}")

    # __ pypsi __
    # sampling + k traj
    def _set_k_trajectories(self):
        # read direction is always fully oversampled, no trajectories to register
        pass

    # recon
    def _set_nav_parameters(self):
        self.interface.recon.set_navigator_params(
            lines_per_nav=int(self.params.resolutionNPhase * self.nav_resolution_defactor / 2),
            num_of_nav=self.params.numberOfCentralLines + self.params.numberOfOuterLines,
            nav_acc_factor=2, nav_resolution_scaling=self.nav_resolution_defactor
        )

    # emc
    def _fill_emc_info(self):
        t_rephase = (self.block_excitation.get_duration() -
                     (self.block_excitation.rf.t_duration_s + self.block_excitation.rf.t_delay_s))
        amp_rephase = self.block_excitation.grad_slice.area[-1] / t_rephase
        self.interface.emc.gradientExcitationRephase = self._set_grad_for_emc(amp_rephase)
        self.interface.emc.durationExcitationRephase = t_rephase * 1e6
        self.interface.emc.durationCrush = self.phase_enc_time * 1e6

    # __ private __
    def _set_acquisition_fid_nav(self) -> list:
        # want to use an EPI style readout with acceleration. i.e. skipping of every other line.
        acceleration_factor = 2
        # want to go center out. i.e:
        # acquire line [0, 1, -2, 3, -4, 5 ...] etc i.e. acc_factor_th of the lines + 1,
        pe_increments = np.arange(
            1, int(self.params.resolutionNPhase * self.nav_resolution_defactor), acceleration_factor
        )
        pe_increments *= np.power(-1, np.arange(pe_increments.shape[0]))
        # in general only nth of resolution
        block_fid_nav = [kernels.Kernel.acquisition_fid_nav(
            pyp_interface=self.params,
            system=self.pp_sys,
            line_num=k,
            reso_degrading=self.nav_resolution_defactor
        ) for k in range(int(self.params.resolutionNPhase * self.nav_resolution_defactor / 2))]
        # add spoiling
        block_fid_nav.append(self.block_spoil_end)
        # add delay
        block_fid_nav.append(kernels.Kernel(system=self.pp_sys, delay=events.DELAY.make_delay(delay=10e-3)))
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
        num_samples_per_read = int(self.params.resolutionNRead * self.nav_resolution_defactor)
        grad_read_area = events.GRAD.make_trapezoid(
            channel=self.params.read_dir, system=self.pp_sys,
            flat_area=num_samples_per_read * self.params.deltaK_read,
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

        self.params.ESP = 2 * np.max([t_exci_ref, t_ref_1_adc, t_ref_2_adc]) * 1e3
        logModule.info(f"\t\t-found minimum ESP: {self.params.ESP:.2f} ms")

        if np.abs(t_ref_1_adc - t_ref_2_adc) > 1e-6:
            logModule.error(f"refocus to adc timing different from adc to refocus. Systematic error in seq. creation")
        t_half_esp = self.params.ESP * 1e-3 / 2
        # add delays
        if t_exci_ref < t_half_esp:
            self.delay_exci_ref1 = events.DELAY.make_delay(t_half_esp - t_exci_ref, system=self.pp_sys)
            if not self.delay_exci_ref1.check_on_block_raster():
                err = f"exci ref delay not on block raster"
                logModule.error(err)
        if t_ref_1_adc < t_half_esp:
            self.delay_ref_adc = events.DELAY.make_delay(t_half_esp - t_ref_1_adc, system=self.pp_sys)
            if not self.delay_ref_adc.check_on_block_raster():
                err = f"adc ref delay not on block raster"
                logModule.error(err)
        tes = np.arange(1, self.params.ETL+1) * self.params.ESP
        self.te = tes.tolist()

    def _calculate_slice_delay(self):
        # time per echo train
        t_pre_etl = self.block_excitation.rf.t_delay_s + self.block_excitation.rf.t_duration_s / 2
        t_etl = self.params.ETL * self.params.ESP * 1e-3  # esp in ms
        t_post_etl = self.block_acquisition.get_duration() / 2 + self.block_spoil_end.get_duration()

        t_total_etl = t_pre_etl + t_etl + t_post_etl
        # time for fid navs - one delay in between
        t_total_fid_nav = np.sum(
            [b.get_duration() for b in self.block_list_fid_nav_acq]
        ) + np.sum(
            [b.get_duration() for b in self.block_list_fid_nav_acq[:-1]]
        )
        logModule.info(f"\t\t-total fid-nav time (2 navs + 1 delay of 10ms): {t_total_fid_nav * 1e3:.2f} ms")
        # deminish TR by FIDnavs
        tr_eff = self.params.TR * 1e-3 - t_total_fid_nav
        max_num_slices = int(np.floor(tr_eff / t_total_etl))
        logModule.info(f"\t\t-total echo train length: {t_total_etl * 1e3:.2f} ms")
        logModule.info(f"\t\t-desired number of slices: {self.params.resolutionNumSlices}")
        logModule.info(f"\t\t-possible number of slices within TR: {max_num_slices}")
        if self.params.resolutionNumSlices > max_num_slices:
            logModule.info(f"increase TR or Concatenation needed")
        # we want to add a delay additionally after fid nav block
        self.delay_slice = events.DELAY.make_delay(
            (tr_eff - self.params.resolutionNumSlices * t_total_etl) / (self.params.resolutionNumSlices + 1),
            system=self.pp_sys
        )
        logModule.info(f"\t\t-time between slices: {self.delay_slice.get_duration() * 1e3:.2f} ms")
        if not self.delay_slice.check_on_block_raster():
            self.delay_slice.set_on_block_raster()
            logModule.info(f"\t\t-adjusting TR delay to raster time: {self.delay_slice.get_duration() * 1e3:.2f} ms")

    def _write_sampling_pattern(self, echo_idx: int, phase_idx: int, slice_idx: int,
                                num_scan: int = -1, nav_flag: bool = False, nav_dir: int = 0):
        sampling_index = {
            "num_scan": num_scan, "navigator": nav_flag,
            "slice_num": slice_idx, "pe_num": phase_idx, "echo_num": echo_idx, "nav_dir": nav_dir
        }
        self.sampling_pattern_constr.append(sampling_index)
        self.sampling_pattern_set = True
        return num_scan + 1

    def _build(self):
        logModule.info(f"build -- calculate minimum ESP")
        self._calculate_min_esp()
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

    def _loop_lines(self):
        # through phase encodes
        line_bar = tqdm.trange(
            self.params.numberOfCentralLines + self.params.numberOfOuterLines, desc="phase encodes"
        )
        # counter for number of scan
        scan_idx = 0
        for idx_n in line_bar:  # We have N phase encodes for all ETL contrasts
            for idx_slice in range(self.params.resolutionNumSlices):
                # looping through slices per phase encode
                self._set_fa(echo_idx=0)
                self._set_phase_grad(phase_idx=idx_n, echo_idx=0)
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
                self.pp_seq.add_block(*self.block_acquisition.list_events_to_ns())
                # write sampling pattern
                scan_idx, _ = self._write_sampling_pattern_entry(
                    scan_num=scan_idx, slice_num=self.trueSliceNum[idx_slice], pe_num=self.k_indexes[0, idx_n],
                    echo_num=0
                )

                # delay if necessary
                if self.delay_ref_adc.get_duration() > 1e-7:
                    self.pp_seq.add_block(self.delay_ref_adc.to_simple_ns())

                # loop
                for echo_idx in np.arange(1, self.params.ETL):
                    # set fa
                    self._set_fa(echo_idx=echo_idx)
                    # set phase
                    self._set_phase_grad(echo_idx=echo_idx, phase_idx=idx_n)
                    # set slice offset
                    self._apply_slice_offset(idx_slice=idx_slice)
                    # add block
                    self.pp_seq.add_block(*self.block_refocus.list_events_to_ns())
                    # delay if necessary
                    if self.delay_ref_adc.get_duration() > 1e-7:
                        self.pp_seq.add_block(self.delay_ref_adc.to_simple_ns())

                    # adc
                    self.pp_seq.add_block(*self.block_acquisition.list_events_to_ns())
                    # write sampling pattern
                    scan_idx, _ = self._write_sampling_pattern_entry(
                        scan_num=scan_idx, slice_num=self.trueSliceNum[idx_slice],
                        pe_num=self.k_indexes[echo_idx, idx_n], echo_num=echo_idx
                    )

                    # delay if necessary
                    if self.delay_ref_adc.get_duration() > 1e-7:
                        self.pp_seq.add_block(self.delay_ref_adc.to_simple_ns())
                # spoil end
                self._set_end_spoil_phase_grad()
                self.pp_seq.add_block(*self.block_spoil_end.list_events_to_ns())
                # insert slice delay
                self.pp_seq.add_block(self.delay_slice.to_simple_ns())

            # navigators
            for nav_idx in range(self.nav_num):
                self._apply_slice_offset_fid_nav(idx_nav=nav_idx)
                # excitation
                # add block
                self.pp_seq.add_block(*self.block_excitation_nav.list_events_to_ns())
                # epi style nav read
                # we set up a counter to track the phase encode line, k-space center is half of num lines
                line_counter = 0
                central_line = int(self.params.resolutionNPhase * self.nav_resolution_defactor / 2) - 1
                # we set up the pahse encode increments
                pe_increments = np.arange(1, int(self.params.resolutionNPhase * self.nav_resolution_defactor), 2)
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
                    if b.adc.get_duration() > 0:
                        nav_read_dir = np.sign(b.grad_read.amplitude[1])
                        # track which line we are writing from the incremental steps
                        nav_line_pe = np.sum(pe_increments[:line_counter]) + central_line
                        scan_idx, _ = self._write_sampling_pattern_entry(
                            scan_num=scan_idx, slice_num=nav_idx, nav_dir=nav_read_dir, nav_acq=True,
                            pe_num=nav_line_pe, echo_num=-1
                        )
                        line_counter += 1

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
        if np.abs(pe_end_amp) > self.pp_sys.max_grad:
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
            logModule.error(err)
            raise ValueError(err)
        sbb.rf.freq_offset_hz = grad_slice_amplitude_hz * z
        # we are setting the phase of a pulse here into its phase offset var.
        # To merge both: given phase parameter and any complex signal array data
        sbb.rf.phase_offset_rad = sbb.rf.phase_rad - 2 * np.pi * sbb.rf.freq_offset_hz * sbb.rf.calculate_center()


if __name__ == '__main__':
    pass
