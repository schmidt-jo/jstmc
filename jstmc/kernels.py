import copy
import typing

import pathlib as plib
from jstmc import events
import pypsi
import numpy as np
import pypulseq as pp
import logging
import pandas as pd
import plotly.express as px
import plotly.subplots as psub
import plotly.graph_objs as pgo

log_module = logging.getLogger(__name__)


def set_on_grad_raster_time(system: pp.Opts, time: float):
    return np.ceil(time / system.grad_raster_time) * system.grad_raster_time


class Kernel:
    """
    kernel class, representation of one block for a sequence containing RF, ADC, Delay and all gradient events.
    Collection of methods to build predefined blocks for reusage
    """

    def __init__(
            self, system: pp.Opts = pp.Opts(),
            rf: events.RF = events.RF(),
            grad_read: events.GRAD = events.GRAD(),
            grad_phase: events.GRAD = events.GRAD(),
            grad_slice: events.GRAD = events.GRAD(),
            adc: events.ADC = events.ADC(),
            delay: events.DELAY = events.DELAY()):

        self.system = system

        self.rf: events.RF = rf

        self.grad_read: events.GRAD = grad_read
        self.grad_phase: events.GRAD = grad_phase
        self.grad_slice: events.GRAD = grad_slice

        self.adc: events.ADC = adc

        self.delay: events.DELAY = delay

    def copy(self):
        return copy.deepcopy(self)

    def list_events_to_ns(self):
        return [ev.to_simple_ns() for ev in self.list_events()]

    def list_events(self):
        event_list = [self.rf, self.grad_read, self.grad_slice, self.grad_phase, self.adc, self.delay]
        return [ev for ev in event_list if ev.get_duration() > 5e-6]

    def get_duration(self):
        return np.max([t.get_duration() for t in self.list_events()])

    def get_k_space_trajectory(self, pre_read_area: float, fs_grad_area: float):
        """ we want to extract the k-space trajectory for a block.
        Currently only in 1D, aka along the read direction. Assume phase encodes are working as expected.
        In principle we could use this for both directions, also add compensations (eg. non linearity)
        The idea is to get the points of the adc wrt. gradient moments and
        use this in kbnufft style gridding during recon"""
        if self.adc.get_duration() < 1e-6:
            err = f"kernel/block has no adc, cant compute trajectory"
            log_module.error(err)
            raise AttributeError(err)
        # find starting point of adc
        # From Pulseq: According to the information from Klaus Scheffler and indirectly from Siemens this
        # is the present convention - the samples are shifted by 0.5 dwell
        t_start = self.adc.t_delay_s + self.adc.t_dwell_s / 2

        # set up times = want to include the gradient before adc start
        t_pre_adc_grad = np.concatenate(
            (self.grad_read.t_array_s[self.grad_read.t_array_s <= t_start], np.array([t_start])),
            axis=0
        )
        t_adc_sampling = t_start + np.arange(self.adc.num_samples) * self.adc.t_dwell_s

        # interpolate gradient amplitudes for pre grad and readout
        grad_amp_pre_adc = np.interp(
            x=t_pre_adc_grad, xp=self.grad_read.t_array_s, fp=self.grad_read.amplitude
        )
        grad_amp_adc_sampling = np.interp(
            x=t_adc_sampling, xp=self.grad_read.t_array_s, fp=self.grad_read.amplitude
        )

        # set array to hold grad areas cumulatively per sample
        grad_areas_for_t_adc = np.zeros(self.adc.num_samples)

        # calculate gradient area til first sample, including prephasing area
        grad_areas_for_t_adc[0] = pre_read_area + np.trapz(x=t_pre_adc_grad, y=grad_amp_pre_adc)
        # calculate gradient area for each sample iteratively, adding to the previous
        for amp_idx in np.arange(1, grad_areas_for_t_adc.shape[0]):
            grad_areas_for_t_adc[amp_idx] = grad_areas_for_t_adc[amp_idx - 1] + np.trapz(
                grad_amp_adc_sampling[amp_idx - 1: amp_idx + 1], dx=self.adc.t_dwell_s
            )
        # calculate k-positions
        k_pos = grad_areas_for_t_adc / fs_grad_area
        return k_pos

    @classmethod
    def excitation_slice_sel(cls, pyp_interface: pypsi.Params.pypulseq, system: pp.Opts,
                             use_slice_spoiling: bool = True, adjust_ramp_area: float = 0.0):
        # Excitation
        log_module.info("setup excitation")
        if use_slice_spoiling:
            # using spoiling gradient defined by interface file
            spoiling_moment = pyp_interface.grad_moment_slice_spoiling
        else:
            spoiling_moment = 2e-7
        if pyp_interface.ext_rf_exc:
            log_module.info(f"rf -- loading rfpf from file: {pyp_interface.ext_rf_exc}")
            rf = events.RF.load_from_pypsi_pulse(
                fname=pyp_interface.ext_rf_exc, flip_angle_rad=pyp_interface.excitation_rf_rad_fa,
                phase_rad=pyp_interface.excitation_rf_rad_phase,
                system=system, duration_s=pyp_interface.excitation_duration * 1e-6,
                pulse_type='excitation'
            )
        else:
            log_module.info(f"rf -- build gauss pulse")
            time_bw_prod = pyp_interface.excitation_rf_time_bw_prod
            rf = events.RF.make_gauss_pulse(
                flip_angle_rad=pyp_interface.excitation_rf_rad_fa,
                phase_rad=pyp_interface.excitation_rf_rad_phase,
                pulse_type="excitation",
                delay_s=0.0,
                duration_s=pyp_interface.excitation_duration * 1e-6,
                time_bw_prod=time_bw_prod,
                freq_offset_hz=0.0, phase_offset_rad=0.0,
                system=system
            )
        # build slice selective gradient

        grad_slice, grad_slice_delay, _ = events.GRAD.make_slice_selective(
            pulse_bandwidth_hz=-rf.bandwidth_hz,
            slice_thickness_m=pyp_interface.resolution_slice_thickness * 1e-3,
            duration_s=pyp_interface.excitation_duration * 1e-6,
            system=system,
            pre_moment=-pyp_interface.excitation_grad_moment_pre,
            re_spoil_moment=-spoiling_moment,
            rephase=pyp_interface.excitation_grad_rephase_factor,
            adjust_ramp_area=adjust_ramp_area
        )
        # adjust start of rf
        rf.t_delay_s = grad_slice_delay

        # sanity checks
        if np.max(np.abs(grad_slice.amplitude)) > system.max_grad:
            err = f"gradient amplitude exceeds maximum allowed"
            log_module.error(err)
            raise ValueError(err)
        return cls(rf=rf, grad_slice=grad_slice)

    @classmethod
    def refocus_slice_sel_spoil(cls, pyp_interface: pypsi.Params.pypulseq, system: pp.Opts,
                                pulse_num: int = 0, duration_spoiler: float = 0.0, return_pe_time: bool = False,
                                read_gradient_to_prephase: float = None):
        if read_gradient_to_prephase is None:
            # calculate read gradient in order to use correct area (corrected for ramps
            grad_read = events.GRAD.make_trapezoid(
                channel=pyp_interface.read_dir, system=system,
                flat_area=pyp_interface.delta_k_read * pyp_interface.resolution_n_read,
                flat_time=pyp_interface.acquisition_time
            )
            read_gradient_to_prephase = 1 / 2 * grad_read.area
        grad_read_pre = events.GRAD.make_trapezoid(
            channel=pyp_interface.read_dir, system=system,
            area=- read_gradient_to_prephase
        )

        # block is first refocusing + spoiling + phase encode
        log_module.info(f"setup refocus {pulse_num + 1}")
        # set up longest phase encode
        max_phase_grad_area = pyp_interface.resolution_n_phase / 2 * pyp_interface.delta_k_phase
        # build longest phase gradient
        grad_phase = events.GRAD.make_trapezoid(
            channel=pyp_interface.phase_dir,
            area=max_phase_grad_area,
            system=system
        )
        duration_phase_grad = set_on_grad_raster_time(
            time=grad_phase.get_duration(), system=system
        )

        duration_pre_read = set_on_grad_raster_time(
            system=system, time=grad_read_pre.get_duration())

        duration_min = np.max([duration_phase_grad, duration_pre_read, duration_spoiler])

        if pyp_interface.ext_rf_ref:
            log_module.info(f"rf -- loading pulse from file {pyp_interface.ext_rf_ref}")
            rf = events.RF.load_from_pypsi_pulse(
                fname=pyp_interface.ext_rf_ref, system=system,
                duration_s=pyp_interface.refocusing_duration * 1e-6, flip_angle_rad=np.pi,
                phase_rad=0.0, pulse_type='refocusing'
            )
        else:
            log_module.info(f"rf -- build sync pulse")
            rf = events.RF.make_gauss_pulse(
                flip_angle_rad=pyp_interface.refocusing_rf_rad_fa[pulse_num],
                phase_rad=pyp_interface.refocusing_rf_rad_phase[pulse_num],
                pulse_type="refocusing",
                delay_s=0.0,
                duration_s=pyp_interface.refocusing_duration * 1e-6,
                time_bw_prod=pyp_interface.excitation_rf_time_bw_prod,
                freq_offset_hz=0.0, phase_offset_rad=0.0,
                system=system
            )
        if pulse_num == 0:
            pre_moment = 0.0
        else:
            pre_moment = pyp_interface.grad_moment_slice_spoiling
        grad_slice, grad_slice_delay, grad_slice_spoil_re_time = events.GRAD.make_slice_selective(
            pulse_bandwidth_hz=-rf.bandwidth_hz,
            slice_thickness_m=pyp_interface.refocusing_grad_slice_scale * pyp_interface.resolution_slice_thickness * 1e-3,
            duration_s=pyp_interface.refocusing_duration * 1e-6,
            system=system,
            pre_moment=-pre_moment,
            re_spoil_moment=-pyp_interface.grad_moment_slice_spoiling,
            t_minimum_re_grad=duration_min
        )
        if duration_min < grad_slice_spoil_re_time:
            log_module.info(f"adjusting phase encode gradient durations (got time to spare)")
            duration_phase_grad = grad_slice_spoil_re_time
            duration_pre_read = grad_slice_spoil_re_time

        # adjust rf start
        rf.t_delay_s = grad_slice_delay

        if pulse_num > 0:
            # set symmetrical x / y
            # duration between - rather take middle part of slice select, rf duration on different raster possible
            t_duration_between = grad_slice.set_on_raster(grad_slice.slice_select_duration)
            grad_phase = events.GRAD.sym_grad(
                system=system, channel=pyp_interface.phase_dir, area_lobe=max_phase_grad_area,
                duration_lobe=duration_phase_grad, duration_between=t_duration_between, reverse_second_lobe=True
            )
            grad_read_pre = events.GRAD.sym_grad(
                system=system, channel=pyp_interface.read_dir, area_lobe=- read_gradient_to_prephase,
                duration_lobe=duration_pre_read,
                duration_between=t_duration_between
            )
        else:
            grad_read_pre = events.GRAD.make_trapezoid(
                channel=pyp_interface.read_dir,
                area=- read_gradient_to_prephase,
                duration_s=duration_pre_read,  # given in [s] via options
                system=system,
            )
            grad_phase = events.GRAD.make_trapezoid(
                channel=pyp_interface.phase_dir,
                area=max_phase_grad_area,
                system=system,
                duration_s=duration_phase_grad
            )
            # adjust phase start
            delay_phase_grad = rf.t_delay_s + rf.t_duration_s
            grad_phase.t_delay_s = delay_phase_grad
            # adjust read start
            grad_read_pre.t_delay_s = delay_phase_grad

        # finished block
        _instance = cls(
            rf=rf, grad_slice=grad_slice,
            grad_phase=grad_phase, grad_read=grad_read_pre
        )
        if return_pe_time:
            return _instance, grad_phase.set_on_raster(duration_phase_grad)
        else:
            return _instance

    @classmethod
    def acquisition_fs(cls, pyp_params: pypsi.Params.pypulseq, system: pp.Opts,
                       invert_grad_read_dir: bool = False):
        # block : adc + read grad
        log_module.info("setup acquisition")
        adc = events.ADC.make_adc(
            num_samples=int(pyp_params.resolution_n_read * pyp_params.oversampling),
            dwell=pyp_params.dwell,
            system=system
        )
        # calculate time for oversampling * dwells more
        t_adc_extended = int((pyp_params.resolution_n_read + 1) * pyp_params.oversampling) * pyp_params.dwell
        # calculate area for 1 read point more, s.th. flat area for original number read stays the same
        flat_area = (
            pyp_params.delta_k_read * (pyp_params.resolution_n_read + 1)
                    ) * np.power(-1, int(invert_grad_read_dir))
        # make at least oversampling * dwell and adc dead time to fit into the falling ramp
        # since we need to shift the adc samples correctly depending on the grad direction
        ramp_times = adc.t_dead_time_s + pyp_params.oversampling * adc.t_dwell_s
        # both adjustments together should prohibit adc stretching out of gradient flat time
        grad_read = events.GRAD.make_trapezoid(
            channel=pyp_params.read_dir,
            ramp_time=ramp_times,
            flat_area=flat_area,
            flat_time=t_adc_extended,
            system=system
        )
        # want to set adc symmetrically into grad read, and we want the middle adc sample to hit k space center.
        t_mid_grad = grad_read.t_mid
        # The 0th line counts as a positive line, hence we have one less line in plus then minus direction.
        # we need to adress this when the readout gradient is inverted (i.e. its not just negating gradient amplitude)
        # if k = n_read / 2, we have the sampling points range from -k ,..., 0, ... k-1
        # if we reverse the direction of the gradient we need to go from k - 1, ..., 0, ... -k
        # hence the middle 0 points is hit at different times
        # From Pulseq: According to the information from Klaus Scheffler and indirectly from Siemens this
        # is the present convention - the samples are shifted by 0.5 dwell
        # we made sure n_read is divisible by 2.
        n_adc_mid = int((pyp_params.resolution_n_read / 2 - int(invert_grad_read_dir) / 2) * pyp_params.oversampling)
        t_adc_mid = n_adc_mid * adc.t_dwell_s
        # if we want to hit t_mid grad after n_adc_mid samples (plus a sample start shift of 0.5 dwell), we need to
        # calculate the right adc start delay
        delay = t_mid_grad - t_adc_mid - adc.t_dwell_s / 2

        if delay < 0:
            err = f"adc longer than read gradient"
            log_module.error(err)
            raise ValueError(err)
        # sanity check
        if delay % system.adc_raster_time > 1e-9:
            # adc delay not fitting on adc raster
            warn = "adc delay set not multiple of adc raster. might lead to small timing deviations in actual .seq file"
            log_module.warning(warn)
        # send warning if last adc reaches into ramp
        if adc.get_duration() > grad_read.get_duration() - grad_read.t_fall_time_s:
            warn = (f"adc duration beyond gradient flat time, reaching into ramp."
                    f"This would shift k-space samples to unwanted positions. check calculations.")
            log_module.warning(warn)
        adc.t_delay_s = delay
        # finished block
        if adc.get_duration() > grad_read.get_duration():
            err = (f"adc duration longer than read gradient duration. \n"
                   f"this might happen from the shifting of the adc samplings to fit the middle of the gradient. \n"
                   f"it shouldnt since we specified enough room for the adc in the gradient ramps, check calculations!")
            log_module.error(err)
            raise AttributeError(err)
        return cls(adc=adc, grad_read=grad_read)

    @classmethod
    def acquisition_fid_nav(cls, pyp_interface: pypsi.Params.pypulseq, system: pp.Opts,
                            line_num: int, reso_degrading: float = 1 / 6):
        if line_num == 0:
            log_module.info("setup FID Navigator")
        # want 1/6th  of resolution of original image (i.e. if 0.7mm iso in read direction, we get 3.5 mm resolution)
        # hence we need only 1/6th of the number of points with same delta k, want this to be divisible by 2
        # (center half line inclusion out)
        num_samples_per_read = int(pyp_interface.resolution_n_read * reso_degrading)
        pe_increments = np.arange(1, int(pyp_interface.resolution_n_phase * reso_degrading), 2)
        pe_increments *= np.power(-1, np.arange(pe_increments.shape[0]))
        # we step by those increments dependent on line number
        grad_phase = events.GRAD.make_trapezoid(
            channel=pyp_interface.phase_dir,
            area=pyp_interface.delta_k_phase * pe_increments[line_num],
            system=system
        )
        log_module.debug(f" pe line: {np.sum(pe_increments[:line_num])}")
        grad_read = events.GRAD.make_trapezoid(
            channel=pyp_interface.read_dir,
            flat_area=np.power(-1, line_num) * pyp_interface.delta_k_read * num_samples_per_read,
            flat_time=pyp_interface.dwell * num_samples_per_read * pyp_interface.oversampling,
            # given in [s] via options
            system=system
        )
        adc = events.ADC.make_adc(
            num_samples=int(num_samples_per_read * pyp_interface.oversampling),
            dwell=pyp_interface.dwell,
            system=system
        )
        delay = (grad_read.get_duration() - adc.t_duration_s) / 2
        if delay < 0:
            err = f"adc longer than read gradient"
            log_module.error(err)
            raise ValueError(err)
        # delay remains dead time if its bigger
        if delay > system.adc_dead_time:
            adc.t_delay_s = delay
        else:
            warn = "cant set adc delay to match read gradient bc its smaller than adc dead time."
            log_module.warning(warn)
        # get duration of adc and start phase blip when adc is over (possibly during ramp of read)
        grad_phase.t_delay_s = grad_phase.set_on_raster(adc.get_duration(), double=False)
        # finished block
        return cls(adc=adc, grad_read=grad_read, grad_phase=grad_phase)

    @classmethod
    def acquisition_pf_undersampled(cls, pyp_interface: pypsi.Params.pypulseq, system: pp.Opts):
        # block : adc + read grad
        log_module.info("setup acquisition w undersampling partial fourier read")
        pf_factor = 0.75
        # we take 100 points in the center, the latter half of the rest is acquired with accelerated readout,
        # the first half is omitted
        # ToDo: make this a parameter in settings
        log_module.info(f"partial fourier for 0th echo, factor: {pf_factor:.2f}")
        num_acq_pts_wo_os = int(pf_factor * pyp_interface.resolution_n_read)
        num_acq_pts_os = int(num_acq_pts_wo_os * pyp_interface.oversampling)
        # set adc
        adc = events.ADC.make_adc(
            system=system, num_samples=num_acq_pts_os,
            dwell=pyp_interface.dwell
        )
        # we take the usual full sampling
        grad_read_fs = events.GRAD.make_trapezoid(
            channel=pyp_interface.read_dir, system=system,
            flat_area=pyp_interface.delta_k_read * num_acq_pts_wo_os, flat_time=adc.t_duration_s
        )
        # due to gradient raster the flat time might be prolonged. we need to figure out the placement of samples
        # and calculate t0
        # second half fully sampled
        samples_to_right = int(pyp_interface.resolution_n_read * pyp_interface.oversampling / 2)
        samples_to_left = num_acq_pts_os - samples_to_right
        # set adc to start at flat area
        adc.t_delay_s = grad_read_fs.t_array_s[1]
        # middle line is first point of samples to right, samples are shifted by dwell / 2
        t0 = (samples_to_left + 0.5) * adc.t_dwell_s
        # we need to prephase read such that we arrive at 0 k in k-space at t0
        # calculate ramp area
        area_ramp = grad_read_fs.amplitude[1] * grad_read_fs.t_array_s[1] * 0.5
        # calculate area to prephase = area
        area_pre_read = area_ramp + t0 * grad_read_fs.amplitude[1]
        # calculate area to rephase
        # we interpolate the amplitudes at all later points to calculate area
        times = np.array(
            [
                t0 + adc.t_delay_s,     # central k - space point
                *grad_read_fs.t_array_s[grad_read_fs.t_array_s > t0 + adc.t_delay_s]
                # gradient time points reached afterwards
            ]
        )
        amps = np.interp(
            x=times,
            xp=grad_read_fs.t_array_s,
            fp=grad_read_fs.amplitude
        )
        area_post_read = np.trapz(x=times, y=amps)

        if adc.t_delay_s < system.adc_dead_time:
            warn = f"adc delay will be bigger than set, due to system dead time constraints"
            log_module.warning(warn)
        # sanity check
        grad_area = grad_read_fs.area
        if np.abs(np.trapz(x=grad_read_fs.t_array_s, y=grad_read_fs.amplitude) - grad_area) > 1e-9:
            logging.error(f"pf read gradient area calculation error")
        if np.abs(grad_area - area_pre_read - area_post_read) > 1e-9:
            logging.error(f"gradient 0th echo rewind calculation error")
        acq_block = cls()
        acq_block.grad_read = grad_read_fs
        acq_block.adc = adc

        acq_block.t_mid = t0 + adc.t_delay_s

        return acq_block, area_pre_read

    @classmethod
    def acquisition_sym_undersampled(cls, pyp_interface: pypsi.Params.pypulseq, system: pp.Opts,
                                     invert_grad_dir: bool = False, asym_accelerated: bool = False):
        log_module.info("setup acquisition w undersampling")
        # calculate maximum acc factor -> want to keep snr -> ie bandwidth ie dwell equal and stretch read grad
        # dwell = 1 / bw / os / num_samples
        grad_amp_fs = pyp_interface.delta_k_read / pyp_interface.dwell / pyp_interface.oversampling
        # we want read to use max 65 % of max grad
        acc_max = 0.65 * system.max_grad / grad_amp_fs
        log_module.info(f"maximum acceleration factor: {acc_max:.2f}, rounding to lower int")
        acc_max = int(np.floor(acc_max))
        grad_amp_us = acc_max * grad_amp_fs

        # calculate ramp between fs and us grads - want to use only half of max slew - minimize stimulation
        ramp_time_between = set_on_grad_raster_time(
            system=system, time=(acc_max - 1) * grad_amp_fs / (0.5 * system.max_slew)
        )
        # want to set it to multiples of dwell time
        # calculate how much lines we miss when ramping (including oversampling)
        num_adc_per_ramp_os = int(np.ceil(ramp_time_between / pyp_interface.dwell))
        ramp_time_between = set_on_grad_raster_time(system=system,
                                                    time=num_adc_per_ramp_os * pyp_interface.dwell)
        # calculate num of outer pts (including oversampling)
        num_outer_lines_os = int(
            (
                    pyp_interface.oversampling * (pyp_interface.resolution_base - pyp_interface.number_central_lines) -
                    2 * num_adc_per_ramp_os
            ) / acc_max
        )
        # total pts including lost ones plus acceleration (including oversampling)
        num_lines_total_os = int(
            pyp_interface.oversampling * pyp_interface.number_central_lines + 2 * num_adc_per_ramp_os +
            num_outer_lines_os
        )
        # per gradient (including oversampling)
        num_out_lines_per_grad_os = int(num_outer_lines_os / 2)
        # flat time
        flat_time_us = set_on_grad_raster_time(
            system=system, time=num_out_lines_per_grad_os * pyp_interface.dwell
        )
        flat_time_fs = set_on_grad_raster_time(
            system=system, time=pyp_interface.number_central_lines * pyp_interface.dwell * pyp_interface.oversampling
        )

        # stitch them together / we cover this with one continous adc and use gridding of kbnufft
        ramp_time = set_on_grad_raster_time(system=system, time=grad_amp_us / system.max_slew)
        # ramp area in between
        ramp_between_area = 0.5 * ramp_time_between * (grad_amp_us - grad_amp_fs) + ramp_time_between * grad_amp_fs

        # build
        grad_read = events.GRAD()
        grad_read.system = system
        grad_read.channel = pyp_interface.read_dir
        grad_read.t_delay_s = 0.0
        grad_read.max_grad = system.max_grad
        grad_read.max_slew = system.max_slew

        if asym_accelerated:
            grad_read.amplitude = np.array([
                0.0,
                grad_amp_fs,
                grad_amp_fs,
                grad_amp_us,
                grad_amp_us,
                0.0
            ])
            # calculate lower grad amp ramp
            grad_read.t_array_s = np.array([
                0.0,
                ramp_time,
                ramp_time + 1.5 * flat_time_fs,
                ramp_time + 1.5 * flat_time_fs + ramp_time_between,
                ramp_time + 1.5 * flat_time_fs + ramp_time_between + flat_time_us,
                2 * ramp_time + 1.5 * flat_time_fs + ramp_time_between + flat_time_fs,
            ])
            # ToDo!
            grad_read.area = np.array([
                flat_time_fs * grad_amp_fs,
                0.5 * ramp_time * grad_amp_us + flat_time_us * grad_amp_us + ramp_between_area,
            ])
            # ToDo
            adc = events.ADC.make_adc(
                system=system,
                dwell=pyp_interface.dwell,
                num_samples=num_lines_total_os
            )

        else:
            grad_read.amplitude = np.array([
                0.0,
                grad_amp_us,
                grad_amp_us,
                grad_amp_fs,
                grad_amp_fs,
                grad_amp_us,
                grad_amp_us,
                0.0
            ])
            grad_read.t_array_s = np.array([
                0.0,
                ramp_time,
                ramp_time + flat_time_us,
                ramp_time + flat_time_us + ramp_time_between,
                ramp_time + flat_time_us + ramp_time_between + flat_time_fs,
                ramp_time + flat_time_us + 2 * ramp_time_between + flat_time_fs,
                ramp_time + 2 * flat_time_us + 2 * ramp_time_between + flat_time_fs,
                2 * ramp_time + 2 * flat_time_us + 2 * ramp_time_between + flat_time_fs,
            ])
            grad_read.area = np.array([
                np.trapz(y=grad_read.amplitude[:4], x=grad_read.t_array_s[:4]),
                flat_time_fs * grad_amp_fs,
                np.trapz(y=grad_read.amplitude[4:], x=grad_read.t_array_s[4:]),
            ])
            grad_read.flat_area = np.array([
                grad_amp_us * flat_time_us,
                grad_amp_fs * flat_time_fs,
                grad_amp_us * flat_time_us
            ])
            grad_read.t_rise_time_s = ramp_time
            grad_read.t_flat_time_s = 2 * flat_time_us * flat_time_fs
            grad_read.t_duration_s = grad_read.get_duration()

            adc = events.ADC.make_adc(
                system=system,
                dwell=pyp_interface.dwell,
                num_samples=num_lines_total_os
            )

        if invert_grad_dir:
            grad_read.amplitude = -grad_read.amplitude
            grad_read.area = - grad_read.area
            grad_read.flat_area = - grad_read.flat_area

        delay = (grad_read.get_duration() - adc.t_duration_s) / 2
        if delay < 0:
            err = f"adc longer than read gradient"
            log_module.error(err)
            raise ValueError(err)
        # delay remains dead time if its bigger
        if delay > system.adc_dead_time:
            adc.t_delay_s = delay
        else:
            warn = "cant set adc delay to match read gradient bc its smaller than adc dead time."
            log_module.warning(warn)
        # want to set adc symmetrically into grad read
        adc.t_delay_s = delay
        adc.set_on_raster()
        acq = Kernel(grad_read=grad_read, adc=adc)

        return acq, acc_max

    @classmethod
    def spoil_all_grads(cls, pyp_interface: pypsi.Params.pypulseq, system: pp.Opts):
        grad_read = events.GRAD.make_trapezoid(
            channel=pyp_interface.read_dir, system=system,
            flat_area=pyp_interface.delta_k_read * pyp_interface.resolution_n_read,
            flat_time=pyp_interface.acquisition_time
        )
        grad_read_spoil = events.GRAD.make_trapezoid(
            channel=pyp_interface.read_dir,
            area=-pyp_interface.read_grad_spoiling_factor * grad_read.area,
            system=system
        )
        grad_phase = events.GRAD.make_trapezoid(
            channel=pyp_interface.phase_dir,
            area=pyp_interface.resolution_n_phase / 2 * pyp_interface.delta_k_phase,
            system=system
        )
        grad_slice = events.GRAD.make_trapezoid(
            channel='z',
            system=system,
            area=pyp_interface.grad_moment_slice_spoiling_end
        )
        duration = grad_phase.set_on_raster(
            np.max([grad_slice.get_duration(), grad_phase.get_duration(), grad_read_spoil.get_duration()])
        )
        # set longest for all
        grad_read_spoil = events.GRAD.make_trapezoid(
            channel=pyp_interface.read_dir,
            area=-pyp_interface.read_grad_spoiling_factor * grad_read.area,
            system=system,
            duration_s=duration
        )
        grad_phase = events.GRAD.make_trapezoid(
            channel=pyp_interface.phase_dir,
            area=pyp_interface.resolution_n_phase / 2 * pyp_interface.delta_k_phase,
            system=system,
            duration_s=duration
        )
        grad_slice = events.GRAD.make_trapezoid(
            channel='z',
            system=system,
            area=-pyp_interface.grad_moment_slice_spoiling_end,
            duration_s=duration
        )
        return cls(system=system, grad_slice=grad_slice, grad_phase=grad_phase, grad_read=grad_read_spoil)

    def plot(self, path: typing.Union[str, plib.Path], name: str = "", file_suffix: str = "png"):
        # build dataframe
        # columns - index, facet, ampl_1, ampl_2, ampl_3, time
        # plot rf abs(1) phase(2) + or adc on one facet, plot grads on one facet
        arr_facet = []
        arr_time = np.zeros(0)
        amplitude = np.zeros(0)
        ev_type = []

        # rf
        rf_toggle = False
        if self.rf.get_duration() > 0:
            rf_toggle = True
            # get time
            times = np.zeros(int(1e6 * self.rf.t_duration_s) + 3)
            times[1:-1] = int(self.rf.t_delay_s * 1e6) - 0.1
            times[2:-1] = times[1] + 0.1 + np.arange(1e6 * self.rf.t_duration_s).astype(int)
            times[-1] = times[-2] + 1
            # get signals
            # amplitude
            abs = np.zeros_like(times)
            abs[2:-1] = np.abs(self.rf.signal)
            abs /= np.max(abs)
            # add
            arr_time = np.concatenate((arr_time, times), axis=0)
            amplitude = np.concatenate((amplitude, abs), axis=0)
            ev_type += ["rf amplitude"] * times.shape[0]
            arr_facet += ["rf / adc"] * times.shape[0]
            # phase
            phase = np.zeros_like(times)
            phase[2:-1] = np.angle(self.rf.signal) / np.pi
            phase[abs > 1e-7] += self.rf.phase_rad / np.pi
            # add
            arr_time = np.concatenate((arr_time, times), axis=0)
            amplitude = np.concatenate((amplitude, phase), axis=0)
            ev_type += ["rf phase"] * times.shape[0]
            arr_facet += ["rf / adc"] * times.shape[0]

        # grad slice
        if self.grad_slice.get_duration() > 0:
            times = (np.array([0, self.grad_slice.t_delay_s - 1e-6,
                               *(self.grad_slice.t_delay_s + self.grad_slice.t_array_s)]) * 1e6).astype(int)
            amp = np.zeros_like(times)
            amp[2:] = self.grad_slice.amplitude * 1e3 / 42577478.518
            # add
            arr_time = np.concatenate((arr_time, times), axis=0)
            amplitude = np.concatenate((amplitude, amp), axis=0)
            ev_type += ["grad slice"] * times.shape[0]
            arr_facet += ["grads"] * times.shape[0]

        # grad read
        if self.grad_read.get_duration() > 0:
            times = (np.array([0, self.grad_read.t_delay_s - 1e-6,
                               *(self.grad_read.t_delay_s + self.grad_read.t_array_s)]) * 1e6).astype(int)
            amp = np.zeros_like(times)
            amp[2:] = self.grad_read.amplitude * 1e3 / 42577478.518
            # add
            arr_time = np.concatenate((arr_time, times), axis=0)
            amplitude = np.concatenate((amplitude, amp), axis=0)
            ev_type += ["grad read"] * times.shape[0]
            arr_facet += ["grads"] * times.shape[0]

        # grad phase
        if self.grad_phase.get_duration() > 0:
            times = (np.array([0, self.grad_phase.t_delay_s - 1e-6,
                               *(self.grad_phase.t_delay_s + self.grad_phase.t_array_s)]) * 1e6).astype(int)
            amp = np.zeros_like(times)
            amp[2:] = self.grad_phase.amplitude * 1e3 / 42577478.518
            # add
            arr_time = np.concatenate((arr_time, times), axis=0)
            amplitude = np.concatenate((amplitude, amp), axis=0)
            ev_type += ["grad phase"] * times.shape[0]
            arr_facet += ["grads"] * times.shape[0]

        # adc
        if self.adc.get_duration() > 0:
            times = (
                    np.array([
                        0,
                        self.adc.t_delay_s - 1e-9,
                        self.adc.t_delay_s + 1e-9,
                        self.adc.t_delay_s + self.adc.t_duration_s - 1e-9,
                        self.adc.t_delay_s + self.adc.t_duration_s + 1e-9,
                        self.adc.get_duration() - 1e-9,
                        self.adc.get_duration() + 1e-9
                    ]) * 1e6
            ).astype(int)
            amp = np.array([0, 0, 1, 1, 0.25, 0.2, 0])
            # we sketch dead time here
            # add
            arr_time = np.concatenate((arr_time, times), axis=0)
            amplitude = np.concatenate((amplitude, amp), axis=0)
            ev_type += ["adc"] * times.shape[0]
            arr_facet += ["rf / adc"] * times.shape[0]

        df = pd.DataFrame({"facet": arr_facet, "time": arr_time, "amplitude": amplitude, "type": ev_type})

        if rf_toggle:
            fig = px.line(df, x="time", y="amplitude", color="type", facet_row="facet", labels={
                "time": "Time [\u00B5s]", "facet": ""
            }, title=name)
            fig.update_yaxes(matches=None)
            fig.update_layout({
                "yaxis2": {"title": "RF Amplitude [A.U.] + Phase [\u03C0]"},
                "yaxis": {"title": "Gradient Amplitude [mT/m]"}
            })
            for annotation in fig.layout.annotations:
                annotation["text"] = ""
        else:
            # Create figure with secondary y-axis
            fig = psub.make_subplots(specs=[[{"secondary_y": True}]])

            # Add traces
            p_df = df[df["facet"] == "rf / adc"]
            fig.add_trace(
                pgo.Scatter(x=p_df["time"], y=p_df["amplitude"], name="adc", fill="tozeroy", opacity=0.6),
                secondary_y=False,
            )
            # Add traces
            p_df = df[df["facet"] == "grads"]
            for dat in px.line(p_df, x="time", y="amplitude", color="type").data:
                fig.add_trace(
                    dat,
                    secondary_y=True,
                )
            # if pf acquisition
            if hasattr(self, "t_mid"):
                fig.add_vline(x=1e6 * self.t_mid, line_dash="dash", line_color="red",
                              annotation_text="k-space-read-center", annotation_position="top left")
            # color cycle (maximally adc + 3 grads) - purple, cyan, orange, lime
            colors = ["#5c15ad", "#1fdeab", "#de681f", "#de681f"]
            for idx_data in range(fig.data.__len__()):
                fig.data[idx_data]["line"].color = colors[idx_data]
            # Add figure title
            fig.update_layout(title_text=name)
            # Set axes titles
            fig.update_yaxes(title_text="ADV on/off", secondary_y=False, range=[0, 2],
                             tickmode="array", tickvals=[0, 1, 2], ticktext=["Off", "On", ""])
            fig.update_yaxes(title_text="Gradient Amplitude [mT/m]", secondary_y=True)
            fig.update_xaxes(title_text="Time [\u00B5s]")
        fig.update_layout(width=900, height=600)
        fig_path = plib.Path(path).absolute().joinpath("plots/")
        fig_path.mkdir(parents=True, exist_ok=True)
        fig_path = fig_path.joinpath(f"plot_{name}").with_suffix(f".{file_suffix}")
        log_module.info(f"writing file: {fig_path.as_posix()}")
        if file_suffix in ["png", "pdf"]:
            fig.write_image(fig_path.as_posix())
        else:
            fig.write_html(fig_path.as_posix())
