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
        t_start = self.adc.t_delay_s
        # set adc sampling point times
        t_adc_sampling = np.arange(self.adc.num_samples) * self.adc.t_dwell_s
        # interpolate grad amp values for adc positions
        grad_amp_for_t_adc = np.interp(t_start + t_adc_sampling, self.grad_read.t_array_s, self.grad_read.amplitude)
        # interpolate ramp / pre grad
        # pick timings
        t_pre = self.grad_read.t_array_s[self.grad_read.t_array_s <= t_start].tolist()
        grad_amps_pre = self.grad_read.amplitude[self.grad_read.t_array_s <= t_start].tolist()
        if t_start - t_pre[-1] > 1e-8:
            # interpolate start amp
            grad_amps_pre.append(np.interp(
                t_start, [t_pre[-1], self.grad_read.t_array_s[self.grad_read.t_array_s > t_start][0]],
                [grad_amps_pre[-1], self.grad_read.amplitude[self.grad_read.t_array_s > t_start][0]]
            ))
            # add start time
            t_pre.append(t_start)
        # calculate k-position before start of adc
        area_pre = pre_read_area + np.trapz(grad_amps_pre, t_pre)
        # first adc position is at t_start, then we sample each dwell time in between the grad might change
        # hence we want to calculate the area between each step
        grad_areas_for_t_adc = np.zeros_like(grad_amp_for_t_adc)
        for amp_idx in np.arange(1, grad_areas_for_t_adc.shape[0]):
            grad_areas_for_t_adc[amp_idx] = grad_areas_for_t_adc[amp_idx - 1] + np.trapz(
                grad_amp_for_t_adc[amp_idx - 1:amp_idx + 1], dx=self.adc.t_dwell_s
            )

        # calculate k-positions
        k_pos = (area_pre + grad_areas_for_t_adc) / fs_grad_area
        return k_pos

    @classmethod
    def excitation_slice_sel(cls, pyp_interface: pypsi.Params.pypulseq, system: pp.Opts,
                             use_slice_spoiling: bool = True, adjust_ramp_area: float = 0.0):
        # Excitation
        log_module.info("setup excitation")
        if use_slice_spoiling:
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
                                pulse_num: int = 0, duration_spoiler: float = 0.0, return_pe_time: bool = False):
        # calculate read gradient in order to use correct area (corrected for ramps
        acquisition_window = set_on_grad_raster_time(system=system, time=pyp_interface.acquisition_time)
        grad_read = events.GRAD.make_trapezoid(
            channel=pyp_interface.read_dir, system=system,
            flat_area=pyp_interface.delta_k_read * pyp_interface.resolution_n_read, flat_time=acquisition_window
        )
        # block is first refocusing + spoiling + phase encode
        log_module.info(f"setup refocus {pulse_num + 1}")
        # set up longest phase encode
        phase_grad_areas = (- np.arange(pyp_interface.resolution_n_phase) + pyp_interface.resolution_n_phase / 2) * \
                           pyp_interface.delta_k_phase
        # build longest phase gradient
        grad_phase = events.GRAD.make_trapezoid(
            channel=pyp_interface.phase_dir,
            area=np.max(phase_grad_areas),
            system=system
        )
        duration_phase_grad = set_on_grad_raster_time(
            time=grad_phase.get_duration(), system=system
        )

        # build read spoiler
        grad_prewind_read = events.GRAD.make_trapezoid(
            channel=pyp_interface.read_dir,
            area=1 / 2 * grad_read.area,
            system=system,
        )
        duration_pre_read = set_on_grad_raster_time(
            system=system, time=grad_prewind_read.get_duration())

        duration_min = np.max([duration_phase_grad, duration_pre_read, duration_spoiler])

        if pyp_interface.ext_rf_ref:
            log_module.info(f"rf -- loading rfpf from file {pyp_interface.ext_rf_ref}")
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
                system=system, channel=pyp_interface.phase_dir, area_lobe=np.max(phase_grad_areas),
                duration_lobe=duration_phase_grad, duration_between=t_duration_between, reverse_second_lobe=True
            )
            grad_read_prewind = events.GRAD.sym_grad(
                system=system, channel=pyp_interface.read_dir, area_lobe=- grad_read.area / 2,
                duration_lobe=duration_pre_read,
                duration_between=rf.t_duration_s
            )
        else:
            grad_read_prewind = events.GRAD.make_trapezoid(
                channel=pyp_interface.read_dir,
                area=- grad_read.area / 2,
                duration_s=duration_pre_read,  # given in [s] via options
                system=system,
            )
            grad_phase = events.GRAD.make_trapezoid(
                channel=pyp_interface.phase_dir,
                area=np.max(phase_grad_areas),
                system=system,
                duration_s=duration_phase_grad
            )
            # adjust phase start
            delay_phase_grad = rf.t_delay_s + rf.t_duration_s
            grad_phase.t_delay_s = delay_phase_grad
            # adjust read start
            grad_read_prewind.t_delay_s = delay_phase_grad

        # finished block
        _instance = cls(
            rf=rf, grad_slice=grad_slice,
            grad_phase=grad_phase, grad_read=grad_read_prewind
        )
        if return_pe_time:
            return _instance, grad_phase.set_on_raster(duration_phase_grad)
        else:
            return _instance

    @classmethod
    def acquisition_fs(cls, pyp_interface: pypsi.Params.pypulseq, system: pp.Opts):
        # block : adc + read grad
        log_module.info("setup acquisition")
        acquisition_window = set_on_grad_raster_time(
            system=system, time=pyp_interface.acquisition_time + system.adc_dead_time
        )
        grad_read = events.GRAD.make_trapezoid(
            channel=pyp_interface.read_dir,
            flat_area=pyp_interface.delta_k_read * pyp_interface.resolution_n_read,
            flat_time=acquisition_window,  # given in [s] via options
            system=system
        )
        adc = events.ADC.make_adc(
            num_samples=int(pyp_interface.resolution_n_read * pyp_interface.oversampling),
            dwell=pyp_interface.dwell,
            system=system
        )
        delay = (grad_read.get_duration() - adc.get_duration()) / 2
        if delay < 0:
            err = f"adc longer than read gradient"
            log_module.error(err)
            raise ValueError(err)
        adc.t_delay_s = delay
        # finished block
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
        acquisition_window = set_on_grad_raster_time(
            system=system,
            time=pyp_interface.dwell * num_samples_per_read * pyp_interface.oversampling + system.adc_dead_time
        )
        log_module.debug(f" pe line: {np.sum(pe_increments[:line_num])}")
        grad_read = events.GRAD.make_trapezoid(
            channel=pyp_interface.read_dir,
            flat_area=np.power(-1, line_num) * pyp_interface.delta_k_read * num_samples_per_read,
            flat_time=acquisition_window,  # given in [s] via options
            system=system
        )
        adc = events.ADC.make_adc(
            num_samples=int(num_samples_per_read * pyp_interface.oversampling),
            dwell=pyp_interface.dwell,
            system=system
        )
        delay = (grad_read.get_duration() - adc.get_duration()) / 2
        if delay < 0:
            err = f"adc longer than read gradient"
            log_module.error(err)
            raise ValueError(err)
        adc.t_delay_s = delay
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
        num_acq_pts = int(pf_factor * pyp_interface.resolution_n_read)
        # set acquisition time on raster
        acq_time_fs = set_on_grad_raster_time(
            system=system, time=pyp_interface.dwell * num_acq_pts * pyp_interface.oversampling
        )
        # we take the usual full sampling
        grad_read_fs = events.GRAD.make_trapezoid(
            channel=pyp_interface.read_dir, system=system,
            flat_area=pyp_interface.delta_k_read * num_acq_pts, flat_time=acq_time_fs
        )
        area_ramp = grad_read_fs.amplitude[1] * grad_read_fs.t_array_s[1] * 0.5
        area_pre_read = (1 - 0.5 / pf_factor) * grad_read_fs.flat_area + area_ramp

        adc = events.ADC.make_adc(
            system=system, num_samples=int(num_acq_pts * pyp_interface.oversampling),
            delay_s=grad_read_fs.t_array_s[1], dwell=pyp_interface.dwell
        )
        acq_block = cls()
        acq_block.grad_read = grad_read_fs
        acq_block.adc = adc

        t_middle = grad_read_fs.t_flat_time_s * (1 - pf_factor) + grad_read_fs.t_array_s[1]
        acq_block.t_mid = t_middle

        # get k_space trajectory
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

        t_delay = 0.5 * (grad_read.get_duration() - adc.get_duration())
        # want to set adc symmetrically into grad read
        adc.t_delay_s = t_delay
        adc.set_on_raster()
        acq = Kernel(grad_read=grad_read, adc=adc)

        return acq, acc_max

    @classmethod
    def spoil_all_grads(cls, pyp_interface: pypsi.Params.pypulseq, system: pp.Opts):
        grad_read = events.GRAD.make_trapezoid(
            channel=pyp_interface.read_dir, system=system,
            flat_area=pyp_interface.delta_k_read * pyp_interface.resolution_n_read, flat_time=pyp_interface.acquisition_time
        )
        phase_grad_areas = (- np.arange(pyp_interface.resolution_n_phase) + pyp_interface.resolution_n_phase / 2) * \
                           pyp_interface.delta_k_phase
        grad_read_spoil = events.GRAD.make_trapezoid(
            channel=pyp_interface.read_dir,
            area=-pyp_interface.read_grad_spoiling_factor * grad_read.area,
            system=system
        )
        grad_phase = events.GRAD.make_trapezoid(
            channel=pyp_interface.phase_dir,
            area=np.max(phase_grad_areas),
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
            area=np.max(phase_grad_areas),
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

    def plot(self, path: typing.Union[str, plib.Path], name=""):
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
            times = (np.array([
                0, self.adc.t_delay_s - 1e-9, self.adc.t_delay_s + 1e-9,
                   self.adc.get_duration() - 1e-9, self.adc.get_duration() + 1e-9]) * 1e6).astype(int)
            amp = np.array([0, 0, 1, 1, 0])
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
            fig.update_xaxes(title_text="Time [\u00B5]")

        fig_path = plib.Path(path).absolute().joinpath("plots/")
        fig_path.mkdir(parents=True, exist_ok=True)
        fig_path = fig_path.joinpath(f"plot_{name}").with_suffix(".html")
        log_module.info(f"writing file: {fig_path.as_posix()}")
        fig.write_html(fig_path.as_posix())
