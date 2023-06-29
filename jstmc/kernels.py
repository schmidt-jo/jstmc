import copy

import matplotlib.pyplot as plt
from jstmc import events, options
import numpy as np
import pypulseq as pp
import logging
copy

logModule = logging.getLogger(__name__)


def set_on_grad_raster_time(system: pp.Opts, time: float):
    return np.ceil(time / system.grad_raster_time) * system.grad_raster_time


class EventBlock:
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
        return [ev for ev in event_list if ev.get_duration() > 1e-5]

    def get_duration(self):
        return np.max([t.get_duration() for t in self.list_events()])

    @classmethod
    def build_excitation(cls, params: options.SequenceParameters, system: pp.Opts,
                         use_slice_spoiling: bool = True, adjust_ramp_area: float = 0.0):
        # Excitation
        logModule.info("setup excitation")
        if use_slice_spoiling:
            spoiling_moment = params.sliceSpoilingMoment
        else:
            spoiling_moment = 2e-7
        if params.extRfExc:
            logModule.info(f"rf -- loading rfpf from file: {params.extRfExc}")
            rf = events.RF.load_from_rfpf(
                fname=params.extRfExc, flip_angle_rad=params.excitationRadFA, phase_rad=params.excitationRadRfPhase,
                system=system, duration_s=params.excitationDuration * 1e-6, pulse_type='excitation'
            )
        else:
            logModule.info(f"rf -- build gauss pulse")
            time_bw_prod = params.excitationTimeBwProd
            rf = events.RF.make_gauss_pulse(
                flip_angle_rad=params.excitationRadFA,
                phase_rad=params.excitationRadRfPhase,
                pulse_type="excitation",
                delay_s=0.0,
                duration_s=params.excitationDuration * 1e-6,
                time_bw_prod=time_bw_prod,
                freq_offset_hz=0.0, phase_offset_rad=0.0,
                system=system
            )
        # build slice selective gradient

        grad_slice, grad_slice_delay, _ = events.GRAD.make_slice_selective(
            pulse_bandwidth_hz=-rf.bandwidth_hz,
            slice_thickness_m=params.resolutionSliceThickness * 1e-3,
            duration_s=params.excitationDuration * 1e-6,
            system=system,
            pre_moment=-params.excitationPreMoment,
            re_spoil_moment=-spoiling_moment,
            rephase=params.excitationRephaseFactor,
            adjust_ramp_area=adjust_ramp_area
        )
        # adjust start of rf
        rf.t_delay_s = grad_slice_delay

        # sanity checks
        if np.max(np.abs(grad_slice.amplitude)) > system.max_grad:
            err = f"gradient amplitude exceeds maximum allowed"
            logModule.error(err)
            raise ValueError(err)
        return cls(rf=rf, grad_slice=grad_slice)

    @classmethod
    def build_refocus(cls, params: options.SequenceParameters, system: pp.Opts,
                      pulse_num: int = 0, duration_spoiler: float = 0.0, return_pe_time: bool = False):
        # calculate read gradient in order to use correct area (corrected for ramps
        acquisition_window = set_on_grad_raster_time(system=system, time=params.acquisitionTime)
        grad_read = events.GRAD.make_trapezoid(
            channel=params.read_dir, system=system,
            flat_area=params.deltaK_read * params.resolutionNRead, flat_time=acquisition_window
        )
        # block is first refocusing + spoiling + phase encode
        logModule.info(f"setup refocus {pulse_num + 1}")
        # set up longest phase encode
        phase_grad_areas = (- np.arange(params.resolutionNPhase) + params.resolutionNPhase / 2) * \
                           params.deltaK_phase
        # build longest phase gradient
        grad_phase = events.GRAD.make_trapezoid(
            channel=params.phase_dir,
            area=np.max(phase_grad_areas),
            system=system
        )
        duration_phase_grad = set_on_grad_raster_time(
            time=grad_phase.get_duration(), system=system
        )

        # build read spoiler
        grad_prewind_read = events.GRAD.make_trapezoid(
            channel=params.read_dir,
            area=1 / 2 * grad_read.area,
            system=system,
        )
        duration_pre_read = set_on_grad_raster_time(
            system=system, time=grad_prewind_read.get_duration())

        duration_min = np.max([duration_phase_grad, duration_pre_read, duration_spoiler])

        if params.extRfRef:
            logModule.info(f"rf -- loading rfpf from file {params.extRfRef}")
            rf = events.RF.load_from_rfpf(
                fname=params.extRfRef, system=system,
                duration_s=params.refocusingDuration * 1e-6, flip_angle_rad=np.pi,
                phase_rad=0.0, pulse_type='refocusing'
            )
        else:
            logModule.info(f"rf -- build gauss pulse")
            rf = events.RF.make_gauss_pulse(
                flip_angle_rad=params.refocusingRadFA[pulse_num],
                phase_rad=params.refocusingRadRfPhase[pulse_num],
                pulse_type="refocusing",
                delay_s=0.0,
                duration_s=params.refocusingDuration * 1e-6,
                time_bw_prod=params.excitationTimeBwProd,
                freq_offset_hz=0.0, phase_offset_rad=0.0,
                system=system
            )
        if pulse_num == 0:
            pre_moment = 0.0
        else:
            pre_moment = params.sliceSpoilingMoment
        grad_slice, grad_slice_delay, grad_slice_spoil_re_time = events.GRAD.make_slice_selective(
            pulse_bandwidth_hz=-rf.bandwidth_hz,
            slice_thickness_m=params.refocusingScaleSliceGrad * params.resolutionSliceThickness * 1e-3,
            duration_s=params.refocusingDuration * 1e-6,
            system=system,
            pre_moment=-pre_moment,
            re_spoil_moment=-params.sliceSpoilingMoment,
            t_minimum_re_grad=duration_min
        )
        if duration_min < grad_slice_spoil_re_time:
            logModule.info(f"adjusting phase encode gradient durations (got time to spare)")
            duration_phase_grad = grad_slice_spoil_re_time
            duration_pre_read = grad_slice_spoil_re_time

        # adjust rf start
        rf.t_delay_s = grad_slice_delay

        if pulse_num > 0:
            # set symmetrical x / y
            # duration between - rather take middle part of slice select, rf duration on different raster possible
            t_duration_between = grad_slice.set_on_raster(grad_slice.slice_select_duration)
            grad_phase = events.GRAD.sym_grad(
                system=system, channel=params.phase_dir, area_lobe=np.max(phase_grad_areas),
                duration_lobe=duration_phase_grad, duration_between=t_duration_between, reverse_second_lobe=True
            )
            grad_read_prewind = events.GRAD.sym_grad(
                system=system, channel=params.read_dir, area_lobe=- grad_read.area / 2, duration_lobe=duration_pre_read,
                duration_between=rf.t_duration_s
            )
        else:
            grad_read_prewind = events.GRAD.make_trapezoid(
                channel=params.read_dir,
                area=- grad_read.area / 2,
                duration_s=duration_pre_read,  # given in [s] via options
                system=system,
            )
            grad_phase = events.GRAD.make_trapezoid(
                channel=params.phase_dir,
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
    def build_fs_acquisition(cls, params: options.SequenceParameters, system: pp.Opts):
        # block : adc + read grad
        logModule.info("setup acquisition")
        acquisition_window = set_on_grad_raster_time(system=system, time=params.acquisitionTime + system.adc_dead_time)
        grad_read = events.GRAD.make_trapezoid(
            channel=params.read_dir,
            flat_area=params.deltaK_read * params.resolutionNRead,
            flat_time=acquisition_window,  # given in [s] via options
            system=system
        )
        adc = events.ADC.make_adc(
            num_samples=int(params.resolutionNRead * params.oversampling),
            dwell=params.dwell,
            system=system
        )
        delay = (grad_read.get_duration() - adc.get_duration()) / 2
        if delay < 0:
            err = f"adc longer than read gradient"
            logModule.error(err)
            raise ValueError(err)
        adc.t_delay_s = delay
        # finished block
        return cls(adc=adc, grad_read=grad_read)

    @classmethod
    def build_pf_acquisition(cls, params: options.SequenceParameters, system: pp.Opts):
        # block : adc + read grad
        logModule.info("setup acquisition w undersampling partial fourier read")
        pf_factor = 0.75
        # we take 100 points in the center, the latter half of the rest is acquired with accelerated readout,
        # the first half is omitted
        # ToDo: make this a parameter in settings
        logModule.info(f"partial fourier for 0th echo, factor: {pf_factor:.2f}")
        num_acq_lines = int(pf_factor * params.resolutionNRead)
        # set acquisition time on raster
        acq_time_fs = set_on_grad_raster_time(
            system=system, time=params.dwell * num_acq_lines
        )
        # we take the usual full sampling
        grad_read_fs = events.GRAD.make_trapezoid(
            channel=params.read_dir, system=system,
            flat_area=params.deltaK_read * num_acq_lines, flat_time=acq_time_fs
        )
        area_ramp = grad_read_fs.amplitude[1] * grad_read_fs.t_array_s[1] * 0.5
        area_pre_read = (1 - 0.5 / pf_factor) * grad_read_fs.flat_area + area_ramp

        adc = events.ADC.make_adc(
            system=system, num_samples=num_acq_lines, delay_s=grad_read_fs.t_array_s[1], dwell=params.dwell
        )
        acq_block = cls()
        acq_block.grad_read = grad_read_fs
        acq_block.adc = adc

        t_middle = grad_read_fs.t_flat_time_s * (1 - pf_factor) + grad_read_fs.t_array_s[1]
        acq_block.t_mid = t_middle
        return acq_block, area_pre_read

    @classmethod
    def build_us_acquisition(cls, params: options.SequenceParameters, system: pp.Opts, invert_grad_dir: bool = False):
        logModule.info("setup acquisition w undersampling")
        # calculate maximum acc factor -> want to keep snr -> ie bandwidth ie. dwell equal and stretch read grad
        grad_amp_fs = params.deltaK_read / params.dwell
        acc_max = system.max_grad / grad_amp_fs
        logModule.info(f"maximum acceleration factor: {acc_max}, rounding to lower int")
        acc_max = int(np.floor(acc_max))
        grad_amp_us = acc_max * grad_amp_fs

        # calculate ramp between fs and us grads
        ramp_time_between = set_on_grad_raster_time(system=system, time=(acc_max - 1) * grad_amp_fs / system.max_slew)
        # calculate how much lines we miss when ramping (including oversampling)
        num_missed_lines_per_ramp = int(np.ceil(ramp_time_between / params.dwell))
        # calculate num of outer lines
        num_outer_lines = params.resolutionBase - params.numberOfCentralLines - 2 * num_missed_lines_per_ramp
        # per gradient
        num_out_lines_per_grad = int(num_outer_lines / 2)
        # flat time
        flat_time_us = set_on_grad_raster_time(system=system, time=num_out_lines_per_grad * params.dwell)
        flat_time_fs = set_on_grad_raster_time(system=system, time=params.numberOfCentralLines * params.dwell)

        # stitch them together / need to make each adc separate so we will return 3 blocks here
        ramp_time = set_on_grad_raster_time(system=system, time=grad_amp_us/system.max_slew)

        # build parts
        grad_read_parts_generic = events.GRAD()
        grad_read_parts_generic.system = system
        grad_read_parts_generic.channel = params.read_dir
        grad_read_parts_generic.t_delay_s = 0.0
        grad_read_parts_generic.max_grad = system.max_grad
        grad_read_parts_generic.max_slew = system.max_slew

        # first section
        grad_read_a = events.GRAD.copy(grad_read_parts_generic)
        grad_read_a.amplitude = np.array([
            0.0, grad_amp_us, grad_amp_us
        ])
        grad_read_a.t_array_s = np.array([
            0.0, ramp_time, ramp_time + flat_time_us
        ])
        grad_read_a.area = 0.5 * ramp_time * grad_amp_us + flat_time_us * grad_amp_us
        grad_read_a.flat_area = grad_amp_us * flat_time_us
        grad_read_a.t_rise_time_s = ramp_time
        grad_read_a.t_flat_time_s = flat_time_us
        grad_read_a.t_duration_s = grad_read_a.get_duration()

        # middle section
        grad_read_b = events.GRAD.copy(grad_read_parts_generic)
        grad_read_b.amplitude = np.array([
            grad_amp_us, grad_amp_fs, grad_amp_fs
        ])
        grad_read_b.t_array_s = np.array([
            0.0, ramp_time_between, ramp_time_between + flat_time_fs
        ])
        grad_read_b.area = 0.5 * ramp_time_between * (grad_amp_us - grad_amp_fs) + ramp_time_between * grad_amp_fs + \
                           grad_amp_fs * flat_time_fs
        grad_read_b.flat_area = grad_amp_fs * flat_time_fs
        grad_read_b.t_rise_time_s = ramp_time_between
        grad_read_b.t_flat_time_s = flat_time_fs
        grad_read_b.t_duration_s = grad_read_b.get_duration()

        # last section
        grad_read_c = events.GRAD.copy(grad_read_parts_generic)
        grad_read_c.amplitude = np.array([
            grad_amp_fs, grad_amp_us, grad_amp_us, 0.0
        ])
        grad_read_c.t_array_s = np.array([
            0.0, ramp_time_between, ramp_time_between + flat_time_us, ramp_time_between + flat_time_us + ramp_time
        ])
        grad_read_c.area = 0.5 * ramp_time_between * (grad_amp_us - grad_amp_fs) + ramp_time_between * grad_amp_fs + \
                           grad_amp_us * flat_time_us + 0.5 * grad_amp_us * ramp_time
        grad_read_c.flat_area = grad_amp_us * flat_time_us
        grad_read_c.t_rise_time_s = ramp_time
        grad_read_c.t_flat_time_s = flat_time_us
        grad_read_c.t_duration_s = grad_read_c.get_duration()

        adc_short = events.ADC.make_adc(
            system=system,
            dwell=params.dwell,
            num_samples=num_out_lines_per_grad
        )
        adc_long = events.ADC.make_adc(
            system=system,
            dwell=params.dwell,
            num_samples=params.numberOfCentralLines
        )

        grads = [grad_read_a, grad_read_b, grad_read_c]
        if invert_grad_dir:
            for grad in grads:
                grad.amplitude = -grad.amplitude
                grad.area = - grad.area
                grad.flat_area = - grad.area

        adc_short.t_delay_s = grad_read_a.t_array_s[1]
        acq_0 = EventBlock(grad_read=grad_read_a, adc=adc_short)

        adc_long.t_delay_s = grad_read_b.t_array_s[1]
        acq_1 = EventBlock(grad_read=grad_read_b, adc=adc_long)

        adc_short.t_delay_s = grad_read_c.t_array_s[1]
        acq_2 = EventBlock(grad_read=grad_read_c, adc=adc_short)

        return acq_0, acq_1, acq_2

    @classmethod
    def build_spoiler_end(cls, params: options.SequenceParameters, system: pp.Opts):
        grad_read = events.GRAD.make_trapezoid(
            channel=params.read_dir, system=system,
            flat_area=params.deltaK_read * params.resolutionNRead, flat_time=params.acquisitionTime
        )
        phase_grad_areas = (- np.arange(params.resolutionNPhase) + params.resolutionNPhase / 2) * \
                           params.deltaK_phase
        grad_read_spoil = events.GRAD.make_trapezoid(
            channel=params.read_dir,
            area=-1 / 2 * grad_read.area,
            system=system
        )
        grad_phase = events.GRAD.make_trapezoid(
            channel=params.phase_dir,
            area=np.max(phase_grad_areas),
            system=system
        )
        grad_slice = events.GRAD.make_trapezoid(
            channel='z',
            system=system,
            area=params.sliceEndSpoilingMoment
        )
        duration = grad_phase.set_on_raster(
            np.max([grad_slice.get_duration(), grad_phase.get_duration(), grad_read_spoil.get_duration()])
        )
        # set longest for all
        grad_read_spoil = events.GRAD.make_trapezoid(
            channel=params.read_dir,
            area=-1 / 2 * grad_read.area,
            system=system,
            duration_s=duration
        )
        grad_phase = events.GRAD.make_trapezoid(
            channel=params.phase_dir,
            area=np.max(phase_grad_areas),
            system=system,
            duration_s=duration
        )
        grad_slice = events.GRAD.make_trapezoid(
            channel='z',
            system=system,
            area=-params.sliceEndSpoilingMoment,
            duration_s=duration
        )
        return cls(system=system, grad_slice=grad_slice, grad_phase=grad_phase, grad_read=grad_read_spoil)

    def plot(self):
        plt.style.use('ggplot')
        rf_color = '#7300e6'
        rf_color2 = '#666699'
        grad_slice_color = '#2eb8b8'
        grad_read_color = '#ff5050'
        grad_phase_color = '#00802b'
        # set axis
        x_arr = np.arange(int(np.round(self.get_duration() * 1e6)))
        # rf
        rf = np.zeros_like(x_arr, dtype=complex)
        start = int(self.rf.t_delay_s * 1e6)
        end = int(1e6 * self.rf.get_duration()) - int(1e6 * self.rf.t_ringdown_s)
        rf[start:end] = self.rf.signal
        rf_abs = np.abs(rf) / np.max(np.abs(rf))
        rf_angle = np.angle(rf) / np.pi
        rf_angle[rf_abs > 1e-7] += self.rf.phase_rad / np.pi

        # grad slice
        grad_ss = np.zeros_like(x_arr, dtype=float)
        del_start = int(1e6 * self.grad_slice.t_delay_s)
        for t_idx in np.arange(1, self.grad_slice.t_array_s.shape[0]):
            start = int(1e6 * self.grad_slice.t_array_s[t_idx - 1])
            end = int(1e6 * self.grad_slice.t_array_s[t_idx])
            grad_ss[del_start + start:del_start + end] = np.linspace(self.grad_slice.amplitude[t_idx - 1],
                                                                     self.grad_slice.amplitude[t_idx], end - start)

        # grad read
        grad_read = np.zeros_like(x_arr, dtype=float)
        del_start = int(1e6 * self.grad_read.t_delay_s)
        for t_idx in np.arange(1, self.grad_read.t_array_s.shape[0]):
            start = int(1e6 * self.grad_read.t_array_s[t_idx - 1])
            end = int(1e6 * self.grad_read.t_array_s[t_idx])
            grad_read[del_start + start:del_start + end] = np.linspace(self.grad_read.amplitude[t_idx - 1],
                                                                       self.grad_read.amplitude[t_idx], end - start)

        # grad phase
        grad_phase = np.zeros_like(x_arr, dtype=float)
        del_start = int(1e6 * self.grad_phase.t_delay_s)
        for t_idx in np.arange(1, self.grad_phase.t_array_s.shape[0]):
            start = int(1e6 * self.grad_phase.t_array_s[t_idx - 1])
            end = int(1e6 * self.grad_phase.t_array_s[t_idx])
            grad_phase[del_start + start:del_start + end] = np.linspace(self.grad_phase.amplitude[t_idx - 1],
                                                                        self.grad_phase.amplitude[t_idx], end - start)

        # cast to mT / m
        grad_ss *= 1e3 / 42577478.518
        grad_read *= 1e3 / 42577478.518
        grad_phase *= 1e3 / 42577478.518
        max_grad = int(self.system.max_grad * 1.2e3 / 42577478.518)

        fig = plt.figure()
        if self.rf.get_duration() > 1e-7:
            # if we have an rf pulse we can title the plot by its use
            fig.suptitle(f"gradient/pulse: {self.rf.pulse_type}")
        ax_rf = fig.add_subplot(2, 1, 1)
        ax_rf.plot(x_arr, rf_abs, label='rf abs', color=rf_color)
        ax_rf.fill_between(x_arr, rf_angle, label='rf phase', alpha=0.4, color=rf_color2)
        ax_rf.plot(x_arr, rf_angle, ls='dotted', color=rf_color2)
        ax_rf.set_xlabel('time [us]')
        ax_rf.set_ylabel(f"rf amp. [a.u.] | phase [$\pi$]")
        ax_grad = ax_rf.twinx()
        ax_grad.plot(x_arr, grad_ss, label='slice grad', color=grad_slice_color)
        ax_grad.fill_between(x_arr, grad_ss, color=grad_slice_color, alpha=0.4)
        ax_grad.set_ylabel('grad slice [mT/m]')
        ax_rf.set_ylim(-1.2, 1.2)
        ax_grad.set_ylim(-max_grad, max_grad)
        # ADD THIS LINE
        ax_grad.set_yticks(np.linspace(
            ax_grad.get_yticks()[0],
            ax_grad.get_yticks()[-1],
            len(ax_rf.get_yticks()))
        )
        ax_grad.grid(False)
        ax_grad.yaxis.label.set_color(grad_slice_color)
        ax_grad.spines['right'].set_color(grad_slice_color)
        ax_grad.tick_params(axis='y', colors=grad_slice_color)
        ax_rf.legend(loc=1)

        ax_gr = fig.add_subplot(2, 1, 2)
        ax_gr.plot(x_arr, grad_read, ls='--', label='slice grad', color=grad_read_color)
        ax_gr.fill_between(x_arr, grad_read, hatch='/', color=grad_read_color, alpha=0.4)
        ax_gr.set_xlabel('time [us]')
        ax_gr.set_ylabel(f"grad read [mT/m]")
        ax_gr.yaxis.label.set_color(grad_read_color)
        ax_gr.spines['left'].set_color(grad_read_color)
        ax_gr.tick_params(axis='y', colors=grad_read_color)

        ax_gp = ax_gr.twinx()
        ax_gp.plot(x_arr, grad_phase, ls='dotted', label='slice grad', color=grad_phase_color)
        ax_gp.fill_between(x_arr, grad_phase, hatch="\\", color=grad_phase_color, alpha=0.4)
        ax_gp.set_ylabel(f"grad phase [mT/m]")
        ax_gp.yaxis.label.set_color(grad_phase_color)
        ax_gp.spines['right'].set_color(grad_phase_color)
        ax_gp.tick_params(axis='y', colors=grad_phase_color)
        ax_gp.set_ylim(-max_grad, max_grad)
        ax_gr.set_ylim(-max_grad, max_grad)
        plt.tight_layout()
        plt.show()
