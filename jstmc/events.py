"""
event specific classes

"""
import types
import typing

import matplotlib.pyplot as plt
import pypulseq as pp

import numpy as np
from rf_pulse_files import rfpf
import logging

logModule = logging.getLogger(__name__)


class Event:
    def __init__(self):
        self.t_duration_s: float = 0.0
        self.system: pp.Opts = NotImplemented

    def get_duration(self):
        raise NotImplementedError

    def to_simple_ns(self):
        raise NotImplementedError


class RF(Event):
    def __init__(self):
        super().__init__()
        self.flip_angle_deg: typing.Union[float, np.ndarray] = 0.0
        self.phase_deg: typing.Union[float, np.ndarray] = 0.0
        self.pulse_type: str = "exc"
        self.extRfFile: str = ""

        self.flip_angle_rad: typing.Union[float, np.ndarray] = self.flip_angle_deg / 180.0 * np.pi
        self.phase_rad: typing.Union[float, np.ndarray] = self.phase_deg / 180.0 * np.pi

        self.freq_offset_hz: float = 0.0
        self.phase_offset_rad: float = 0.0

        self.t_delay_s: float = 0.0
        self.t_ringdown_s: float = 0.0
        self.t_dead_time_s: float = 0.0

        self.bandwidth_hz: float = 0.0
        self.time_bandwidth: float = 0.0

        self.signal: np.ndarray = np.zeros(0, dtype=complex)

    @classmethod
    def load_from_rfpf(cls, fname: str, flip_angle_rad: float, phase_rad: float, system: pp.Opts,
                       duration_s: float = 2e-3, delay_s: float = 0.0, pulse_type: str = 'excitation'):
        rf_instance = cls()
        rf = rfpf.RF.load(fname)
        rf_instance.extRfFile = fname
        rf_instance.pulse_type = pulse_type

        # get signal envelope
        signal = rf.amplitude * np.exp(1j * rf.phase)
        # calculate raster with assigned duration
        delta_t = duration_s / rf.num_samples
        # normalise flip angle
        flip = np.sum(np.abs(signal)) * delta_t * 2 * np.pi

        # assign values
        rf_instance.signal = signal * flip_angle_rad / flip
        rf_instance.flip_angle_rad = flip_angle_rad
        rf_instance.flip_angle_deg = flip_angle_rad / np.pi * 180.0

        rf_instance.phase_rad = phase_rad
        rf_instance.phase_deg = phase_rad / np.pi * 180.0

        rf_instance.t_duration_s = duration_s
        rf_instance.time_bandwidth = rf.time_bandwidth
        rf_instance.bandwidth_hz = rf_instance.time_bandwidth / duration_s

        rf_instance.system = system
        rf_instance.t_delay_s = delay_s
        rf_instance.t_ringdown_s = system.rf_ringdown_time
        rf_instance.t_dead_time_s = system.rf_dead_time
        return rf_instance

    @classmethod
    def make_sinc_pulse(cls, flip_angle_rad: float, system: pp.Opts, phase_rad: float = 0.0, pulse_type: str = 'excitation',
                        delay_s: float = 0.0, duration_s: float = 2e-3,
                        freq_offset_hz: float = 0.0, phase_offset_rad: float = 0.0,
                        time_bw_prod: float = 2):
        rf_instance = cls()
        rf_simple_ns = pp.make_sinc_pulse(
            use=pulse_type,
            flip_angle=flip_angle_rad,
            delay=delay_s,
            duration=duration_s,
            freq_offset=freq_offset_hz,
            phase_offset=phase_offset_rad + phase_rad,
            return_gz=False,
            time_bw_product=time_bw_prod,
            system=system
        )
        rf_instance.flip_angle_deg = flip_angle_rad * 180.0 / np.pi
        rf_instance.phase_deg = phase_offset_rad * 180.0 / np.pi
        rf_instance.pulse_type = pulse_type
        rf_instance.extRfFile = ""

        rf_instance.flip_angle_rad = flip_angle_rad
        rf_instance.phase_rad = phase_rad

        rf_instance.freq_offset_hz = freq_offset_hz
        rf_instance.phase_offset_rad = phase_offset_rad

        rf_instance.t_delay_s = delay_s
        rf_instance.t_duration_s = duration_s
        rf_instance.t_ringdown_s = system.rf_ringdown_time
        rf_instance.t_dead_time_s = system.rf_dead_time

        rf_instance.bandwidth_hz = time_bw_prod / duration_s
        rf_instance.time_bandwidth = time_bw_prod

        rf_instance.signal = rf_simple_ns.signal
        rf_instance.system = system
        return rf_instance

    def get_duration(self):
        return self.t_delay_s + self.t_duration_s + self.t_ringdown_s

    def set_on_raster(self, input_value: typing.Union[int, float, np.ndarray]):
        is_single = isinstance(input_value, (int, float))
        if is_single:
            us_value = np.array(input_value) * 1e6
        else:
            us_value = 1e6 * input_value
        us_raster = 1e6 * self.system.rf_raster_time
        choice = us_value % us_raster
        us_value[choice < 1e-4] = us_value[choice < 1e-4]
        us_value[choice > 1e-4] = np.round(us_value / us_raster) * us_raster
        if is_single:
            return 1e-6 * us_value[0]
        else:
            return 1e-6 * us_value

    def to_simple_ns(self):
        return types.SimpleNamespace(
            use=self.pulse_type, dead_time=self.t_dead_time_s, delay=self.t_delay_s,
            freq_offset=self.freq_offset_hz, phase_offset=self.phase_offset_rad,
            ringdown_time=self.t_ringdown_s, shape_dur=self.t_duration_s,
            signal=self.signal, t=self.set_on_raster(np.linspace(0, self.t_duration_s, self.signal.shape[0])),
            type='rf'
        )

    def calculate_center(self):
        """
        calculate the central point of rf shape
        for now assume middle, but can extend to max
        """
        return self.t_duration_s / 2


class GRAD(Event):
    def __init__(self):
        super().__init__()
        self.channel: str = 'z'
        self.amplitude: typing.Union[float, np.ndarray] = 0.0
        self.area: float = 0.0
        self.flat_area: float = 0.0

        self.t_array_s: np.ndarray = np.zeros(0)
        self.t_delay_s: float = 0.0
        self.t_fall_time_s = 1e-5
        self.t_rise_time_s = 1e-5
        self.t_flat_time_s = 1e-5

        self.system: pp.Opts = pp.Opts()

        self.max_slew: float = self.system.max_slew
        self.max_grad: float = self.system.max_grad

    def set_on_raster(self, value: float, return_delay: bool = False, double: bool = True):
        raster_time = float(self.system.grad_raster_time)
        if double:
            # helps with maintaining raster when calculating esp
            raster_time *= 2.0
        if np.abs(value) % raster_time < 1e-9:
            rastered_value = value
        else:
            rastered_value = np.ceil(value / raster_time) * raster_time
        if not return_delay:
            return rastered_value
        else:
            delay = (rastered_value - value) / 2
            return rastered_value, delay

    @classmethod
    def make_trapezoid(cls, channel: str, system: pp.Opts, amplitude: float = 0.0, area: float = None,
                       delay_s: float = 0.0, duration_s: float = 0.0,
                       flat_area: float = 0.0, flat_time: float = -1.0,
                       rise_time: float = 0.0):
        grad_instance = cls()
        grad_instance.system = system

        grad_simple_ns = pp.make_trapezoid(
            channel=channel,
            amplitude=amplitude,
            area=area,
            delay=delay_s,
            duration=duration_s,
            flat_area=flat_area,
            flat_time=flat_time,
            rise_time=rise_time,
            system=system
        )
        grad_instance.channel = channel
        grad_instance.amplitude = np.array([
            0.0, grad_simple_ns.amplitude,
            grad_simple_ns.amplitude, 0.0
        ])
        grad_instance.area = grad_simple_ns.area
        grad_instance.flat_area = grad_simple_ns.flat_area

        grad_instance.t_array_s = np.array([
            0.0, grad_simple_ns.rise_time,
            grad_simple_ns.rise_time + grad_simple_ns.flat_time,
            grad_simple_ns.rise_time + grad_simple_ns.flat_time + grad_simple_ns.fall_time
        ])
        grad_instance.t_delay_s = delay_s
        grad_instance.t_fall_time_s = rise_time
        grad_instance.t_rise_time_s = rise_time
        grad_instance.t_flat_time_s = flat_time

        grad_instance.system = system

        grad_instance.max_slew = system.max_slew
        grad_instance.max_grad = system.max_grad
        grad_instance.t_duration_s = grad_instance.get_duration()
        return grad_instance

    @classmethod
    def make_slice_selective(
            cls, pulse_bandwidth_hz: float, slice_thickness_m: float, duration_s: float,
            system: pp.Opts, pre_moment: float = 0.0, re_spoil_moment: float = 0.0,
            rephase: float = 0.0, t_minimum_re_grad: float = 0.0, adjust_ramp_area: float = None):
        # init
        grad_instance = cls()
        grad_instance.system = system
        duration_s, rf_raster_delay = grad_instance.set_on_raster(duration_s, return_delay=True)
        # set slice select amplitude
        amplitude = pulse_bandwidth_hz / slice_thickness_m
        amps = [0.0]
        times = [0.0]
        areas = []
        # set ramp times to max grad/ slew rate + 2%. could be optimized timing wise!
        t_ramp_unipolar = grad_instance.set_on_raster(1.02 * system.max_grad / system.max_slew)
        t_ramp_bipolar = grad_instance.set_on_raster(2.04 * system.max_grad / system.max_slew)
        t_minimum_re_grad = grad_instance.set_on_raster(t_minimum_re_grad)

        # calculations
        def get_asym_grad_amplitude(
                duration: float, moment: float,
                t_asym_ramp: float = t_ramp_unipolar, t_zero_ramp: float = t_ramp_unipolar,
                asym_amplitude: float = amplitude) -> float:
            return (moment - asym_amplitude * t_asym_ramp / 2) / (duration - t_asym_ramp / 2 - t_zero_ramp / 2)

        def get_asym_grad_min_duration(max_amplitude: float, moment: float,
                                       t_asym_ramp: float = t_ramp_unipolar, t_zero_ramp: float = t_ramp_unipolar,
                                       asym_amplitude: float = amplitude) -> float:
            return grad_instance.set_on_raster(
                (moment - asym_amplitude * t_asym_ramp / 2) / max_amplitude + t_asym_ramp / 2 + t_zero_ramp / 2
            )

        # pre moment
        if np.abs(pre_moment) > 1e-7:
            # add to area array
            areas.append(pre_moment)
            # we assume moment of slice select and pre phaser to act in same direction
            if np.sign(pre_moment) != np.sign(amplitude):
                logModule.error(f"pre-phase / spoil pre -- slice select not optimized for opposite sign grads")
            # want to minimize timing of gradient - use max grad
            pre_grad_amplitude = np.sign(pre_moment) * system.max_grad
            duration_pre_grad = get_asym_grad_min_duration(max_amplitude=pre_grad_amplitude, moment=pre_moment)
            pre_t_flat = grad_instance.set_on_raster(duration_pre_grad - 2 * t_ramp_unipolar)
            if duration_pre_grad < t_minimum_re_grad:
                # stretch to minimum required time
                duration_pre_grad = t_minimum_re_grad
                pre_grad_amplitude = get_asym_grad_amplitude(duration=duration_pre_grad, moment=pre_moment)
            amps.extend([pre_grad_amplitude, pre_grad_amplitude, amplitude])
            times.extend([t_ramp_unipolar, t_ramp_unipolar + pre_t_flat, duration_pre_grad])
        else:
            # ramp up
            duration_pre_grad = grad_instance.set_on_raster(np.abs(amplitude / system.max_slew))
            times.append(duration_pre_grad)
            amps.append(amplitude)
        delay = times[-1] + rf_raster_delay
        # slice select grad
        amps.append(amplitude)
        times.append(duration_pre_grad + duration_s)
        areas.append(amplitude * duration_s)
        t = duration_pre_grad + duration_s
        re_start_time = t
        # re / spoil
        if np.abs(re_spoil_moment) > 1e-7:
            if np.sign(re_spoil_moment) != np.sign(amplitude):
                logModule.error(f"pre-phase / spoil pre -- slice select not optimized for opposite sign grads")
            # optimised for same sign gradients, if rephaseing we could end up with opposite sign grads.
            t_ramp_asym = t_ramp_unipolar
            if rephase > 1e-6:
                # set up ramp for this case - just choose double the ramp time to account for complete gradient swing
                t_ramp_asym = t_ramp_bipolar
            re_spoil_moment += - 0.5 * rephase * areas[-1]
            if adjust_ramp_area:
                re_spoil_moment -= adjust_ramp_area
            areas.append(re_spoil_moment)
            if t_minimum_re_grad > 1e-6:
                # duration given - use symmetrical timing : same time as re gradient
                duration_re_grad = grad_instance.set_on_raster(t_minimum_re_grad)
                if t_ramp_unipolar > duration_re_grad:
                    err = "ramp times longer than available gradient time, slew rate limit"
                    logModule.error(err)
                    raise ValueError(err)
                # we want to fit the pre moment into the given duration
                # i.e. ramp 0 to pre_amplitude, flat time, ramp pre_amplitude to amplitude
                re_grad_amplitude = get_asym_grad_amplitude(
                    duration=duration_re_grad, moment=re_spoil_moment,
                    t_asym_ramp=t_ramp_asym)
                # very specific requirement jstmc sequence. adjust for ramp up of next slice selective gradient
                if np.abs(re_grad_amplitude) > system.max_grad:
                    re_grad_amplitude = np.sign(re_spoil_moment) * system.max_grad
                    duration_re_grad = get_asym_grad_min_duration(
                        max_amplitude=re_grad_amplitude, moment=re_spoil_moment, t_asym_ramp=t_ramp_asym)
                re_t_flat = grad_instance.set_on_raster(duration_re_grad - t_ramp_unipolar - t_ramp_bipolar)
                amps.extend([re_grad_amplitude, re_grad_amplitude])
                times.extend([t + t_ramp_asym, t + t_ramp_asym + re_t_flat])
            else:
                # want to minimize timing of gradient - use max grad
                re_grad_amplitude = np.sign(re_spoil_moment) * system.max_grad
                duration_re_grad = get_asym_grad_min_duration(
                    max_amplitude=re_grad_amplitude, moment=re_spoil_moment, t_asym_ramp=t_ramp_asym
                )
                re_t_flat = grad_instance.set_on_raster(duration_re_grad - t_ramp_asym - t_ramp_unipolar)
                if adjust_ramp_area:
                    # adjust amplitude after all timing calculated
                    re_grad_amplitude = get_asym_grad_amplitude(
                        duration=duration_re_grad, moment=re_spoil_moment - adjust_ramp_area, t_asym_ramp=t_ramp_asym
                    )
                amps.extend([re_grad_amplitude, re_grad_amplitude])
                times.extend([t + t_ramp_asym, t + t_ramp_asym + re_t_flat])
            t += duration_re_grad
        # ramp down
        amps.append(0.0)
        times.append(t)

        # end of re gradient
        re_end_time = t
        duration_re_grad = float(re_end_time - re_start_time)
        amps = np.array(amps)
        times = np.array(times)

        times = np.array(times)
        grad_instance.channel = 'z'
        grad_instance.amplitude = np.array(amps)
        grad_instance.area = np.array(areas)
        grad_instance.flat_area = np.zeros(3)

        grad_instance.t_array_s = times
        grad_instance.t_delay_s = 0.0
        grad_instance.t_fall_time_s = times[-1] - times[-2]
        grad_instance.t_rise_time_s = times[1] - times[0]
        grad_instance.t_flat_time_s = 0.0

        grad_instance.system = system

        grad_instance.max_slew = system.max_slew
        grad_instance.max_grad = system.max_grad
        grad_instance.t_duration_s = grad_instance.get_duration()
        # last sanity check max grad / slew times
        if np.max(np.abs(amps)) > system.max_grad:
            err = f"amplitude violation, maximum gradient exceeded"
            logModule.error(err)
            raise ValueError(err)
        grad_slew = np.abs(np.diff(amps) / np.diff(times))
        if np.max(grad_slew) > system.max_slew:
            err = f"slew rate violation, maximum slew rate exceeded"
            logModule.error(err)
            raise ValueError(err)

        return grad_instance, delay, duration_re_grad

    @classmethod
    def make_slice_selective_rxiv(
            cls, pulse_bandwidth_hz: float, slice_thickness_m: float, duration_s: float,
            system: pp.Opts, pre_moment: float = 0.0, re_spoil_moment: float = 0.0,
            rephase: float = 0.0, min_re_time: float = 0.0):
        grad_instance = cls()
        grad_instance.system = system
        duration_s, rf_raster_delay = grad_instance.set_on_raster(duration_s, return_delay=True)
        # set slice select amplitude
        amplitude = pulse_bandwidth_hz / slice_thickness_m



        grad_simple_ns = pp.make_trapezoid(
            channel='z', amplitude=amplitude, flat_time=duration_s,
            system=system
        )

        amps = [0.0]
        times = [0.0]
        areas = []

        if np.abs(pre_moment) > 1e-7:
            grad_pre_simple_ns = pp.make_trapezoid(
                channel='z', area=pre_moment, system=system
            )
            # amplitudes
            pre_amps = [
                grad_pre_simple_ns.amplitude,
                grad_pre_simple_ns.amplitude,
                amplitude
            ]
            # times
            # ramp from pre to slice select
            diff_amps = np.abs(amplitude - grad_pre_simple_ns.amplitude)
            ramp_time_pre = grad_instance.set_on_raster(diff_amps / system.max_slew)
            # calculate ramp area
            ramp_area = ramp_time_pre * np.min([amplitude, grad_pre_simple_ns.amplitude]) + 0.5 * diff_amps * ramp_time_pre
            # subtract original ramp
            ramp_area -= 0.5 * grad_pre_simple_ns.amplitude * grad_pre_simple_ns.fall_time

            # want the pre gradient to be adapted by this area.
            t_adjust = grad_instance.set_on_raster(ramp_area / grad_pre_simple_ns.amplitude)
            t = grad_pre_simple_ns.rise_time + grad_pre_simple_ns.flat_time + ramp_time_pre - t_adjust

            pre_times = [
                grad_pre_simple_ns.rise_time,
                grad_pre_simple_ns.rise_time + grad_pre_simple_ns.flat_time,
                t
            ]

            amps.extend(pre_amps)
            times.extend(pre_times)
            areas.append(grad_pre_simple_ns.area)
        else:
            # starting point
            amps.append(amplitude)
            times.append(grad_simple_ns.rise_time)
            t = grad_simple_ns.rise_time

        delay = times[-1] + rf_raster_delay
        # flat part slice select
        amps.append(amplitude)
        # save time + also start of rephasing / spoiler grad
        t += grad_simple_ns.flat_time
        re_start_time = t
        times.append(t)
        fall_time = grad_simple_ns.fall_time
        areas.append(grad_simple_ns.area)

        if np.abs(re_spoil_moment) > 1e-7:
            re_spoil_moment += - 0.5 * rephase * grad_simple_ns.area
            grad_re_simple_ns = pp.make_trapezoid(
                channel='z', area=re_spoil_moment, system=system
            )
            if pp.calc_duration(grad_re_simple_ns) < min_re_time:
                logModule.info(
                    f"adjusting re / spoiler gradient length due to time constraint {1e3 * min_re_time:.2f} ms")
                grad_re_simple_ns = pp.make_trapezoid(
                    channel='z', area=re_spoil_moment, system=system, duration=min_re_time
                )
            # ramp to re spoil moment
            diff_amps = np.abs(amplitude - grad_re_simple_ns.amplitude)
            ramp_time_re = grad_instance.set_on_raster(
                diff_amps / system.max_slew
            )
            # here we need to check for a sign change in case of rephasing excitation grads
            if np.sign(amplitude) != np.sign(grad_re_simple_ns.ampltiude):
                total_triangular_area = ramp_time_re * diff_amps * 0.5
                sum_amps = np.abs(amplitude) + np.abs(grad_re_simple_ns)
                area_re = np.abs(grad_re_simple_ns.amplitude) / sum_amps * total_triangular_area
                area_deficient = np.abs(amplitude) / sum_amps * total_triangular_area
                # want total area to be grad_re area but exchanged the ramp up with the ramp from slice select grad
                area_exchange = area_re - area_deficient
            else:
                # calculate ramp area
                ramp_area = ramp_time_re * np.min([amplitude, grad_re_simple_ns.amplitude]) + 0.5 * diff_amps * ramp_time_re
                # subtract original ramp area
                ramp_area -= 0.5 * grad_re_simple_ns.amplitude * grad_re_simple_ns.fall_time

            # want the pre gradient to be adapted by this area.
            t_adjust = grad_instance.set_on_raster(ramp_area / grad_re_simple_ns.amplitude)

            t += ramp_time_re
            amps.append(grad_re_simple_ns.amplitude)
            times.append(t)
            # flat part
            if grad_re_simple_ns.flat_time > 1e-6:
                # save the time in variable, adjust for ramp area
                t += grad_re_simple_ns.flat_time - t_adjust
                amps.append(grad_re_simple_ns.amplitude)
                times.append(t)

            fall_time = grad_re_simple_ns.fall_time
            areas.append(grad_re_simple_ns.area)

        # ramp down
        amps.append(0)
        t += fall_time
        times.append(t)
        # end of re gradient
        re_end_time = t
        duration_re_grad = float(re_end_time - re_start_time)

        times = np.array(times)
        grad_instance.channel = 'z'
        grad_instance.amplitude = np.array(amps)
        grad_instance.area = np.array(areas)
        grad_instance.flat_area = np.zeros(3)

        grad_instance.t_array_s = times
        grad_instance.t_delay_s = 0.0
        grad_instance.t_fall_time_s = times[-1] - times[-2]
        grad_instance.t_rise_time_s = times[1] - times[0]
        grad_instance.t_flat_time_s = 0.0

        grad_instance.system = system

        grad_instance.max_slew = system.max_slew
        grad_instance.max_grad = system.max_grad
        grad_instance.t_duration_s = grad_instance.get_duration()
        return grad_instance, delay, duration_re_grad

    @classmethod
    def sym_grad(cls, system: pp.Opts, channel: str = 'x', pre_delay: float = 0.0, area_lobe: float = 0.0,
                 amplitude_lobe: float = 0.0, duration_lobe: float = 0.0, duration_between: float = 0.0,
                 reverse_second_lobe: bool = False):
        grad_instance = cls()
        grad_instance.system = system
        duration_lobe = grad_instance.set_on_raster(float(duration_lobe))
        duration_between = grad_instance.set_on_raster(duration_between)

        grad_ns = pp.make_trapezoid(
            channel=channel, area=area_lobe, amplitude=amplitude_lobe, duration=duration_lobe, system=system
        )

        # set up arrays
        grad_instance.t_delay_s = pre_delay
        times = np.array([
            0.0,
            grad_ns.rise_time,
            grad_ns.rise_time + grad_ns.flat_time,
            duration_lobe,
            duration_lobe + duration_between,
            duration_lobe + duration_between + grad_ns.rise_time,
            duration_lobe + duration_between + grad_ns.rise_time + grad_ns.flat_time,
            2 * duration_lobe + duration_between,
        ])
        # for second lobe
        sign = 1
        if reverse_second_lobe:
            sign *= -1
        amps = np.array([
            0.0,
            grad_ns.amplitude,
            grad_ns.amplitude,
            0.0,
            0.0,
            sign * grad_ns.amplitude,
            sign * grad_ns.amplitude,
            0.0
        ])

        areas = [grad_ns.area, 0, sign * grad_ns.area]

        grad_instance.channel = channel
        grad_instance.amplitude = np.array(amps)
        grad_instance.area = np.array(areas)

        grad_instance.t_array_s = times
        grad_instance.t_fall_time_s = times[-1] - times[-2]
        grad_instance.t_rise_time_s = times[1] - times[0]

        grad_instance.system = system

        grad_instance.max_slew = system.max_slew
        grad_instance.max_grad = system.max_grad
        grad_instance.t_duration_s = grad_instance.get_duration()
        return grad_instance

    def get_duration(self):
        # 0 for empty init grad
        if self.t_array_s.__len__() < 1:
            return 0.0
        return self.t_array_s[-1]

    def to_simple_ns(self):
        return types.SimpleNamespace(
            channel=self.channel, type='grad',
            delay=self.t_delay_s, first=self.amplitude[0], last=self.amplitude[-1],
            shape_dur=self.t_duration_s, tt=self.t_array_s, waveform=self.amplitude
        )

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        gamma = self.system.gamma
        amplitude = self.amplitude / gamma * 1e3
        ax.plot(self.t_array_s, amplitude)
        plt.show()


class ADC(Event):
    def __init__(self):
        super().__init__()
        self.num_samples: int = 0

        self.t_dwell_s: float = 0.0
        self.t_delay_s: float = 0.0
        self.t_duration_s: float = 0.0
        self.t_dead_time_s: float = 0.0

        self.freq_offset_hz: float = 0.0
        self.phase_offset_rad: float = 0.0

        self.system: pp.Opts = pp.Opts()

    @classmethod
    def make_adc(cls, system: pp.Opts,
                 num_samples: int = 0, delay_s: float = 0, duration_s: float = 0,
                 dwell: float = 0, freq_offset_hz: float = 0, phase_offset_rad: float = 0.0
                 ):
        adc_ns = pp.make_adc(
            num_samples=num_samples,
            delay=delay_s,
            duration=duration_s,
            dwell=dwell,
            freq_offset=freq_offset_hz,
            phase_offset=phase_offset_rad,
            system=system
        )
        adc_instance = cls()
        adc_instance.system = system
        adc_instance.num_samples = adc_ns.num_samples
        adc_instance.t_delay_s = adc_ns.delay
        adc_instance.t_dwell_s = adc_ns.dwell
        adc_instance.t_duration_s = adc_ns.dwell*adc_ns.num_samples
        adc_instance.t_dead_time_s = adc_ns.dead_time
        adc_instance.freq_offset_hz = adc_ns.freq_offset
        adc_instance.phase_offset_rad = adc_ns.phase_offset
        return adc_instance

    def get_duration(self):
        return self.t_duration_s

    def to_simple_ns(self):
        return types.SimpleNamespace(
            dead_time=self.t_dead_time_s, delay=self.t_delay_s, dwell=self.t_dwell_s,
            freq_offset=self.freq_offset_hz, num_samples=self.num_samples,
            phase_offset=self.phase_offset_rad, type='adc'
        )


class DELAY(Event):
    def __init__(self):
        super().__init__()
        self.system = pp.Opts()

    @classmethod
    def make_delay(cls, delay: float, system: pp.Opts = pp.Opts()):
        delay_instance = cls()
        delay_instance.system = system
        delay_instance.t_duration_s = delay
        return delay_instance

    def check_on_block_raster(self) -> bool:
        us_raster = 1e6 * self.system.grad_raster_time
        us_value = 1e6 * self.t_duration_s
        if us_value % us_raster < 1e-4:
            rastered_value = us_value * 1e-6
        else:
            rastered_value = np.round(us_value / us_raster) * us_raster
        if np.abs(rastered_value - self.t_duration_s) > 1e-8:
            return False
        else:
            return True

    def set_on_block_raster(self):
        us_raster = 1e6 * self.system.grad_raster_time
        us_value = 1e6 * self.t_duration_s
        if us_value % us_raster < 1e-4:
            rastered_value = us_value * 1e-6
        else:
            rastered_value = np.round(us_value / us_raster) * us_raster * 1e-6
        return rastered_value

    def get_duration(self):
        return self.t_duration_s

    def to_simple_ns(self):
        return types.SimpleNamespace(delay=self.t_duration_s, type='delay')


if __name__ == '__main__':
    rf = pp.make_sinc_pulse(
        np.pi/2,
        system=pp.Opts(),
        use='excitation'
    )
    rf_new = RF.make_sinc_pulse(
        flip_angle_rad=np.pi/2,
        pulse_type='excitation',
        system=pp.Opts()
    )
    rf_new_ns = rf_new.to_simple_ns()
    logModule.info("compare rf")

    grad_ns = pp.make_extended_trapezoid(
        'z',
        amplitudes=np.array([0, 755857.8987, 755857.8987, 0]),
        times=np.array([0, 0.00011, 0.00189, 0.002])
    )
    grad_new = GRAD.make_trapezoid('z', system=pp.Opts(), area=1/0.7*1e3, duration_s=2e-3)
    grad_new_ns = grad_new.to_simple_ns()
    logModule.info("compare grad")

    adc_ns = pp.make_adc(num_samples=304, duration=3e-3)
    adc_new = ADC.make_adc(system=pp.Opts(), num_samples=304, duration_s=3e-3)
    adc_new_ns = adc_new.to_simple_ns()

    logModule.info("compare adc")

    delay_ns = pp.make_delay(1e-3)
    delay_new = DELAY.make_delay(1e-3)

    logModule.info("compare delays")
