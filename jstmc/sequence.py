"""
Caution: grad amplitudes of pypulseq methods are in rad!
need to divide by 2pi for Hz
"""

import logging
import typing
import types
from jstmc import options
import numpy as np
import pypulseq as pp
import tqdm
import pathlib as plib

logModule = logging.getLogger(__name__)


def set_on_grad_raster_time(time: float, system: pp.Opts):
    return np.ceil(time / system.grad_raster_time) * system.grad_raster_time


def load_external_rf(rf_file) -> np.ndarray:
    """
        if pulse profile is provided, read in
        :param filename: name of file (txt or pta) of pulse
        :return: pulse array, pulse length
        """
    # load file content
    ext_file = plib.Path(rf_file)
    with open(ext_file, "r") as f:
        content = f.readlines()

    # find line where actual data starts
    start_count = -1
    while True:
        start_count += 1
        line = content[start_count]
        start_line = line.strip().split('\t')[0]
        if start_line.replace('.', '', 1).isdigit():
            break

    # read to array
    content = content[start_count:]
    temp = [line.strip().split('\t')[0] for line in content]

    pulseShape = np.array(temp, dtype=float)
    return pulseShape


    pass


class Acquisition:
    def __init__(self, params: options.SequenceParameters, system: pp.Opts):
        self.params = params
        self.system = system
        # grad
        self.read_grad: types.SimpleNamespace = types.SimpleNamespace()
        self.read_grad_pre: types.SimpleNamespace = types.SimpleNamespace()
        self.read_grad_spoil: types.SimpleNamespace = types.SimpleNamespace()
        # adc
        self.adc: types.SimpleNamespace = types.SimpleNamespace()
        # phase
        self.phase_grad_areas: np.ndarray = np.zeros(0)
        self.phase_grad_pre_adc: types.SimpleNamespace = types.SimpleNamespace()
        self.phase_grad_post_adc: types.SimpleNamespace = types.SimpleNamespace()
        # timing
        self.t_phase: float = 0.0  # time needed for longest phase enc -> all phase enc
        self.t_read_pre: float = 0.0
        # init
        self._make_read_gradients()
        self._set_phase_areas()

    # methods private
    def _make_read_gradients(self):
        acquisition_window = set_on_grad_raster_time(self.params.acquisitionTime, system=self.system)
        self.read_grad = pp.make_trapezoid(
            channel=self.params.read_dir,
            flat_area=self.params.deltaK_read * self.params.resolutionNRead,
            flat_time=acquisition_window,  # given in [s] via options
            system=self.system
        )
        self.read_grad_pre = pp.make_trapezoid(
            channel=self.params.read_dir,
            area=self.read_grad.area / 2,
            system=self.system
        )
        self.read_grad_spoil = pp.make_trapezoid(
            channel=self.params.read_dir,
            area=self.read_grad.area / 3,
            system=self.system
        )
        self.t_read_pre = pp.calc_duration(self.read_grad_pre)
        # set adc
        # check timing
        acquisition_time = self.params.dwell * int(self.params.resolutionNRead * self.params.oversampling)
        if acquisition_window < acquisition_time:
            err = "adc timing not compatible with read gradient"
            logModule.error(err)
            raise ValueError(err)

        delay = (acquisition_window - acquisition_time) / 2 + self.read_grad.rise_time
        self.adc = pp.make_adc(
            num_samples=int(self.params.resolutionNRead * self.params.oversampling),
            delay=delay,
            dwell=self.params.dwell,
            system=self.system)

    def _set_phase_areas(self):
        self.phase_grad_areas = (- np.arange(self.params.resolutionNPhase) + self.params.resolutionNPhase / 2) * \
                                self.params.deltaK_phase
        # build longest phase gradient
        gPhase_max = pp.make_trapezoid(
            channel=self.params.phase_dir,
            area=np.max(self.phase_grad_areas),
            system=self.system
        )
        # calculate time needed for biggest phase grad
        self.t_phase = set_on_grad_raster_time(pp.calc_duration(gPhase_max), self.system)

    def set_phase_grads(self, idx_phase):
        if np.abs(self.phase_grad_areas[idx_phase]) > 0:
            # calculate phase step
            self.phase_grad_pre_adc = pp.make_trapezoid(
                channel=self.params.phase_dir,
                area=self.phase_grad_areas[idx_phase],
                duration=self.t_phase,
                system=self.system
            )
            self.phase_grad_post_adc = pp.make_trapezoid(
                channel=self.params.phase_dir,
                area=-self.phase_grad_areas[idx_phase],
                duration=self.t_phase,
                system=self.system
            )
        else:
            self.phase_grad_pre_adc = pp.make_delay(self.t_phase)
            self.phase_grad_post_adc = pp.make_delay(self.t_phase)

    def reset_read_grad_pre(self, t_read_grad_pre: float):
        self.read_grad_pre = pp.make_trapezoid(
            channel=self.params.read_dir,
            area=self.read_grad_pre.area,
            duration=t_read_grad_pre,
            system=self.system
        )
        self.read_grad_spoil = pp.make_trapezoid(
            channel=self.params.read_dir,
            area=self.read_grad_spoil.area,
            duration=t_read_grad_pre,
            system=self.system
        )

    def get_t_read_grad_pre(self) -> float:
        return self.t_read_pre

    def reset_t_phase(self, t_phase: float):
        self.t_phase = t_phase

    def get_t_phase(self) -> float:
        return self.t_phase


class SliceGradPulse:
    def __init__(self, params: options.SequenceParameters, system: pp.Opts, t_xy_grad: float,
                 is_excitation: bool):
        self.params = params
        self.system = system
        self.t_xy_grad = t_xy_grad
        self.is_excitation = is_excitation

        # slice
        self.slice_grad: types.SimpleNamespace = types.SimpleNamespace()
        self.slice_grad_from_zero: types.SimpleNamespace = types.SimpleNamespace()
        # pre gradient, this is pre moment upon excitation and symmetric spoil upon refocusing
        self.slice_grad_pre: types.SimpleNamespace = types.SimpleNamespace()
        # if excitation we rephase slice grad and spoil, if refocusing this is the spoiler grad
        self.slice_grad_post: types.SimpleNamespace = types.SimpleNamespace()
        # last spoiling grad needs to be separate
        self.slice_grad_post_from_zero: types.SimpleNamespace = types.SimpleNamespace()

        if self.is_excitation:
            flip_angle_rad = self.params.excitationRadFA
            phase_rad = self.params.excitationRadRfPhase
            time_bw_prod = self.params.excitationTimeBwProd
            duration = self.params.excitationDuration * 1e-6
        else:
            flip_angle_rad = self.params.refocusingRadFA  # these are lists now!
            phase_rad = self.params.refocusingRadRfPhase  # these are lists now!
            time_bw_prod = self.params.refocusingTimeBwProd
            duration = self.params.refocusingDuration * 1e-6

        slice_thickness = self.params.resolutionSliceThickness * 1e-3
        duration = set_on_grad_raster_time(duration, system=self.system)

        # rf
        self.rf: typing.Union[list, types.SimpleNamespace] = types.SimpleNamespace()
        # delay
        self.delay: types.SimpleNamespace = pp.make_delay(0.0)

        # timing
        self.t_re_spoil: float = 0.0

        # init
        # build rf gradient pulse
        self._make_rf_grad_pulse(
            flip_angle_rad=flip_angle_rad,
            phase_rad=phase_rad,
            time_bw_prod=time_bw_prod,
            duration=duration,
            slice_thickness=slice_thickness
        )
        if self.is_excitation:
            # if excitation we pre wind
            self.slice_grad_pre: types.SimpleNamespace = pp.make_trapezoid(
                'z',
                area=-self.params.excitationPreMoment,
                max_slew=self.system.max_slew
            )
            # if excitation we adjust rephasing + spoiling
            self._recalculate_rephase_grad()
        else:
            # if refocusing we set spoiler grads
            self.slice_grad_pre = self._make_spoiler_gradient()
            self.slice_grad_post = self._make_spoiler_gradient()
            self.slice_grad_post_from_zero = self._make_spoiler_gradient()

        self._merge_grads()  # merge slice gradients to continuous waveform

    def _make_rf_grad_pulse(self, flip_angle_rad: typing.Union[float, np.ndarray, list],
                            phase_rad: typing.Union[float, np.ndarray, list], time_bw_prod: float,
                            duration: float, slice_thickness: float):
        # need to convert to list if only single value
        if isinstance(flip_angle_rad, float):
            flip_angle_rad = [flip_angle_rad]
        if isinstance(phase_rad, float):
            phase_rad = [phase_rad]
        # cast all to array
        flip_angle_rad = np.asarray(flip_angle_rad)
        phase_rad = np.asarray(phase_rad)
        if self.params.useExtRf:

            logModule.info(f"Loading external RF File: {self.params.useExtRf}")
            rf, self.slice_grad, slice_grad_re = self._make_ext_rf_pulse(
                flip_angle_rad=flip_angle_rad[0], phase_rad=phase_rad[0],
                duration=duration, slice_thickness=slice_thickness
            )
        else:
            rf, self.slice_grad, slice_grad_re = self._make_sinc_pulse(
                flip_angle_rad=flip_angle_rad[0], phase_rad=phase_rad[0],
                tbw=time_bw_prod, duration=duration,
                slice_thickness=slice_thickness
            )

        if self.is_excitation:
            self.rf = rf
            self.rf.init_phase = self.rf.phase_offset
            self.slice_grad_post = slice_grad_re
        else:
            self.rf = []
            for k in range(flip_angle_rad.__len__()):
                if self.params.useExtRf:
                    rf, _, _ = self._make_ext_rf_pulse(
                        flip_angle_rad=flip_angle_rad[k],
                        phase_rad=phase_rad[k],
                        delay=rf.delay,
                        duration=duration,
                        slice_thickness=slice_thickness
                    )
                else:
                    rf, _, _ = self._make_sinc_pulse(
                        flip_angle_rad=flip_angle_rad[k],
                        phase_rad=phase_rad[k],
                        delay=rf.delay,
                        tbw=time_bw_prod,
                        duration=duration,
                        slice_thickness=slice_thickness
                    )
                rf.init_phase = phase_rad[k]
                self.rf.append(rf)

    def _make_sinc_pulse(self, flip_angle_rad, phase_rad, tbw, duration, slice_thickness, delay=0.0):
        if self.is_excitation:
            use = "excitation"
            # can change apodization here
            apodization = 0.0
        else:
            use = "refocusing"
            # can change apodization here
            apodization = 0.0
        return pp.make_sinc_pulse(
            flip_angle=flip_angle_rad,
            phase_offset=phase_rad,
            delay=delay,
            apodization=apodization,
            time_bw_product=tbw,
            duration=duration,
            max_slew=self.system.max_slew,
            system=self.system,
            slice_thickness=slice_thickness,
            return_gz=True,
            use=use
        )

    def _make_ext_rf_pulse(self, flip_angle_rad, phase_rad, duration, slice_thickness, delay=0.0):
        if self.is_excitation:
            use = "excitation"
            scale_gz = 1.0
        else:
            use = "refocusing"
            scale_gz = 2/3
        bandwidth = 1.92 / duration   # for gauss pulse tbw = 1.92 -> specific to external pulse
        N = int(duration / self.system.rf_raster_time)
        pulse_shape = load_external_rf(self.params.useExtRf)
        # interpolate shape to duration raster
        shape_to_duration = np.interp(
            np.linspace(0, pulse_shape.shape[0], N),
            np.arange(pulse_shape.shape[0]),
            pulse_shape
        )
        rf, slice_grad = pp.make_arbitrary_rf(
            signal=shape_to_duration, flip_angle=flip_angle_rad, phase_offset=phase_rad, bandwidth=bandwidth,
            delay=delay, max_slew=self.system.max_slew, return_gz=True,
            slice_thickness=slice_thickness, system=self.system, use=use
        )
        slice_grad.amplitude *= scale_gz
        slice_grad_re = pp.make_trapezoid(
            channel="z",
            system=self.system,
            area=-slice_grad.area * 0.5,
        )
        return rf, slice_grad, slice_grad_re

    def _recalculate_rephase_grad(self):
        # calculate spoil grad area -> cast thickness from mm to m
        spoil_area = - self.params.spoilerScaling * 1e3 / self.params.resolutionSliceThickness
        # reset rephaser
        self.slice_grad_post = pp.make_trapezoid(
            channel='z',
            area=spoil_area + self.slice_grad_post.area,
            system=self.system
        )
        self.t_re_spoil = set_on_grad_raster_time(pp.calc_duration(self.slice_grad_post), system=self.system)
        if self._check_timing_changed():
            # if we need to adjust timing do so
            self.slice_grad_post = pp.make_trapezoid(
                channel='z',
                area=self.slice_grad_post.area,
                system=self.system,
                duration=self.t_re_spoil
            )

    def _make_spoiler_gradient(self):
        slice_grad_spoil = pp.make_trapezoid(
            channel='z',
            area=- self.params.spoilerScaling * 1e3 / self.params.resolutionSliceThickness,
            # slice thickness given in mm
            system=self.system
        )
        self.t_re_spoil = set_on_grad_raster_time(pp.calc_duration(slice_grad_spoil), system=self.system)
        if self._check_timing_changed():
            # if we need to adjust timing do so
            slice_grad_spoil = pp.make_trapezoid(
                channel='z',
                area=slice_grad_spoil.area,
                # slice thickness given in mm
                system=self.system,
                duration=self.t_re_spoil
            )
        return slice_grad_spoil

    def _check_timing_changed(self) -> bool:
        # set longer time after excitation and optimize gradients to timing
        if self.t_re_spoil > self.t_xy_grad:
            logModule.debug(f"Rephaser / Spoiler time restricting: {self.t_re_spoil * 1e3:.2f} ms")
            return False
        else:
            logModule.debug(f"Read or Phase gradient time restricting: {self.t_xy_grad * 1e3:.2f} ms")
            self.t_re_spoil = set_on_grad_raster_time(self.t_xy_grad, system=self.system)
            return True

    def _merge_grads(self):
        # quick sanity checks
        rf = self.rf
        if isinstance(self.rf, list):
            rf = self.rf[0]
        if np.abs(rf.delay - self.slice_grad.rise_time - self.slice_grad.delay) > self.system.grad_raster_time:
            err = f"rf delay ({1e3 * rf.delay:.2f} ms) != slice selective gradient rise time (" \
                  f"{1e3 * self.slice_grad.rise_time:.2f} ms). But events are supposed to start in same block"
            logModule.error(err)
            raise ValueError(err)
        if rf.ringdown_time > self.slice_grad.fall_time:
            err = f"rf ringdown extends beyond slice selective gradient!"
            logModule.error(err)
            raise ValueError(err)

        rise = set_on_grad_raster_time(self.slice_grad.rise_time,self.system)
        # sanity check rise and fall times to equal
        if np.abs(self.slice_grad.fall_time - rise) > 0:
            err = f"rise time and fall time not equal"
            logModule.error(err)
            raise ValueError(err)

        amp = self.slice_grad.amplitude
        flat_time = set_on_grad_raster_time(self.slice_grad.flat_time, self.system)

        # --- merge at edges ---
        # we want to interpolate between the pre gradient ramp down and slice selection ramp up
        t_rd_ru = self.slice_grad_pre.fall_time + rise
        interpol_gradient_pre_to_slice_sel = np.divide(
            self.slice_grad_pre.amplitude - self.slice_grad.amplitude,
            t_rd_ru
        ) * self.slice_grad.rise_time + self.slice_grad.amplitude
        # we want to interpolate between the slice selection ramp down and the rephasing ramp up,
        t_rd_ru = rise + self.slice_grad_post.rise_time
        interpol_gradient_slice_sel_to_spoil = np.divide(
            self.slice_grad_post.amplitude - self.slice_grad.amplitude,
            t_rd_ru
        ) * self.slice_grad.fall_time + self.slice_grad.amplitude

        # slice selection core part: interpolation from pre + flat + interpolation to post
        self.slice_grad = pp.make_extended_trapezoid(
            'z',
            amplitudes=np.array([interpol_gradient_pre_to_slice_sel, amp, amp, interpol_gradient_slice_sel_to_spoil]),
            times=np.array([0, rise, rise + flat_time, 2 * rise + flat_time])
        )

        # pre to slice selection
        t_arr = np.array([
            0.0,
            self.slice_grad_pre.rise_time,
            self.slice_grad_pre.rise_time + self.slice_grad_pre.flat_time,
            self.slice_grad_pre.rise_time + self.slice_grad_pre.flat_time + self.slice_grad_pre.fall_time
        ])
        amps = np.array([
            0.0,
            self.slice_grad_pre.amplitude,
            self.slice_grad_pre.amplitude,
            interpol_gradient_pre_to_slice_sel
        ])
        self.slice_grad_pre = pp.make_extended_trapezoid(
            'z',
            amplitudes=amps,
            times=t_arr
        )
        # slice selection to post
        # interpolate gradient re/spoiler
        t_arr = np.array([
            0.0,
            self.slice_grad_post.rise_time,
            self.slice_grad_post.flat_time + self.slice_grad_post.rise_time,
            self.slice_grad_post.flat_time + 2 * self.slice_grad_post.rise_time
        ])
        amps = np.array([
            interpol_gradient_slice_sel_to_spoil,
            self.slice_grad_post.amplitude,
            self.slice_grad_post.amplitude,
            0.0
        ])
        re_spoil_amp = self.slice_grad_post.amplitude
        self.slice_grad_post = pp.make_extended_trapezoid('z', amplitudes=amps, times=t_arr)
        self.slice_grad_post.amplitude = re_spoil_amp

        logModule.debug(f"slice sel grad: {1e3 * self.slice_grad.last / self.system.gamma:.2f} mT/m")
        logModule.debug(
            f"rephasing grad: {1e3 * self.slice_grad_post.amplitude / self.system.gamma:.2f} mT/m"
        )
        logModule.debug(
            f"spoiling grad: {1e3 * self.slice_grad_post.amplitude / self.system.gamma:.2f} mT/m"
        )
        self.slice_grad.amplitude = amp

        # additionally we keep one slice selection from ramp up
        self.slice_grad_from_zero = pp.make_extended_trapezoid(
            'z',
            amplitudes=np.array([
                0.0, self.slice_grad.amplitude, self.slice_grad.amplitude, interpol_gradient_slice_sel_to_spoil]),
            times=np.array([
                0.0, rise, rise + flat_time, 2 * rise + flat_time
            ])
        )

    def check_post_slice_selection_timing(self):
        return not self._check_timing_changed()

    def get_timing_post_slice_selection(self):
        return self.t_re_spoil


class SequenceBlockEvents:
    def __init__(self, seq: options.Sequence):
        self.seq = seq
        # ___ define all block event vars ___
        # Acquisition
        logModule.info("Setting up Acquisition")
        # Excitation
        self.acquisition = Acquisition(params=self.seq.params, system=self.seq.ppSys)
        logModule.info("Setting up Excitation")
        self.excitation = SliceGradPulse(
            params=self.seq.params,
            system=self.seq.ppSys,
            t_xy_grad=self.acquisition.get_t_read_grad_pre(),
            is_excitation=True
        )
        if self.excitation.check_post_slice_selection_timing():
            logModule.info(f"Excitation rephase timing longer than readout prephasing, readjusting readout pre")
            self.acquisition.reset_read_grad_pre(self.excitation.get_timing_post_slice_selection())
        # Refocusing
        logModule.info("Setting up Refocusing")
        self.refocusing = SliceGradPulse(
            params=self.seq.params,
            system=self.seq.ppSys,
            t_xy_grad=self.acquisition.get_t_phase(),
            is_excitation=False)
        if self.refocusing.check_post_slice_selection_timing():
            logModule.info(f"Spoiling timing longer than phase encode, readjusting phase enc timing")
            self.acquisition.reset_t_phase(self.refocusing.get_timing_post_slice_selection())

        # Timing
        self.t_duration_echo_train: float = 0.0
        self.t_delay_slice: types.SimpleNamespace = pp.make_delay(0.0)
        self._calculate_min_esp()

        # k space
        self.k_indexes: np.ndarray = np.zeros(
            (self.seq.params.ETL, self.seq.params.numberOfCentralLines + self.seq.params.numberOfOuterLines),
            dtype=int
        )
        self.sampling_pattern: list = []
        # slice loop
        numSlices = self.seq.params.resolutionNumSlices
        self.z = np.zeros((2, int(np.ceil(numSlices / 2))))
        self.trueSliceNum = np.zeros(numSlices)

    def _write_emc_info(self) -> dict:
        emc_dict = {
            "gammaHz": self.seq.specs.gamma,
            "ETL": self.seq.params.ETL,
            "ESP": self.seq.params.ESP,
            "bw": self.seq.params.bandwidth,
            "gradMode": "Normal",
            "excitationAngle": self.seq.params.excitationRadFA / np.pi * 180.0,
            "excitationPhase": self.seq.params.excitationRfPhase,
            "gradientExcitation": self._set_grad_for_emc(self.excitation.slice_grad.amplitude),
            "durationExcitation": self.seq.params.excitationDuration,
            "gradientExcitationRephase": self._set_grad_for_emc(self.excitation.slice_grad_post.amplitude),
            "durationExcitationRephase": self.excitation.t_re_spoil * 1e6,
            "gradientExcitationVerse1": 0.0,
            "gradientExcitationVerse2": 0.0,
            "durationExcitationVerse1": 0.0,
            "durationExcitationVerse2": 0.0,
            "refocusAngle": self.seq.params.refocusingFA,
            "refocusPhase": self.seq.params.refocusingRfPhase,
            "gradientRefocus": self._set_grad_for_emc(self.refocusing.slice_grad.amplitude),
            "durationRefocus": self.seq.params.refocusingDuration,
            "gradientCrush": self._set_grad_for_emc(self.refocusing.slice_grad_post.amplitude),
            "durationCrush": self.refocusing.t_re_spoil * 1e6,
            "gradientRefocusVerse1": 0.0,
            "gradientRefocusVerse2": 0.0,
            "durationRefocusVerse1": 0.0,
            "durationRefocusVerse2": 0.0
        }
        return emc_dict

    def _set_grad_for_emc(self, grad):
        return 1e3 / self.seq.specs.gamma * grad

    def get_sampling_pattern(self) -> list:
        return self.sampling_pattern

    def get_pulse_amplitudes(self) -> np.ndarray:
        exc_pulse = self.excitation.rf.signal
        return exc_pulse

    def _calculate_min_esp(self):
        # find minimal echo spacing

        # between excitation and refocus = esp / 2 -> attention to calculating timing!
        # the rf starts with the merged grads to use them later as a block event, this introduces a delay
        # not part of the echo spacing. from the focal point of the rf we have half the duration + ringdown
        timing_excitation_refocus = pp.calc_duration(self.excitation.slice_grad) / 2 + \
                                    self.excitation.t_re_spoil + \
                                    self.refocusing.rf[0].shape_dur / 2 + \
                                    self.refocusing.rf[0].delay
        timing_excitation_refocus = set_on_grad_raster_time(timing_excitation_refocus, system=self.seq.ppSys)

        # between refocus and adc = esp / 2
        timing_refoucs_adc = pp.calc_duration(self.refocusing.slice_grad) / 2 + \
                             self.refocusing.t_re_spoil + \
                             pp.calc_duration(self.acquisition.read_grad) / 2
        timing_refoucs_adc = set_on_grad_raster_time(timing_refoucs_adc, system=self.seq.ppSys)

        # diff
        t_diff = set_on_grad_raster_time(np.abs(timing_refoucs_adc - timing_excitation_refocus), system=self.seq.ppSys)
        # choose longer time as half echo spacing
        if timing_refoucs_adc > timing_excitation_refocus:
            esp = 2 * timing_refoucs_adc
            self.excitation.delay = pp.make_delay(t_diff)
        else:
            esp = 2 * timing_excitation_refocus
            self.refocusing.delay = pp.make_delay(t_diff)
        self.seq.params.set_esp(esp)

        logModule.info(f"Found minimum TE: {esp * 1e3:.2f} ms")

        self.t_duration_echo_train = set_on_grad_raster_time(
            # prewind gradient
            pp.calc_duration(self.excitation.slice_grad_pre) +
            # first half of slice sel grad / rf
            pp.calc_duration(self.excitation.slice_grad) / 2 +
            # etl * esp -> we land on the middle of the last adc
            self.seq.params.ETL * esp +
            # last half of last adc
            pp.calc_duration(self.acquisition.read_grad) / 2 +
            # final spoiling
            pp.calc_duration(self.refocusing.slice_grad_post),
            system=self.seq.ppSys
        )
        logModule.info(f"echo train length: {self.t_duration_echo_train * 1e3:.2f} ms")

    def _calculate_num_slices(self):
        # calculate how many slices can be accommodated
        numSlices = np.min([
            self.seq.params.resolutionNumSlices,
            int(np.floor(self.seq.params.TR * 1e-3 / self.t_duration_echo_train))
        ])
        logModule.info(
            f"{int(np.floor(self.seq.params.TR * 1e-3 / self.t_duration_echo_train))} "
            f"Slices can be accommodated in one TR; "
            f"{self.seq.params.resolutionNumSlices} were desired"
        )
        if numSlices < self.seq.params.resolutionNumSlices:
            logModule.info(f"need concatenation!")

        delay_slice_time = set_on_grad_raster_time(
            self.seq.params.TR * 1e-3 / numSlices - self.t_duration_echo_train,
            system=self.seq.ppSys
        )
        self.t_delay_slice = pp.make_delay(delay_slice_time)
        logModule.info(f"Delay between slices: {self.t_delay_slice.delay * 1e3:.2f} ms")

    def _set_k_space(self):
        # calculate center of k space and indexes for full sampling band
        k_central_phase = round(self.seq.params.resolutionNPhase / 2)
        k_half_central_lines = round(self.seq.params.numberOfCentralLines / 2)
        # set indexes for start and end of full k space center sampling
        k_start = k_central_phase - k_half_central_lines
        k_end = k_central_phase + k_half_central_lines

        # The rest of the lines we will use tse style phase step blip between the echoes of one echo train
        # Trying random sampling, ie. pick random line numbers for remaining indices,
        # we dont want to pick the same positive as negative phase encodes to account for conjugate symmetry in k-space.
        # Hence, we pick from the positive indexes twice (thinking of the center as 0) without allowing for duplexes
        # and negate half the picks
        # calculate indexes
        k_remaining = np.arange(0, k_start)
        # build array with dim [num_slices, num_outer_lines] to sample different random scheme per slice
        for idx_echo in range(self.seq.params.ETL):
            # same encode for all echoes -> central lines
            self.k_indexes[idx_echo, :self.seq.params.numberOfCentralLines] = np.arange(k_start, k_end)
            # random encodes for different echoes
            k_indices = np.random.choice(
                k_remaining,
                size=self.seq.params.numberOfOuterLines,
                replace=False)
            k_indices[::2] = self.seq.params.resolutionNPhase - 1 - k_indices[::2]
            self.k_indexes[idx_echo, self.seq.params.numberOfCentralLines:] = np.sort(k_indices)

    def _set_delta_slices(self):
        # multi-slice
        numSlices = self.seq.params.resolutionNumSlices
        sliThick = self.seq.params.resolutionSliceThickness
        # there is one gap less than number of slices, cast  thickness from mm / gap from %
        delta_z = sliThick * (numSlices + self.seq.params.resolutionSliceGap / 100.0 * (numSlices - 1)) * 1e-3
        if self.seq.params.interleavedAcquisition:
            logModule.info("Set interleaved Acquisition")
            # want to go through the slices alternating from beginning and middle
            self.z.flat[:numSlices] = np.linspace((-delta_z / 2), (delta_z / 2), numSlices)
            # reshuffle slices mid+1, 1, mid+2, 2, ...
            self.z = self.z.transpose().flatten()[:numSlices]
        else:
            logModule.info("Set sequential Acquisition")
            self.z = np.linspace((-delta_z / 2), (delta_z / 2), numSlices)
        # find reshuffled slice numbers
        for idx_slice_num in range(numSlices):
            z_val = self.z[idx_slice_num]
            z_pos = np.where(np.unique(self.z) == z_val)[0][0]
            self.trueSliceNum[idx_slice_num] = z_pos

    def _apply_slice_offset(self, idx_slice: int, is_excitation: bool = True, pulse_num: int = 0):
        if is_excitation:
            # excitation
            grad_amplitude = self.excitation.slice_grad.amplitude
            rf = self.excitation.rf
        else:
            # refocus
            grad_amplitude = self.refocusing.slice_grad.amplitude
            rf = self.refocusing.rf[pulse_num]
        # apply slice offset -> caution grad_amp in rad!
        freq_offset = grad_amplitude * self.z[idx_slice]
        phase_offset = rf.init_phase - 2 * np.pi * freq_offset * pp.calc_rf_center(rf)[0]  # radiant again
        return freq_offset, phase_offset  # casting
        # freq to Hz, phase is in radiant here

    def _add_blocks_excitation_first_read(self, phase_idx: int, slice_idx: int):
        idx_phase = self.k_indexes[0, phase_idx]
        # set phase grads
        self.acquisition.set_phase_grads(idx_phase=idx_phase)
        # set pre phasing
        self.seq.ppSeq.add_block(self.excitation.slice_grad_pre)

        # excitation
        self.seq.ppSeq.add_block(self.excitation.rf, self.excitation.slice_grad)

        # rephasing
        self.seq.ppSeq.add_block(self.excitation.slice_grad_post, self.acquisition.read_grad_pre)

        # delay if necessary
        if self.excitation.delay.delay > 1e-6:
            self.seq.ppSeq.add_block(self.excitation.delay)

        # first refocus
        self.seq.ppSeq.add_block(self.refocusing.rf[0], self.refocusing.slice_grad_from_zero)

        # spoiling phase encode, delay if necessary
        self.seq.ppSeq.add_block(self.refocusing.slice_grad_post, self.acquisition.phase_grad_pre_adc)

        # delay if necessary
        if self.refocusing.delay.delay > 1e-6:
            self.seq.ppSeq.add_block(self.refocusing.delay)

        # read
        self.seq.ppSeq.add_block(self.acquisition.read_grad, self.acquisition.adc)

        # write sampling pattern
        sampling_index = {"pe_num": idx_phase, "slice_num": int(self.trueSliceNum[slice_idx]), "echo_num": 0}
        self.sampling_pattern.append(sampling_index)

    def _add_blocks_refocusing_adc(self, phase_idx: int, slice_idx: int):
        for contrast_idx in np.arange(1, self.seq.params.ETL):
            # delay if necessary
            if self.refocusing.delay.delay > 1e-6:
                self.seq.ppSeq.add_block(self.refocusing.delay)

            # dephase, spoil
            self.seq.ppSeq.add_block(self.acquisition.phase_grad_post_adc, self.refocusing.slice_grad_pre)

            # refocus
            self.seq.ppSeq.add_block(self.refocusing.rf[contrast_idx], self.refocusing.slice_grad)

            # spoil phase encode
            # order of indices (aka same phase encode per contrast or tse style phase encode change per contrast)
            # is encoded in array
            idx_phase = self.k_indexes[contrast_idx, phase_idx]

            # set phase
            self.acquisition.set_phase_grads(idx_phase=idx_phase)
            self.seq.ppSeq.add_block(
                self.acquisition.phase_grad_pre_adc, self.refocusing.slice_grad_post)
            # read
            self.seq.ppSeq.add_block(self.acquisition.read_grad, self.acquisition.adc)

            # write sampling pattern
            sampling_index = {"pe_num": idx_phase, "slice_num": int(self.trueSliceNum[slice_idx]),
                              "echo_num": contrast_idx}
            self.sampling_pattern.append(sampling_index)

        # spoil end
        self.seq.ppSeq.add_block(
            self.acquisition.phase_grad_post_adc,
            self.refocusing.slice_grad_post_from_zero,
            self.acquisition.read_grad_spoil
        )
        self.seq.ppSeq.add_block(self.t_delay_slice)

    def _loop_lines(self):
        # through phase encodes
        line_bar = tqdm.trange(
            self.seq.params.numberOfCentralLines + self.seq.params.numberOfOuterLines, desc="phase encodes"
        )
        for idx_n in line_bar:  # We have N phase encodes for all ETL contrasts
            for idx_slice in range(self.seq.params.resolutionNumSlices):
                # apply slice offset
                self.excitation.rf.freq_offset, self.excitation.rf.phase_offset = self._apply_slice_offset(
                    idx_slice=idx_slice,
                    is_excitation=True
                )
                for idx_rf in range(self.seq.params.ETL):
                    self.refocusing.rf[idx_rf].freq_offset, self.refocusing.rf[idx_rf].phase_offset = self._apply_slice_offset(
                        idx_slice=idx_slice,
                        is_excitation=False,
                        pulse_num=idx_rf
                    )

                # excitation to first read
                self._add_blocks_excitation_first_read(phase_idx=idx_n, slice_idx=idx_slice)

                # refocusing blocks
                self._add_blocks_refocusing_adc(phase_idx=idx_n, slice_idx=idx_slice)

    def build(self):
        # calculate number of slices
        self._calculate_num_slices()
        # set k-space sampling indices
        self._set_k_space()
        # set positions for slices
        self._set_delta_slices()

        # loop through phase encode line building blocks
        self._loop_lines()

    def get_seq(self):
        # write info into seq obj
        self._write_emc_info()
        return self.seq

    def get_emc_info(self) -> dict:
        return self._write_emc_info()
