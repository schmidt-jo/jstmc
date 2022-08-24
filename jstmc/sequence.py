import logging
import types
from jstmc import options
import numpy as np
import pypulseq as pp
import tqdm

logModule = logging.getLogger(__name__)


def set_on_grad_raster_time(time: float, seq: options.Sequence):
    return np.ceil(time / seq.ppSeq.grad_raster_time) * seq.ppSeq.grad_raster_time


def load_external_rf(rf_file) -> np.ndarray:
    pass


class Acquisition:
    def __init__(self, seq: options.Sequence):
        self.seq = seq
        # grad
        self.read_grad: types.SimpleNamespace = types.SimpleNamespace()
        self.read_grad_pre: types.SimpleNamespace = types.SimpleNamespace()
        # adc
        self.adc: types.SimpleNamespace = types.SimpleNamespace()
        # phase
        self.phase_grad_areas: np.ndarray = np.zeros(0)
        self.phase_grad_pre_adc: types.SimpleNamespace = types.SimpleNamespace()
        self.phase_grad_post_adc: types.SimpleNamespace = types.SimpleNamespace()
        # timing
        self.t_phase: float = 0.0  # time needed for longest phase enc -> all phase enc
        # init
        self._make_read_gradients()
        self._set_phase_areas()

    # methods private
    def _make_read_gradients(self):
        self.read_grad = pp.make_trapezoid(
            channel='x',
            flat_area=self.seq.params.deltaK * self.seq.params.resolutionNRead,
            flat_time=self.seq.params.acquisitionTime,  # given in [s] via options
            rise_time=self.seq.params.rise_time_grad,
            system=self.seq.ppSys
        )
        self.read_grad_pre = pp.make_trapezoid(
            channel='x',
            area=self.read_grad.area / 2,
            rise_time=self.seq.params.rise_time_grad,
            system=self.seq.ppSys
        )
        # set adc
        self.adc = pp.make_adc(
            num_samples=self.seq.params.resolutionNRead,
            delay=self.read_grad.rise_time,
            duration=self.seq.params.acquisitionTime,
            system=self.seq.ppSys)

    def _set_phase_areas(self):
        self.phase_grad_areas = (np.arange(self.seq.params.resolutionNPhase) - self.seq.params.resolutionNPhase / 2) * \
                                self.seq.params.deltaK
        # build longest phase gradient
        gPhase_max = pp.make_trapezoid(
            channel='y',
            area=np.max(self.phase_grad_areas),
            rise_time=self.seq.params.rise_time_grad,
            system=self.seq.ppSys
        )
        # calculate time needed for biggest phase grad
        self.t_phase = set_on_grad_raster_time(pp.calc_duration(gPhase_max), self.seq)

    def set_phase_grads(self, idx_phase):
        if np.abs(self.phase_grad_areas[idx_phase]) > 0:
            # calculate phase step
            self.phase_grad_pre_adc = pp.make_trapezoid(
                channel='y',
                area=self.phase_grad_areas[idx_phase],
                duration=self.t_phase,
                system=self.seq.ppSys
            )
            self.phase_grad_post_adc = pp.make_trapezoid(
                channel='y',
                area=-self.phase_grad_areas[idx_phase],
                duration=self.t_phase,
                system=self.seq.ppSys
            )
        else:
            self.phase_grad_pre_adc = pp.make_delay(self.t_phase)
            self.phase_grad_post_adc = pp.make_delay(self.t_phase)

    def reset_read_grad_pre(self, read_grad_pre: types.SimpleNamespace):
        self.read_grad_pre = read_grad_pre

    def get_read_grad_pre(self) -> types.SimpleNamespace:
        return self.read_grad_pre

    def reset_t_phase(self, t_phase: float):
        self.t_phase = t_phase

    def get_t_phase(self) -> float:
        return self.t_phase


class Excitation:
    def __init__(self, seq: options.Sequence, read_grad_pre: types.SimpleNamespace):
        self.seq = seq
        # slice
        self.slice_grad: types.SimpleNamespace = types.SimpleNamespace()
        self.slice_grad_re: types.SimpleNamespace = types.SimpleNamespace()
        # read
        self.read_grad_pre: types.SimpleNamespace = read_grad_pre
        # rf
        self.rf: types.SimpleNamespace = types.SimpleNamespace()
        # delay
        self.delay: types.SimpleNamespace = pp.make_delay(0.0)

        # timing post excitation for prephasing read and rephasing slice
        self.t_read_pre = set_on_grad_raster_time(pp.calc_duration(self.read_grad_pre), seq=self.seq)
        self.t_pre_rephase: float = 0.0

        # init
        self._make_rf_grad_pulse()  # build pulse gradient
        self._recalculate_rephase_grad()  # recalculate rephasing and spoiling grad
        self._find_time_recalc()  # find longer time between read - prephasing and slice - rephasing
        # and reset grads
        self._merge_grads()  # merge slice gradients to continuous waveform

    def _make_rf_grad_pulse(self):
        duration = set_on_grad_raster_time(self.seq.params.excitationDuration * 1e-6, seq=self.seq)
        self.rf, self.slice_grad, self.slice_grad_re = pp.make_gauss_pulse(
            flip_angle=self.seq.params.excitationRadFA,
            phase_offset=self.seq.params.excitationRadRfPhase,
            delay=0.0,
            time_bw_product=self.seq.params.excitationTimeBwProd,
            duration=duration,  # given in [us] via options
            max_slew=0.8 * self.seq.ppSys.max_slew,
            system=self.seq.ppSys,
            slice_thickness=self.seq.params.resolutionSliceThickness * 1e-3,
            return_gz=True,
            use='excitation'
        )
        self.rf.signal = self.rf.signal[:-int(self.rf.ringdown_time * 1e6)]
        self.rf.t = self.rf.t[:-int(self.rf.ringdown_time * 1e6)]

    def _recalculate_rephase_grad(self):
        # calculate spoil grad area -> cast thickness from mm to m
        spoil_area = self.seq.params.spoilerScaling * 1e3 / self.seq.params.resolutionSliceThickness
        # reset rephaser
        self.slice_grad_re = pp.make_trapezoid(
            channel='z',
            area=spoil_area + self.slice_grad_re.area,
            rise_time=self.seq.params.rise_time_grad,
            system=self.seq.ppSys
        )

    def _find_time_recalc(self):
        # set longer time after excitation and optimize gradients to timing
        if self.t_read_pre > pp.calc_duration(self.slice_grad_re):
            t_excitation_post = set_on_grad_raster_time(pp.calc_duration(self.read_grad_pre), seq=self.seq)
            logModule.debug(f"Read prephaser time restricting: {t_excitation_post * 1e3:.2f} ms")
            self.slice_grad_re = pp.make_trapezoid(
                channel='z',
                area=self.slice_grad_re.area,
                duration=t_excitation_post,
                system=self.seq.ppSys
            )
        else:
            t_excitation_post = set_on_grad_raster_time(pp.calc_duration(self.slice_grad_re), seq=self.seq)
            logModule.debug(f"Slice rephaser time restricting: {t_excitation_post * 1e3:.2f} ms")
            self.read_grad_pre = pp.make_trapezoid(
                channel='x',
                area=self.read_grad_pre.area,
                duration=t_excitation_post,
                system=self.seq.ppSys
            )
        self.t_pre_rephase = t_excitation_post

    def _merge_grads(self):
        # --- merge excitation and rephaser at edges ---
        excRise = self.rf.delay
        excAmp = self.slice_grad.amplitude
        # take ramp up and flat area as new excitation gradient
        t_arr = np.array([
            0.0,
            excRise,
            self.slice_grad.flat_time + excRise
        ])
        amps = np.array([
            0.0,
            excAmp,
            excAmp
        ])
        self.slice_grad = pp.make_extended_trapezoid('z', amplitudes=amps, times=t_arr)

        # take ramp down/up to rephaser and rephaser as new re gradient
        t_arr = np.array([
            0.0,
            excRise + self.slice_grad_re.rise_time,
            self.slice_grad_re.flat_time + excRise + self.slice_grad_re.rise_time,
            self.slice_grad_re.flat_time + excRise + 2 * self.slice_grad_re.rise_time
        ])
        amps = np.array([
            self.slice_grad.last,
            self.slice_grad_re.amplitude,
            self.slice_grad_re.amplitude,
            0.0
        ])
        logModule.debug(f"excitation grad: {1e3 * self.slice_grad.last / self.seq.specs.gamma:.2f} mT/m")
        logModule.debug(
            f"excitation rephasing grad: {1e3 * self.slice_grad_re.amplitude / self.seq.specs.gamma:.2f} mT/m"
        )
        self.slice_grad_re = pp.make_extended_trapezoid('z', amplitudes=amps, times=t_arr)
        self.slice_grad.amplitude = excAmp


class Refocusing:
    def __init__(self, seq: options.Sequence, t_phase_max: float):
        self.seq = seq
        self.t_phase_max = t_phase_max
        # slice
        self.slice_grad: types.SimpleNamespace = types.SimpleNamespace()
        self.slice_grad_re: types.SimpleNamespace = types.SimpleNamespace()
        self.slice_grad_ru: types.SimpleNamespace = types.SimpleNamespace()

        self.slice_grad_spoil_pre: types.SimpleNamespace = types.SimpleNamespace()
        self.slice_grad_spoil_post: types.SimpleNamespace = types.SimpleNamespace()
        self.slice_grad_spoil: types.SimpleNamespace = types.SimpleNamespace()

        # rf
        self.rf: types.SimpleNamespace = types.SimpleNamespace()

        # timing
        self.delay = types.SimpleNamespace = pp.make_delay(0.0)
        self.t_spoil: float = 0.0

        # init
        self._make_rf_grad_pulse()  # build rf gradient pulse
        self._make_spoiler_gradient()  # build spoiler gradients
        self._find_time_recalc()
        self._merge_grads()  # merge slice gradients to continuous waveform

    def _make_rf_grad_pulse(self):
        duration = set_on_grad_raster_time(self.seq.params.refocusingDuration * 1e-6, seq=self.seq)
        self.rf, self.slice_grad, self.slice_grad_re = pp.make_gauss_pulse(
            flip_angle=self.seq.params.refocusingRadFA,
            phase_offset=self.seq.params.refocusingRadRfPhase,
            duration=duration,  # given in [us] via options
            max_slew=0.8 * self.seq.ppSys.max_slew,
            apodization=0.5,
            delay=0.0,
            time_bw_product=self.seq.params.refocusingTimeBwProd,
            system=self.seq.ppSys,
            slice_thickness=self.seq.params.resolutionSliceThickness * 1e-3,
            return_gz=True,
            use='refocusing'
        )
        self.rf.signal = self.rf.signal[:-int(self.rf.ringdown_time * 1e6)]
        self.rf.t = self.rf.t[:-int(self.rf.ringdown_time * 1e6)]

    def _make_spoiler_gradient(self):
        self.slice_grad_spoil = pp.make_trapezoid(
            channel='z',
            area=self.seq.params.spoilerScaling * 1e3 / self.seq.params.resolutionSliceThickness,
            # slice thickness given in mm
            rise_time=self.seq.params.rise_time_grad,
            system=self.seq.ppSys
        )
        self.t_spoil = set_on_grad_raster_time(pp.calc_duration(self.slice_grad_spoil), seq=self.seq)

    def _merge_grads(self):
        # merge refocussing and spoil at edges
        refRise = self.rf.delay
        self.rf.delay = 0.0
        refAmp = self.slice_grad.amplitude
        # flat part
        self.slice_grad = pp.make_extended_trapezoid('z', amplitudes=[refAmp, refAmp],
                                                     times=[0, self.slice_grad.flat_time])
        # first ramp
        self.slice_grad_ru = pp.make_extended_trapezoid('z', amplitudes=[0, refAmp], times=[0, refRise])
        # spoilers
        spoil_amps = [
            0.0,
            self.slice_grad_spoil.amplitude,
            self.slice_grad_spoil.amplitude,
            refAmp
        ]
        spoil_timings = [
            0.0,
            self.slice_grad_spoil.rise_time,
            self.slice_grad_spoil.rise_time + self.slice_grad_spoil.flat_time,
            2 * self.slice_grad_spoil.rise_time + self.slice_grad_spoil.flat_time
        ]
        self.slice_grad_spoil_pre = pp.make_extended_trapezoid(
            'z',
            amplitudes=spoil_amps,
            times=spoil_timings)
        self.slice_grad_spoil_post = pp.make_extended_trapezoid(
            'z',
            amplitudes=spoil_amps[::-1],
            times=spoil_timings
        )
        self.slice_grad.amplitude = refAmp

    def _find_time_recalc(self):
        self.t_phase_max = set_on_grad_raster_time(self.t_phase_max, seq=self.seq)
        # look for bigger timing upon spoiling (before and after refocusing -> we spoil and phase enc/dec)
        if self.t_spoil > self.t_phase_max:
            self.t_phase_max = self.t_spoil
            logModule.debug(f"Spoiler time restricting: {self.t_spoil * 1e3:.2f} ms")
        else:
            self.t_spoil = self.t_phase_max
            logModule.debug(f"Phase enc time restricting: {self.t_spoil * 1e3:.2f} ms")
            # redo spoiling gradients with longer timing
            self.slice_grad_spoil = pp.make_trapezoid(
                channel='z',
                duration=self.t_phase_max,
                area=self.slice_grad_spoil.area,
                system=self.seq.ppSys
            )


class SequenceBlockEvents:
    def __init__(self, seq: options.Sequence):
        self.seq = seq
        # ___ define all block event vars ___
        # Acquisition
        logModule.info("Setting up Acquisition")
        # Excitation
        self.acquisition = Acquisition(seq=self.seq)
        logModule.info("Setting up Excitation")
        self.excitation = Excitation(seq=self.seq, read_grad_pre=self.acquisition.get_read_grad_pre())
        self.acquisition.reset_read_grad_pre(self.excitation.read_grad_pre)
        # Refocusing
        logModule.info("Setting up Refocusing")
        self.refocusing = Refocusing(seq=self.seq, t_phase_max=self.acquisition.get_t_phase())
        # Timing
        self.t_duration_echo_train: float = 0.0
        self.t_delay_slice: types.SimpleNamespace = pp.make_delay(0.0)
        self._calculate_min_esp()
        # make sure timing is set
        self.acquisition.reset_t_phase(self.refocusing.t_spoil)

        # k space
        self.k_start: int = -1
        self.k_end: int = -1
        self.k_indexes: np.ndarray = np.zeros(0)
        # slice loop
        numSlices = self.seq.params.resolutionNumSlices
        self.z = np.zeros((2, int(np.ceil(numSlices / 2))))

    def _calculate_min_esp(self):
        # find minimal echo spacing

        # between excitation and refocus = esp / 2 -> rf with delay?
        timing_excitation_refocus = pp.calc_duration(self.excitation.rf) / 2 + \
                                    self.excitation.t_pre_rephase + \
                                    pp.calc_duration(self.refocusing.rf) / 2
        timing_excitation_refocus = set_on_grad_raster_time(timing_excitation_refocus, seq=self.seq)

        # between refocus and adc = esp / 2
        timing_refoucs_adc = pp.calc_duration(self.refocusing.rf) / 2 + \
                             self.refocusing.t_spoil + \
                             pp.calc_duration(self.acquisition.read_grad) / 2
        timing_refoucs_adc = set_on_grad_raster_time(timing_refoucs_adc, seq=self.seq)

        # diff
        t_diff = set_on_grad_raster_time(np.abs(timing_refoucs_adc - timing_excitation_refocus), seq=self.seq)
        # choose longer time as half echo spacing
        if timing_refoucs_adc > timing_excitation_refocus:
            esp = 2 * timing_refoucs_adc
            self.excitation.delay = pp.make_delay(t_diff)
        else:
            esp = 2 * timing_excitation_refocus
            self.refocusing.delay = pp.make_delay(t_diff)
        self.seq.params.ESP = 1e3 * esp

        logModule.info(f"Found minimum TE: {esp * 1e3:.2f} ms")

        self.t_duration_echo_train = set_on_grad_raster_time(
            self.excitation.rf.delay +
            (pp.calc_duration(self.excitation.rf) - self.excitation.rf.delay) / 2 +  # before middle of rf
            self.seq.params.ETL * esp +  # whole TE train
            pp.calc_duration(self.acquisition.read_grad) / 2 +  # half of last read gradient
            self.refocusing.t_spoil,  # spoiler
            seq=self.seq
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
            seq=self.seq
        )
        self.t_delay_slice = pp.make_delay(delay_slice_time)
        logModule.info(f"Delay between slices: {self.t_delay_slice.delay * 1e3:.2f} ms")

    def _set_k_space(self):
        # calculate center of k space and indexes for full sampling band
        k_central_phase = int(self.seq.params.resolutionNPhase / 2)
        k_half_central_lines = int(self.seq.params.numberOfCentralLines / 2)
        # set indexes for start and end of full k space center sampling
        self.k_start = k_central_phase - k_half_central_lines
        self.k_end = k_central_phase + k_half_central_lines

        # The rest of the lines we will use tse style phase step blip between the echoes of one echo train
        # -> acceleration increases with number of contrasts
        k_end_low = self.k_start - self.seq.params.ETL
        k_end_high = self.seq.params.resolutionNPhase - self.seq.params.ETL
        # calculate indexes
        self.k_indexes = np.concatenate((np.arange(0, k_end_low, self.seq.params.accelerationFactor),
                                         np.arange(self.k_end, k_end_high, self.seq.params.accelerationFactor)))

    def _set_delta_slices(self):
        # multi-slice
        # want to go through the slices alternating from beginning and middle
        delta_z = self.seq.params.resolutionSliceThickness * self.seq.params.resolutionNumSlices * \
                  (1 + self.seq.params.resolutionSliceGap / 100.0) * 1e-3  # cast from % / cast from mm
        numSlices = self.seq.params.resolutionNumSlices
        self.z.flat[:numSlices] = np.linspace((-delta_z / 2), (delta_z / 2), numSlices)
        # reshuffle slices mid+1, 1, mid+2, 2, ...
        self.z = self.z.transpose().flatten()[:numSlices]

    def _add_blocks_excitation_first_read(self, phase_idx: int):
        # set phase grads
        self.acquisition.set_phase_grads(idx_phase=phase_idx)

        # excitation
        self.seq.ppSeq.add_block(self.excitation.rf, self.excitation.slice_grad)
        # rephasing
        self.seq.ppSeq.add_block(self.excitation.slice_grad_re, self.excitation.read_grad_pre)
        # delay if necessary
        self.seq.ppSeq.add_block(self.excitation.delay)

        # refocus
        self.seq.ppSeq.add_block(self.refocusing.slice_grad_ru)
        self.seq.ppSeq.add_block(self.refocusing.rf, self.refocusing.slice_grad)
        # spoiling phase encode, delay if necessary
        self.seq.ppSeq.add_block(self.refocusing.slice_grad_spoil_post, self.acquisition.phase_grad_pre_adc)
        # delay if necessary
        self.seq.ppSeq.add_block(self.refocusing.delay)

        # read
        self.seq.ppSeq.add_block(self.acquisition.read_grad, self.acquisition.adc)

    def _add_blocks_refocusing_adc(self, phase_idx: int, tse_style: bool = False):
        for contrast_idx in np.arange(1, self.seq.params.ETL):
            # delay if necessary
            self.seq.ppSeq.add_block(self.refocusing.delay)

            # dephase, spoil
            self.seq.ppSeq.add_block(self.acquisition.phase_grad_post_adc, self.refocusing.slice_grad_spoil_pre)

            # refocus
            self.seq.ppSeq.add_block(self.refocusing.rf, self.refocusing.slice_grad)

            # spoil phase encode
            # jump to next line if tse style acquisition
            if tse_style:
                idx_phase = phase_idx + contrast_idx
            else:
                idx_phase = phase_idx
            # set phase
            self.acquisition.set_phase_grads(idx_phase=idx_phase)
            self.seq.ppSeq.add_block(self.acquisition.phase_grad_pre_adc, self.refocusing.slice_grad_spoil_post)
            # read
            self.seq.ppSeq.add_block(self.acquisition.read_grad, self.acquisition.adc)

        # spoil end
        self.seq.ppSeq.add_block(
            self.acquisition.phase_grad_post_adc,
            self.refocusing.slice_grad_spoil_pre,
            self.excitation.read_grad_pre
        )
        self.seq.ppSeq.add_block(self.t_delay_slice)

    def _loop_central_mc(self):
        logModule.info(f"Central lines")
        # through phase encodes
        line_bar = tqdm.trange(self.seq.params.numberOfCentralLines, desc="phase encodes")
        for idx_n in line_bar:  # We have N phase encodes for all ETL contrasts
            for idx_slice in range(self.seq.params.resolutionNumSlices):
                # apply slice offset
                freq_offset = self.excitation.slice_grad.amplitude * self.z[idx_slice]
                self.excitation.rf.freq_offset = freq_offset

                freq_offset = self.refocusing.slice_grad.amplitude * self.z[idx_slice]
                self.refocusing.rf.freq_offset = freq_offset

                # we start at lower end and move through central lines
                idx_phase = self.k_start + idx_n
                # excitation to first read
                self._add_blocks_excitation_first_read(phase_idx=idx_phase)

                # refocusing blocks
                self._add_blocks_refocusing_adc(phase_idx=idx_phase, tse_style=False)

    def _loop_acc_tse(self):
        logModule.info(f"TSE acc lines")
        # The rest of the lines we will use tse style phase step blip between the echoes of one echo train
        # -> acceleration increases with number of contrasts

        line_bar = tqdm.trange(len(self.k_indexes), desc="phase encodes")
        for idx_n in line_bar:  # We have N phase encodes for all ETL contrasts
            for idx_slice in range(self.seq.params.resolutionNumSlices):
                # apply slice offset
                freq_offset = self.excitation.slice_grad.amplitude * self.z[idx_slice]
                self.excitation.rf.freq_offset = freq_offset

                freq_offset = self.refocusing.slice_grad.amplitude * self.z[idx_slice]
                self.refocusing.rf.freq_offset = freq_offset

                # for idx_slice in range(num_slices):
                idx_phase = self.k_indexes[idx_n]
                # add blocks excitation til first read
                self._add_blocks_excitation_first_read(phase_idx=idx_phase)

                # add blocks for refocussing pulses, tse style
                self._add_blocks_refocusing_adc(phase_idx=idx_phase, tse_style=True)

    def build(self):
        # calculate number of slices
        self._calculate_num_slices()
        # set k-space sampling indices
        self._set_k_space()
        # set positions for slices
        self._set_delta_slices()

        # loop through central multi contrast building blocks
        self._loop_central_mc()
        # loop through tse style outer k-space
        self._loop_acc_tse()

    def get_seq(self):
        return self.seq
