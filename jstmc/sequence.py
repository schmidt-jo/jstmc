import matplotlib.pyplot as plt

from jstmc import events, options
import numpy as np
import pypulseq as pp
import logging
import tqdm

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

    def list_events_to_ns(self):
        return [ev.to_simple_ns() for ev in self.list_events()]

    def list_events(self):
        event_list = [self.rf, self.grad_read, self.grad_slice, self.grad_phase, self.adc, self.delay]
        return [ev for ev in event_list if ev.get_duration() > 1e-5]

    def get_duration(self):
        return np.max([t.get_duration() for t in self.list_events()])

    @classmethod
    def build_excitation(cls, params: options.SequenceParameters, system: pp.Opts, adjust_ramp_area: float = None):
        # Excitation
        logModule.info("setup excitation")
        if params.extRfExc:
            logModule.info(f"rf -- loading rfpf from file: {params.extRfExc}")
            rf = events.RF.load_from_rfpf(
                fname=params.extRfExc, flip_angle_rad=params.excitationRadFA, phase_rad=params.excitationRadRfPhase,
                system=system, duration_s=params.excitationDuration * 1e-6, pulse_type='excitation'
            )
        else:
            logModule.info(f"rf -- build sync pulse")
            time_bw_prod = params.excitationTimeBwProd
            rf = events.RF.make_sinc_pulse(
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
            re_spoil_moment=-params.sliceSpoilingMoment,
            rephase=1.0,
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
            logModule.info(f"rf -- build sync pulse")
            rf = events.RF.make_sinc_pulse(
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
            slice_thickness_m=1 / params.refocusingScaleSliceGrad * params.resolutionSliceThickness * 1e-3,
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
    def build_acquisition(cls, params: options.SequenceParameters, system: pp.Opts):
        # block : adc + read grad
        logModule.info("setup acquisition")
        acquisition_window = set_on_grad_raster_time(system=system, time=params.acquisitionTime)
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
            area=params.sliceSpoilingMoment
        )
        duration = grad_phase.set_on_raster(
            np.max([grad_slice.get_duration(), grad_phase.get_duration(), grad_read_spoil.get_duration()])
        )
        # set longest for all
        grad_read_spoil = events.GRAD.make_trapezoid(
            channel=params.read_dir,
            area=-1 / 2 * grad_read.area,  # keep 1/5 as spoiling moment
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
            area=-params.sliceSpoilingMoment,
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
        x_arr = np.arange(int(self.get_duration() * 1e6))
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


class JsTmcSequence:
    def __init__(self, seq_opts: options.Sequence):
        logModule.info(f"init jstmc algorithm")
        self.seq = seq_opts
        self.params = seq_opts.params
        self.system = seq_opts.ppSys

        # phase grads
        self.phase_areas: np.ndarray = (- np.arange(self.params.resolutionNPhase) +
                                        self.params.resolutionNPhase / 2) * self.params.deltaK_phase
        # slice loop
        numSlices = self.params.resolutionNumSlices
        self.z = np.zeros((2, int(np.ceil(numSlices / 2))))
        self.trueSliceNum = np.zeros(numSlices)
        # k space
        self.k_indexes: np.ndarray = np.zeros(
            (self.params.ETL, self.params.numberOfCentralLines + self.params.numberOfOuterLines),
            dtype=int
        )
        self.sampling_pattern: list = []

        # timing
        self.esp: float = 0.0
        self.delay_exci_ref1: events.DELAY = events.DELAY()
        self.delay_ref_adc: events.DELAY = events.DELAY()
        self.phase_enc_time: float = 0.0
        self.delay_slice: events.DELAY = events.DELAY()

        # sbbs
        self.block_refocus_1: EventBlock = EventBlock.build_refocus(params=self.params, system=self.system, pulse_num=0)
        ramp_area_ref_1 = self.block_refocus_1.grad_slice.t_array_s[1] * self.block_refocus_1.grad_slice.amplitude[
            1] / 2.0
        self.block_excitation: EventBlock = EventBlock.build_excitation(params=self.params, system=self.system,
                                                                        adjust_ramp_area=ramp_area_ref_1)
        self.block_acquisition: EventBlock = EventBlock.build_acquisition(params=self.params, system=self.system)
        self.block_refocus, self.phase_enc_time = EventBlock.build_refocus(
            params=self.params, system=self.system, pulse_num=1, return_pe_time=True
        )
        self.block_spoil_end: EventBlock = EventBlock.build_spoiler_end(params=self.params, system=self.system)
        if self.seq.config.visualize:
            self.block_excitation.plot()
            self.block_refocus_1.plot()
            self.block_refocus.plot()

    def calculate_min_esp(self):
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
            self.delay_exci_ref1 = events.DELAY.make_delay(t_half_esp - t_exci_ref, system=self.system)
            if not self.delay_exci_ref1.check_on_block_raster():
                err = f"exci ref delay not on block raster"
                logModule.error(err)
        if t_ref_1_adc < t_half_esp:
            self.delay_ref_adc = events.DELAY.make_delay(t_half_esp - t_ref_1_adc, system=self.system)
            if not self.delay_ref_adc.check_on_block_raster():
                err = f"adc ref delay not on block raster"
                logModule.error(err)

    def _calculate_slice_delay(self):
        # time per echo train
        t_pre_etl = self.block_excitation.rf.t_delay_s + self.block_excitation.rf.t_duration_s / 2
        t_etl = self.params.ETL * self.params.ESP * 1e-3  # esp in ms
        t_post_etl = self.block_acquisition.get_duration() / 2 + self.phase_enc_time

        t_total_etl = t_pre_etl + t_etl + t_post_etl

        max_num_slices = int(np.floor(self.params.TR * 1e-3 / t_total_etl))
        logModule.info(f"\t\t-total echo train length: {t_total_etl * 1e3:.2f} ms")
        logModule.info(f"\t\t-desired number of slices: {self.params.resolutionNumSlices}")
        logModule.info(f"\t\t-possible number of slices within TR: {max_num_slices}")
        if self.params.resolutionNumSlices > max_num_slices:
            logModule.info(f"increase TR or Concatenation needed")

        self.delay_slice = events.DELAY.make_delay(
            self.params.TR * 1e-3 / self.params.resolutionNumSlices - t_total_etl, system=self.system
        )
        logModule.info(f"\t\t-time between slices: {self.delay_slice.get_duration() * 1e3:.2f} ms")
        if not self.delay_slice.check_on_block_raster():
            self.delay_slice.set_on_block_raster()
            logModule.info(f"\t\t-adjusting TR delay to raster time: {self.delay_slice.get_duration() * 1e3:.2f} ms")

    def _calculate_scan_time(self):
        t_total = self.params.TR * 1e-3 * (self.params.numberOfCentralLines + self.params.numberOfOuterLines)
        logModule.info(f"\t\t-total scan time: {t_total / 60:.1f} min ({t_total:.1f} s)")

    def build(self):
        logModule.info(f"__Build Sequence__")
        logModule.info(f"build -- calculate minimum ESP")
        self.calculate_min_esp()
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

    def _write_sampling_pattern(self, echo_idx: int, phase_idx: int, slice_idx: int):
        pe_num = self.k_indexes[echo_idx, phase_idx]
        sampling_index = {"pe_num": pe_num, "slice_num": int(self.trueSliceNum[slice_idx]), "echo_num": echo_idx}
        self.sampling_pattern.append(sampling_index)

    def _loop_lines(self):
        # through phase encodes
        line_bar = tqdm.trange(
            self.params.numberOfCentralLines + self.params.numberOfOuterLines, desc="phase encodes"
        )
        for idx_n in line_bar:  # We have N phase encodes for all ETL contrasts
            for idx_slice in range(self.params.resolutionNumSlices):
                # looping through slices per phase encode
                # apply slice offset
                self._apply_slice_offset(idx_slice=idx_slice)

                # excitation
                # add block
                self.seq.ppSeq.add_block(*self.block_excitation.list_events_to_ns())

                # delay if necessary
                if self.delay_exci_ref1.get_duration() > 1e-7:
                    self.seq.ppSeq.add_block(self.delay_exci_ref1.to_simple_ns())

                # first refocus
                self._set_fa(echo_idx=0)
                self._set_phase_grad(phase_idx=idx_n, echo_idx=0)
                # add block
                self.seq.ppSeq.add_block(*self.block_refocus_1.list_events_to_ns())

                # delay if necessary
                if self.delay_ref_adc.get_duration() > 1e-7:
                    self.seq.ppSeq.add_block(self.delay_ref_adc.to_simple_ns())

                # adc
                self.seq.ppSeq.add_block(*self.block_acquisition.list_events_to_ns())
                # write sampling pattern
                self._write_sampling_pattern(phase_idx=idx_n, echo_idx=0, slice_idx=idx_slice)
                # delay if necessary
                if self.delay_ref_adc.get_duration() > 1e-7:
                    self.seq.ppSeq.add_block(self.delay_ref_adc.to_simple_ns())

                # loop
                for echo_idx in np.arange(1, self.params.ETL):
                    # set fa
                    self._set_fa(echo_idx=echo_idx)
                    # set phase
                    self._set_phase_grad(echo_idx=echo_idx, phase_idx=idx_n)
                    # set slice offset
                    self._apply_slice_offset(idx_slice=idx_slice)
                    # add block
                    self.seq.ppSeq.add_block(*self.block_refocus.list_events_to_ns())
                    # delay if necessary
                    if self.delay_ref_adc.get_duration() > 1e-7:
                        self.seq.ppSeq.add_block(self.delay_ref_adc.to_simple_ns())

                    # adc
                    self.seq.ppSeq.add_block(*self.block_acquisition.list_events_to_ns())
                    # write sampling pattern
                    self._write_sampling_pattern(echo_idx=echo_idx, phase_idx=idx_n, slice_idx=idx_slice)

                    # delay if necessary
                    if self.delay_ref_adc.get_duration() > 1e-7:
                        self.seq.ppSeq.add_block(self.delay_ref_adc.to_simple_ns())
                # spoil end
                self.seq.ppSeq.add_block(*self.block_spoil_end.list_events_to_ns())
                # insert TR
                self.seq.ppSeq.add_block(self.delay_slice.to_simple_ns())
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
        if np.abs(self.phase_areas[idx_phase]) > 0:
            sbb.grad_phase.amplitude[-3:-1] = - self.phase_areas[idx_phase] / self.phase_enc_time
        else:
            sbb.grad_phase.amplitude = np.zeros_like(sbb.grad_phase.amplitude)
        self.block_spoil_end.grad_phase.amplitude[1:3] = - sbb.grad_phase.amplitude[-3:-1]

    def _apply_slice_offset(self, idx_slice: int):
        for sbb in [self.block_excitation, self.block_refocus_1, self.block_refocus]:
            grad_slice_amplitude_hz = sbb.grad_slice.amplitude[sbb.grad_slice.t_array_s >= sbb.rf.t_delay_s][0]
            sbb.rf.freq_offset_hz = grad_slice_amplitude_hz * self.z[idx_slice]
            # we are setting the phase of a pulse here into its phase offset var.
            # To merge both: given phase parameter and any complex signal array data
            sbb.rf.phase_offset_rad = sbb.rf.phase_rad - 2 * np.pi * sbb.rf.freq_offset_hz * sbb.rf.calculate_center()

    def _set_delta_slices(self):
        # multi-slice
        numSlices = self.params.resolutionNumSlices
        # cast from mm
        delta_z = self.params.z_extend * 1e-3
        if self.params.interleavedAcquisition:
            logModule.info("\t\t-set interleaved acquisition")
            # want to go through the slices alternating from beginning and middle
            self.z.flat[:numSlices] = np.linspace((-delta_z / 2), (delta_z / 2), numSlices)
            # reshuffle slices mid+1, 1, mid+2, 2, ...
            self.z = self.z.transpose().flatten()[:numSlices]
        else:
            logModule.info("\t\t-set sequential acquisition")
            self.z = np.linspace((-delta_z / 2), (delta_z / 2), numSlices)
        # find reshuffled slice numbers
        for idx_slice_num in range(numSlices):
            z_val = self.z[idx_slice_num]
            z_pos = np.where(np.unique(self.z) == z_val)[0][0]
            self.trueSliceNum[idx_slice_num] = z_pos

    def _set_k_space(self):
        if self.params.accelerationFactor > 1.1:
            # calculate center of k space and indexes for full sampling band
            k_central_phase = round(self.params.resolutionNPhase / 2)
            k_half_central_lines = round(self.params.numberOfCentralLines / 2)
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
            for idx_echo in range(self.params.ETL):
                # same encode for all echoes -> central lines
                self.k_indexes[idx_echo, :self.params.numberOfCentralLines] = np.arange(k_start, k_end)
                # random encodes for different echoes
                k_indices = np.random.choice(
                    k_remaining,
                    size=self.params.numberOfOuterLines,
                    replace=False)
                k_indices[::2] = self.params.resolutionNPhase - 1 - k_indices[::2]
                self.k_indexes[idx_echo, self.params.numberOfCentralLines:] = np.sort(k_indices)
        else:
            self.k_indexes[:, :] = np.arange(self.params.numberOfCentralLines + self.params.numberOfOuterLines)

    def get_pypulseq_seq(self):
        return self.seq.ppSeq

    def get_seq(self):
        return self.seq

    def _set_grad_for_emc(self, grad):
        return 1e3 / self.seq.specs.gamma * grad

    def get_emc_info(self) -> dict:
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
            "gradientExcitationRephase": self._set_grad_for_emc(self.block_excitation.grad_slice.amplitude[-2]),
            "durationExcitationRephase": np.sum(np.diff(self.block_excitation.grad_slice.t_array_s[-4:])) * 1e6,
            "gradientExcitationVerse1": 0.0,
            "gradientExcitationVerse2": 0.0,
            "durationExcitationVerse1": 0.0,
            "durationExcitationVerse2": 0.0,
            "refocusAngle": self.params.refocusingFA,
            "refocusPhase": self.params.refocusingRfPhase,
            "gradientRefocus": self._set_grad_for_emc(self.block_refocus.grad_slice.slice_select_amplitude),
            "durationRefocus": self.params.refocusingDuration,
            "gradientCrush": self._set_grad_for_emc(self.block_refocus.grad_slice.amplitude[1]),
            "durationCrush": self.phase_enc_time * 1e6,
            "gradientRefocusVerse1": 0.0,
            "gradientRefocusVerse2": 0.0,
            "durationRefocusVerse1": 0.0,
            "durationRefocusVerse2": 0.0
        }
        return emc_dict

    def get_sampling_pattern(self) -> list:
        return self.sampling_pattern

    def get_pulse_amplitudes(self) -> np.ndarray:
        exc_pulse = self.block_excitation.rf.signal
        return exc_pulse

    def get_z(self):
        # get slice extend
        return self.z


if __name__ == '__main__':
    seq = JsTmcSequence(options.Sequence())
    seq.build()
    emc_dict = seq.get_emc_info()
    pp_seq = seq.get_pypulseq_seq()
    pp_seq.plot()
