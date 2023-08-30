import pandas as pd
from jstmc import options, kernels, plotting
import numpy as np
import logging
import pypulseq as pp
import pypsi
import pathlib as plib
import abc

log_module = logging.getLogger(__name__)


class Sequence(abc.ABC):
    def __init__(self, pypsi_params: pypsi.Params = pypsi.Params()):

        self.interface: pypsi.Params = pypsi_params
        # create shorthand for pypulseq parameters
        self.params: pypsi_params.pypulseq = self.interface.pypulseq

        self.pp_sys: pp.Opts = self._set_pp_sys_from_pypsi()
        self.pp_seq: pp.Sequence = self._set_pp_seq_from_pp_sys()

        # track echoes
        self.te: list = []

        # phase grads
        self.phase_areas: np.ndarray = (- np.arange(
            self.params.resolution_n_phase
        ) + self.params.resolution_n_phase / 2) * self.params.delta_k_phase
        # slice loop
        self.num_slices = self.params.resolution_slice_num
        self.z = np.zeros((2, int(np.ceil(self.num_slices / 2))))

        self.trueSliceNum = np.zeros(self.num_slices, dtype=int)
        # k space
        self.k_indexes: np.ndarray = np.zeros(
            (self.params.etl,
             self.params.number_central_lines + self.params.number_outer_lines
             ),
            dtype=int
        )
        # set exciation and refocusing at least to be present
        self.block_excitation: kernels.Kernel = NotImplemented
        self.block_refocus: kernels.Kernel = NotImplemented

        self._sampling_pattern_constr: list = []
        self.block_excitation: kernels.Kernel = kernels.Kernel()

        # use random state for reproducibiltiy of eg sampling patterns
        # (for that matter any random or semi random sampling used)
        self.prng = np.random.RandomState(0)

        # to check interface set
        self.sampling_pattern_set: bool = False
        self.k_trajectory_set: bool = False
        self.recon_params: bool = False

    # __ public __
    # create
    @classmethod
    def from_cli(cls, args: options.Config):
        pypsi_params = pypsi.Params()
        loads = [args.i, args.s]
        msg = ["sequence configuration", "system specifications"]
        att = ["pypulseq", "specs"]
        for l_idx in range(len(loads)):
            # make plib Path
            l_file = plib.Path(loads[l_idx]).absolute()
            if l_file.is_file():
                log_module.info(f"loading {msg[l_idx]}: {l_file.as_posix()}")
                pypsi_params.__setattr__(att[l_idx], pypsi_params.__getattribute__(att[l_idx]).load(l_file))
            else:
                err = f"A {msg[l_idx]} file needs to be provided and {l_file} was not found to be a valid file."
                log_module.error(err)
                raise FileNotFoundError(err)

        if args.o:
            # set output path
            o_path = plib.Path(args.o).absolute()
            if o_path.suffixes:
                md = o_path.parent
            else:
                md = o_path
            # check if exist
            md.mkdir(parents=True, exist_ok=True)
        else:
            # use input path
            o_path = plib.Path(args.i).absolute()
            if o_path.suffixes:
                o_path = o_path.parent
            log_module.info(f"no output path specified, using same as input: {o_path}")

        pypsi_params.config.output_path = o_path

        # overwrite extra arguments if not default_config
        d_extra = {
            "vv": "version",
            "r": "report",
            "v": "visualize",
            "n": "name",
            "t": "type"
        }
        def_conf = options.Config()
        for key, val in d_extra.items():
            if def_conf.__getattribute__(key) != args.__getattribute__(key):
                pypsi_params.pypulseq.__setattr__(val, args.__getattribute__(key))

        return cls(pypsi_params=pypsi_params)

    # get
    def get_pypulseq_seq(self):
        return self.pp_seq

    def get_z(self):
        # get slice extend
        return self.z

    # writes
    def write_seq(self, name: str = ""):
        file_name = plib.Path(self.interface.config.output_path).absolute()
        if not name:
            name = f"{self.params.name}_{self.params.version}"
        name = f"pyp_seq_{name}"
        save_file = file_name.joinpath(name).with_suffix(".seq")
        log_module.info(f"writing file: {save_file.as_posix()}")
        self._check_interface_set()
        self.set_pyp_definitions()
        self.pp_seq.write(save_file.as_posix())

    def write_pypsi(self, name: str = ""):
        path = plib.Path(self.interface.config.output_path).absolute()
        if not name:
            name = f"{self.params.name}_{self.params.version}"
        name = f"pypsi_{name}"
        save_file = path.joinpath(name).with_suffix(".pkl")
        self._check_interface_set()
        # write
        self.interface.save(save_file)

    def set_pyp_definitions(self):
        self.pp_seq.set_definition(
            "FOV",
            [*self.params.get_fov()]
        )
        self.pp_seq.set_definition(
            "Name",
            f"jstmc{self.params.version}"
        )
        self.pp_seq.set_definition(
            "AdcRasterTime",
            1e-07
        )
        self.pp_seq.set_definition(
            "GradientRasterTime",
            self.interface.specs.grad_raster_time
        )
        self.pp_seq.set_definition(
            "RadiofrequencyRasterTime",
            self.interface.specs.rf_raster_time
        )

    def simulate_grad_moments(self, t_end_ms: int, dt_steps_us: int):
        log_module.info(f"simulating gradient moments")
        # build axis of length TR in steps of us
        ax = np.arange(t_end_ms * 1e3 / dt_steps_us)
        t = 0
        # get gradient shapes
        grads = np.zeros((4, ax.shape[0]))  # grads [read, phase, slice, adc]
        # get seq data until defined length
        block_times = np.cumsum(self.pp_seq.block_durations)
        end_id = np.where(block_times >= t_end_ms * 1e-3)[0][0]
        for block_counter in range(end_id):
            block = self.pp_seq.get_block(block_counter + 1)
            if getattr(block, "adc", None) is not None:  # ADC
                b_adc = block.adc
                # From Pulseq: According to the information from Klaus Scheffler and indirectly from Siemens this
                # is the present convention - the samples are shifted by 0.5 dwell
                t_start = t + int(1e6 * b_adc.delay / dt_steps_us)
                t_end = t_start + int(1e6 * b_adc.num_samples * b_adc.dwell / dt_steps_us)
                grads[3, t_start:t_end] = 1

            grad_channels = ["gx", "gy", "gz"]
            for x in range(len(grad_channels)):  # Gradients
                if getattr(block, grad_channels[x], None) is not None:
                    grad = getattr(block, grad_channels[x])
                    t_start = t + int(1e6 * grad.delay / dt_steps_us)
                    t_end = t_start + int(1e6 * grad.shape_dur / dt_steps_us)
                    grad_shape = np.interp(np.arange(t_end - t_start), 1e6 * grad.tt / dt_steps_us, grad.waveform)
                    grads[x, t_start:t_end] = grad_shape

            t += int(1e6 * self.pp_seq.block_durations[block_counter] / dt_steps_us)

        # want to get the moments, basically just cumsum over the grads, multiplied by delta t = 5us
        grad_moments = np.copy(grads)
        grad_moments[:3] = np.cumsum(grads[:3], axis=1) * dt_steps_us * 1e-6
        # do lazy maximization to 2 for visual purpose, we are only interested in visualizing the drift
        grad_moments[:3] = 2 * grad_moments[:3] / np.max(np.abs(grad_moments[:3]), axis=1, keepdims=True)
        # want to plot the moments
        if self.params.visualize:
            self._plot_grad_moments(grad_moments, dt_in_us=dt_steps_us)

    def build(self):
        log_module.info(f"__Build Sequence__")
        log_module.info(f"build -- calculate total scan time")
        self._calculate_scan_time()
        log_module.info(f"build -- set up k-space")
        self._set_k_space()
        log_module.info(f"build -- set up slices")
        self._set_delta_slices()
        log_module.info(f"build variant specifics")
        self._build_variant()
        log_module.info(f"build -- loop lines")
        self._loop_lines()
        log_module.info(f"set pypsi interface")
        # sampling + k traj
        self._write_sampling_pattern()
        self._set_k_trajectories()  # raises error if not implemented
        # recon
        self._set_recon_parameters_img()
        self._set_nav_parameters()  # raises error if not implemented
        # emc
        self._set_emc_parameters()  # raises error if not implemented
        # pulse
        self._set_pulse_info()

    # __ private __
    @abc.abstractmethod
    def _build_variant(self):
        # to be defined for each sequence variant
        pass

    @abc.abstractmethod
    def _loop_lines(self):
        # to be implemented for each variant, looping through the phase encodes
        pass

    # caution: this is closely tied to the pypsi module and changes in either might affect the other!
    def _check_interface_set(self):
        if any([not state for state in [self.k_trajectory_set, self.recon_params_set, self.sampling_pattern_set]]):
            warn = f"pypsi interface might not have been set:" \
                   f" Sampling Pattern ({self.sampling_pattern_set}), K-Trajectory ({self.k_trajectory_set})," \
                   f" Recon Parameters ({self.recon_params_set})"
            log_module.warning(warn)
        if not (self.interface.sampling_k_traj.sampling_pattern["acq_type"].unique() ==
                self.interface.sampling_k_traj.k_trajectories["acquisition"].unique()).all():
            warn = f"acquisitions registered in sampling pattern: " \
                   f"{self.interface.sampling_k_traj.sampling_pattern['acq_type'].unique()} and registered " \
                   f"k-trajectory - types " \
                   f"{self.interface.sampling_k_traj.k_trajectories['acquisition'].unique()} do not coincide"
            log_module.warning(warn)

    # sampling & k - space
    def _write_sampling_pattern_entry(self, scan_num: int, slice_num: int, pe_num: int, echo_num: int,
                                      acq_type: str = "", echo_type: str = "", echo_type_num: int = -1,
                                      nav_acq: bool = False, nav_dir: int = 0):
        log_module.debug(f"set pypsi sampling pattern")
        self.sampling_pattern_set = True
        # save to list
        self._sampling_pattern_constr.append({
            "scan_num": scan_num, "slice_num": slice_num, "pe_num": pe_num, "acq_type": acq_type,
            "echo_num": echo_num, "echo_type": echo_type, "echo_type_num": echo_type_num,
            "nav_acq": nav_acq, "nav_dir": nav_dir
        })
        return scan_num + 1, echo_type_num + 1

    def _write_sampling_pattern(self):
        self.interface.sampling_k_traj.sampling_pattern_from_list(sp_list=self._sampling_pattern_constr)

    @abc.abstractmethod
    def _set_k_trajectories(self):
        log_module.debug(f"set pypsi k-traj")
        # to be implemented by sequence variants
        pass

    def _register_k_trajectory(self, trajectory: np.ndarray, identifier: str):
        log_module.debug(f"pypsi: register k - trajectory ({identifier}) in interface")
        self.k_trajectory_set = True
        # build shorthand
        self.interface.sampling_k_traj.register_trajectory(
            trajectory=trajectory, identifier=identifier
        )

    # recon info
    def _set_recon_parameters_img(self):
        log_module.debug(f"set pypsi recon")
        self.interface.recon.set_recon_params(
            img_n_read=self.params.resolution_n_read, img_n_phase=self.params.resolution_n_phase,
            img_n_slice=self.params.resolution_slice_num,
            img_resolution_read=self.params.resolution_voxel_size_read,
            img_resolution_phase=self.params.resolution_voxel_size_phase,
            img_resolution_slice=self.params.resolution_slice_thickness,
            etl=self.params.etl,
            os_factor=self.params.oversampling,
            read_dir=self.params.read_dir,
            acc_factor_phase=self.params.acceleration_factor,
            acc_read=False,
            te=self.te
        )
        self.recon_params_set = True

    @abc.abstractmethod
    def _set_nav_parameters(self):
        log_module.debug(f"set pypsi recon nav")
        # to be implemented for each variant
        pass

    # emc
    def _set_emc_parameters(self):
        log_module.debug(f"set pypsi emc")
        self.interface.emc.gamma_hz = self.interface.specs.gamma
        self.interface.emc.gamma_pi = self.interface.specs.gamma / 2 / np.pi
        self.interface.emc.ETL = self.params.etl
        self.interface.emc.ESP = self.params.esp
        self.interface.emc.bw = self.params.bandwidth
        # self.interface.emc.gradMode = "Normal"
        self.interface.emc.excitationAngle = self.params.excitation_rf_rad_fa / np.pi * 180.0
        self.interface.emc.excitationPhase = self.params.excitation_rf_phase
        self.interface.emc.gradientExcitation = self._set_grad_for_emc(
            self.block_excitation.grad_slice.slice_select_amplitude
        )
        self.interface.emc.durationExcitation = self.params.excitation_duration
        self.interface.emc.gradientExcitationVerse1 = 0.0
        self.interface.emc.gradientExcitationVerse2 = 0.0
        self.interface.emc.durationExcitationVerse1 = 0.0
        self.interface.emc.durationExcitationVerse2 = 0.0
        self.interface.emc.refocusAngle = self.params.refocusing_rf_fa
        self.interface.emc.refocusPhase = self.params.refocusing_rf_phase
        self.interface.emc.gradientRefocus = self._set_grad_for_emc(
            self.block_refocus.grad_slice.slice_select_amplitude
        )
        self.interface.emc.durationRefocus = self.params.refocusing_duration
        self.interface.emc.gradientCrush = self._set_grad_for_emc(self.block_refocus.grad_slice.amplitude[1])
        self.interface.emc.durationCrush = np.sum(np.diff(self.block_refocus.grad_slice.t_array_s[-4:])) * 1e6
        self.interface.emc.gradientRefocusVerse1 = 0.0
        self.interface.emc.gradientRefocusVerse2 = 0.0
        self.interface.emc.durationRefocusVerse1 = 0.0
        self.interface.emc.durationRefocusVerse2 = 0.0

        self._fill_emc_info()

    @abc.abstractmethod
    def _fill_emc_info(self):
        log_module.debug(f"fill pypsi emc")
        # to be implemented for each variant
        pass

    # pulse
    def _set_pulse_info(self):
        log_module.debug(f"set pypsi pulse")
        self.interface.pulse.bandwidth_in_Hz = self.block_excitation.rf.bandwidth_hz
        self.interface.pulse.duration_in_us = self.block_excitation.rf.t_duration_s * 1e6
        self.interface.pulse.time_bandwidth = self.block_excitation.rf.t_duration_s * \
                                              self.block_excitation.rf.bandwidth_hz
        self.interface.pulse.num_samples = self.block_excitation.rf.signal.shape[0]

        self.interface.pulse.amplitude = np.abs(self.block_excitation.rf.signal)
        self.interface.pulse.phase = np.angle(self.block_excitation.rf.signal)

    # inits
    def _set_pp_sys_from_pypsi(self) -> pp.Opts:
        log_module.info(f"set pypulseg system limits")
        return pp.Opts(
            B0=self.interface.specs.b_0,
            adc_dead_time=self.interface.specs.adc_dead_time,
            gamma=self.interface.specs.gamma,
            grad_raster_time=self.interface.specs.grad_raster_time,
            grad_unit=self.interface.specs.grad_unit,
            max_grad=self.interface.specs.max_grad,
            max_slew=self.interface.specs.max_slew,
            rf_dead_time=self.interface.specs.rf_dead_time,
            rf_raster_time=self.interface.specs.rf_raster_time,
            rf_ringdown_time=self.interface.specs.rf_ringdown_time,
            rise_time=self.interface.specs.rise_time,
            slew_unit=self.interface.specs.slew_unit
        )

    def _set_pp_seq_from_pp_sys(self) -> pp.Sequence:
        return pp.Sequence(system=self.pp_sys)

    # methods
    def _set_k_space(self):
        if self.params.acceleration_factor > 1.1:
            # calculate center of k space and indexes for full sampling band
            k_central_phase = round(self.params.resolution_n_phase / 2)
            k_half_central_lines = round(self.params.number_central_lines / 2)
            # set indexes for start and end of full k space center sampling
            k_start = k_central_phase - k_half_central_lines
            k_end = k_central_phase + k_half_central_lines

            # The rest of the lines we will use tse style phase step blip between the echoes of one echo train
            # Trying random sampling, ie. pick random line numbers for remaining indices,
            # we dont want to pick the same positive as negative phase encodes to account
            # for conjugate symmetry in k-space.
            # Hence, we pick from the positive indexes twice (thinking of the center as 0)
            # without allowing for duplexes and negate half the picks
            # calculate indexes
            k_remaining = np.arange(0, k_start)
            # build array with dim [num_slices, num_outer_lines] to sample different random scheme per slice
            weighting_factor = np.clip(self.params.sample_weighting, 0.01, 1)
            if weighting_factor > 0.05:
                log_module.info(f"\t\t-weighted random sampling of k-space phase encodes, factor: {weighting_factor}")
            # random encodes for different echoes - random choice weighted towards center
            weighting = np.clip(np.power(np.linspace(0, 1, k_start), weighting_factor), 1e-5, 1)
            weighting /= np.sum(weighting)
            for idx_echo in range(self.params.etl):
                # same encode for all echoes -> central lines
                self.k_indexes[idx_echo, :self.params.number_central_lines] = np.arange(k_start, k_end)

                k_indices = self.prng.choice(
                    k_remaining,
                    size=self.params.number_outer_lines,
                    replace=False,
                    p=weighting

                )
                k_indices[::2] = self.params.resolution_n_phase - 1 - k_indices[::2]
                self.k_indexes[idx_echo, self.params.number_central_lines:] = np.sort(k_indices)
        else:
            self.k_indexes[:, :] = np.arange(
                self.params.number_central_lines + self.params.number_outer_lines
            )

    def _set_grad_for_emc(self, grad):
        return 1e3 / self.interface.specs.gamma * grad

    def _calculate_scan_time(self):
        t_total = self.params.tr * 1e-3 * (
                self.params.number_central_lines + self.params.number_outer_lines
        )
        log_module.info(f"\t\t-total scan time: {t_total / 60:.1f} min ({t_total:.1f} s)")

    def _set_delta_slices(self):
        # multi-slice
        numSlices = self.params.resolution_slice_num
        # cast from mm
        delta_z = self.params.z_extend * 1e-3
        if self.params.interleaved_acquisition:
            log_module.info("\t\t-set interleaved acquisition")
            # want to go through the slices alternating from beginning and middle
            self.z.flat[:numSlices] = np.linspace((-delta_z / 2), (delta_z / 2), numSlices)
            # reshuffle slices mid+1, 1, mid+2, 2, ...
            self.z = self.z.transpose().flatten()[:numSlices]
        else:
            log_module.info("\t\t-set sequential acquisition")
            self.z = np.linspace((-delta_z / 2), (delta_z / 2), numSlices)
        # find reshuffled slice numbers
        for idx_slice_num in range(numSlices):
            z_val = self.z[idx_slice_num]
            z_pos = np.where(np.unique(self.z) == z_val)[0][0]
            self.trueSliceNum[idx_slice_num] = z_pos

    def _set_name_fov(self) -> str:
        fov_r = int(self.params.resolution_fov_read)
        fov_p = int(self.params.resolution_fov_phase / 100 * self.params.resolution_fov_read)
        fov_s = int(self.params.resolution_slice_thickness * self.params.resolution_slice_num)
        return f"fov{fov_r}-{fov_p}-{fov_s}"

    def _set_name_fa(self) -> str:
        return f"fa{int(self.params.refocusing_rf_fa[0])}"

    def _plot_grad_moments(self, grad_moments: np.ndarray, dt_in_us: int):
        ids = ["gx"] * grad_moments.shape[1] + ["gy"] * grad_moments.shape[1] + ["gz"] * grad_moments.shape[1] + \
              ["adc"] * grad_moments.shape[1]
        ax_time = np.tile(np.arange(grad_moments.shape[1]) * dt_in_us, 4)
        df = pd.DataFrame({
            "moments": grad_moments.flatten(), "id": ids,
            "time": ax_time
        })
        plotting.plot_grad_moments(mom_df=df, out_path=self.interface.config.output_path, name="sim_moments")
