import logging

import numpy as np
import simple_parsing as sp
import dataclasses as dc
from typing import List
import pypulseq as pp
from pathlib import Path
import json
import pandas as pd

logModule = logging.getLogger(__name__)


@dc.dataclass
class SequenceConfig(sp.helpers.Serializable):
    configFile: str = sp.field(default="", alias=["-c"])
    outputPath: str = sp.field(default="./test/", alias=["-o"])
    version: str = "3b"
    report: bool = sp.field(default=False, alias=["-r"])
    visualize: bool = sp.field(default=True, alias=["-v"])


@dc.dataclass
class ScannerSpecs(sp.helpers.Serializable):
    """
    Holding all Scanning System Parameters
    """
    # magnet
    b_0: float = 6.98    # [T]
    # gradients
    max_grad: int = 40.0
    grad_unit: str = 'mT/m'
    max_slew: int = 200.0
    slew_unit: str = 'T/m/s'
    rise_time: int = 0  # watch out, rise time != 0 gives max_slew = 0 in opts method
    grad_raster_time: float = 10e-6

    # rf
    rf_dead_time: float = 100e-6
    rf_raster_time: float = 1e-6
    rf_ringdown_time: float = 30e-6

    # general
    adc_dead_time: float = 20e-6
    gamma: float = 42577478.518  # [Hz/T]


@dc.dataclass
class SequenceParameters(sp.helpers.Serializable):
    """
    Holding all Sequence Parameters
    """
    resolutionFovRead: float = 100  # [mm]
    resolutionFovPhase: float = 100.0  # [%]
    resolutionBase: int = 100
    resolutionSliceThickness: float = 1.0  # [mm]
    resolutionNumSlices: int = 10
    resolutionSliceGap: int = 20  # %

    numberOfCentralLines: int = 40
    accelerationFactor: float = 2.0

    excitationFA: float = 90.0
    excitationRfPhase: float = 90.0  # °
    excitationDuration: int = 2500  # [us]
    excitationTimeBwProd: float = 2.0
    excitationPreMoment: float = 1000.0    # Hz/m
    excitationRephaseFactor: float = 1.04  # Correction factor for insufficient rephasing

    refocusingFA: List = dc.field(default_factory=lambda: [140.0])
    refocusingRfPhase: List = dc.field(default_factory=lambda: [0.0])  # °
    refocusingDuration: int = 3000  # [us]
    refocusingTimeBwProd: float = 2.0
    refocusingScaleSliceGrad: float = 2/3   # adjust slice selective gradient sice of refocusing -
    # caution: this broadens the slice profile of the pulse, the further away from 180 fa
    # we possibly get saturation outside the slice

    sliceSpoilingMoment: float = 2500.0     # [Hz/m]
    interleavedAcquisition: bool = True
    # interfacing with rfpf
    extRfExc: str = ""
    extRfRef: str = ""

    ESP: float = 7.6  # [ms] echo spacing
    ETL: int = 8  # echo train length
    TR: float = 4500.0  # [ms]

    bandwidth: float = 250.0  # [Hz / px]
    oversampling: float = 2.0   # oversampling factor

    phaseDir: str = "PA"

    def __post_init__(self):
        # resolution
        self.resolutionNRead = self.resolutionBase  # number of freq encodes
        self.resolutionNPhase = int(self.resolutionBase * self.resolutionFovPhase / 100)  # number of phase encodes
        self.resolutionVoxelSizeRead = self.resolutionFovRead / self.resolutionBase  # [mm]
        self.resolutionVoxelSizePhase = self.resolutionFovRead / self.resolutionBase  # [mm]
        self.deltaK_read = 1e3 / self.resolutionFovRead  # cast to m
        self.deltaK_phase = 1e3 / (self.resolutionFovRead * self.resolutionFovPhase / 100.0)  # cast to m
        self.TE = np.arange(1, self.ETL + 1) * self.ESP  # [ms] echo times
        # there is one gap less than number of slices,
        self.z_extend = self.resolutionSliceThickness * (
                self.resolutionNumSlices + self.resolutionSliceGap / 100.0 * (self.resolutionNumSlices - 1))    # in mm
        # acc
        self.numberOfOuterLines = round((self.resolutionNPhase - self.numberOfCentralLines) / self.accelerationFactor)
        # sequence
        self.acquisitionTime = 1 / self.bandwidth
        self.dwell = self.acquisitionTime / self.resolutionNRead / self.oversampling   # oversampling
        logModule.info(f"Bandwidth: {self.bandwidth:.1f} Hz/px;"
                       f"Readout time: {self.acquisitionTime * 1e3:.1f} ms;"
                       f"DwellTime: {self.dwell * 1e6:.1f} us;"
                       f"Number of Freq Encodes: {self.resolutionNRead}")
        # ref list
        if self.refocusingFA.__len__() != self.refocusingRfPhase.__len__():
            err = f"provide same amount of refocusing pulse angle ({self.refocusingFA.__len__()}) " \
                  f"and phases ({self.refocusingRfPhase.__len__()})"
            logModule.error(err)
            raise AttributeError(err)
        # check for phase values
        for l_idx in range(self.refocusingRfPhase.__len__()):
            while np.abs(self.refocusingRfPhase[l_idx]) > 180.0:
                self.refocusingRfPhase[l_idx] = self.refocusingRfPhase[l_idx] -\
                                                np.sign(self.refocusingRfPhase[l_idx]) * 180.0
            while np.abs(self.refocusingFA[l_idx]) > 180.0:
                self.refocusingFA[l_idx] = self.refocusingFA[l_idx] - np.sign(self.refocusingFA[l_idx]) * 180.0
        while self.refocusingFA.__len__() < self.ETL:
            # fill up list with last value
            self.refocusingFA.append(self.refocusingFA[-1])
            self.refocusingRfPhase.append(self.refocusingRfPhase[-1])

        # casting
        self.excitationRadFA = self.excitationFA / 180.0 * np.pi
        self.excitationRadRfPhase = self.excitationRfPhase / 180.0 * np.pi
        self.refocusingRadFA = np.array(self.refocusingFA) / 180.0 * np.pi
        self.refocusingRadRfPhase = np.array(self.refocusingRfPhase) / 180.0 * np.pi
        self.get_voxel_size()
        if self.phaseDir == "PA":
            self.read_dir = 'x'
            self.phase_dir = 'y'
        elif self.phaseDir == "RL":
            self.phase_dir = 'x'
            self.read_dir = 'y'
        else:
            err = 'Unknown Phase direction: chose either PA or RL'
            logModule.error(err)
            raise AttributeError(err)

        # error catches
        if self.sliceSpoilingMoment < 1e-7:
            err = f"this implementation needs a spoiling moment supplied: provide spoiling Moment > 0"
            logModule.error(err)
            raise ValueError(err)

    def get_voxel_size(self):
        logModule.info(
            f"Voxel Size [read, phase, slice] in mm: "
            f"{[self.resolutionVoxelSizeRead, self.resolutionVoxelSizePhase, self.resolutionSliceThickness]}")
        return self.resolutionVoxelSizeRead, self.resolutionVoxelSizePhase, self.resolutionSliceThickness

    def get_fov(self):
        fov_read = 1e-3 * self.resolutionFovRead
        fov_phase = 1e-3 * self.resolutionFovRead * self.resolutionFovPhase / 100
        fov_slice = self.z_extend * 1e-3
        if self.read_dir == 'x':
            logModule.info(
                f"FOV (xyz) Size [read, phase, slice] in mm: "
                f"[{1e3*fov_read:.1f}, {1e3*fov_phase:.1f}, {1e3*fov_slice:.1f}]")
            return fov_read, fov_phase, fov_slice
        else:
            logModule.info(
                f"FOV (xyz) Size [phase, read, slice] in mm: "
                f"[{1e3*fov_phase:.1f}, {1e3*fov_read:.1f}, {1e3*fov_slice:.1f}]")
            return fov_phase, fov_read, fov_slice

    def set_esp(self, esp: float):
        if esp < 1.0:
            self.ESP = 1e3 * esp
        else:
            self.ESP = esp
        self.TE = np.arange(1, self.ETL+1) * self.ESP


@dc.dataclass
class Sequence:
    config: SequenceConfig = SequenceConfig()
    specs: ScannerSpecs = ScannerSpecs()
    ppSys: pp.Opts = pp.Opts()
    ppSeq: pp.Sequence = pp.Sequence(ppSys)
    params: SequenceParameters = SequenceParameters()

    def _set_name_fov(self) -> str:
        fov_r = int(self.params.resolutionFovRead)
        fov_p = int(self.params.resolutionFovPhase / 100 * self.params.resolutionFovRead)
        fov_s = int(self.params.resolutionSliceThickness * self.params.resolutionNumSlices)
        return f"fov{fov_r}-{fov_p}-{fov_s}"

    def _set_name_fa(self) -> str:
        return f"fa{int(self.params.refocusingFA[0])}"

    @classmethod
    def load(cls, path):
        Seq = Sequence()
        path = Path(path).absolute()
        if not path.is_file():
            raise AttributeError(f"{path} not a file")
        if path.suffix == ".json":
            with open(path, "r") as j_file:
                load_dict = json.load(j_file)
            Seq.config = SequenceConfig.from_dict(load_dict["config"])
            Seq.params = SequenceParameters.from_dict(load_dict["params"])
            Seq.specs = ScannerSpecs.from_dict(load_dict["specs"])
        elif path.suffix == ".seq":
            if path.exists() and path.is_file():
                Seq.ppSeq.read(path.__str__())
            Seq.config.outputPath = path.parent
        else:
            raise ValueError(f"{path} file ending not recognized!")
        return Seq

    @classmethod
    def from_cmd_args(cls, prog_args: sp.ArgumentParser.parse_args):
        Seq = Sequence(specs=prog_args.specs, params=prog_args.params, config=prog_args.config)
        if prog_args.config.configFile:
            Seq = Seq.load(prog_args.config.configFile)

        system = pp.Opts(
            B0=prog_args.specs.b_0,
            adc_dead_time=prog_args.specs.adc_dead_time,
            gamma=prog_args.specs.gamma,
            grad_raster_time=prog_args.specs.grad_raster_time,
            grad_unit=prog_args.specs.grad_unit,
            max_grad=prog_args.specs.max_grad,
            max_slew=prog_args.specs.max_slew,
            rf_dead_time=prog_args.specs.rf_dead_time,
            rf_raster_time=prog_args.specs.rf_raster_time,
            rf_ringdown_time=prog_args.specs.rf_ringdown_time,
            rise_time=prog_args.specs.rise_time,
            slew_unit=prog_args.specs.slew_unit
        )
        Seq.ppSys = system
        Seq.ppSeq = pp.Sequence(system=system)
        Seq.setDefinitions()
        return Seq

    def save(self, emc_info: dict = None, sampling_pattern: list = None, pulse_signal: tuple = None):
        if not self.ppSeq.definitions:
            err = "no export definitions were set (FOV, Name)"
            logModule.error(err)
            raise AttributeError(err)

        if self.config.outputPath:
            path = Path(self.config.outputPath).absolute()
            path.mkdir(parents=True, exist_ok=True)
            if path.is_file():
                path = path.parent
            if len(self.config.version) > 2:
                self.config.version = self.config.version[:2]
            file_name = path.joinpath(
                f"jstmc{self.config.version}_{self._set_name_fa()}_{self._set_name_fov()}_{self.params.phaseDir}"
            )
            self.write_seq(file_name)

            self.write_config(file_name)

            if emc_info is not None:
                self.write_emc_info(file_name, emc_info)

            # write k_space_sampling pattern
            if sampling_pattern is not None:
                self.write_sampling_pattern(sampling_pattern=sampling_pattern)

            if pulse_signal is not None:
                self.write_pulse_to_txt(file_name, pulse_signal)
        else:
            logModule.info("Not Saving: no Path given")

    def write_seq(self, file_name):
        save_file = file_name.with_suffix(".seq").__str__()
        logModule.info(f"writing file: {save_file}")
        self.ppSeq.write(save_file)

    def write_config(self, file_name):
        save_dict = {
            "config": self.config.to_dict(),
            "specs": self.specs.to_dict(),
            "params": self.params.to_dict()
        }
        if self.config.configFile:
            save_file = Path(self.config.configFile).absolute()
        else:
            save_file = file_name.with_name(f"{file_name.name}_config").with_suffix(".json")
        save_dict["config"].__setitem__("configFile", save_file.__str__())
        logModule.info(f"writing file: {save_file}")
        with open(save_file, "w") as j_file:
            json.dump(save_dict, j_file, indent=2)

    @staticmethod
    def write_emc_info(file_name, emc_info):
        save_file = file_name.with_name(f"{file_name.name}_emc_sequence_conf").with_suffix(".json")
        logModule.info(f"writing file: {save_file}")
        with open(save_file, "w") as j_file:
            json.dump(emc_info, j_file, indent=2)

    @staticmethod
    def write_pulse_to_txt(file_name, pulse_signal):
        save_file = file_name.with_name(f"{file_name.name}_pulse").with_suffix(".txt")
        logModule.info(f"writing file: {save_file}")
        with open(save_file, "w") as f:
            for idx_sig in range(len(pulse_signal)):
                f.write(f'{pulse_signal[idx_sig]}\t{0.0}\n')

    def write_sampling_pattern(self, sampling_pattern: list):
        path = Path(self.config.outputPath).absolute()
        path.mkdir(parents=True, exist_ok=True)
        sp = pd.DataFrame(sampling_pattern)
        file_name = path.joinpath(
            f"jstmc{self.config.version}_{self._set_name_fa()}_{self._set_name_fov()}_{self.params.phaseDir}_sampling"
            f"-pattern"
        )
        save_file = path.joinpath(file_name).with_suffix(".csv")
        logModule.info(f"writing file: {save_file}")
        sp.to_csv(save_file)

    def check_output_path(self):
        if not self.config.outputPath:
            logModule.info("Not Saving: no Path given; will run through without saving.")

    def setDefinitions(self):
        self.ppSeq.set_definition(
            "FOV",
            [*self.params.get_fov()]
        )
        self.ppSeq.set_definition(
            "Name",
            f"jstmc{self.config.version}"
        )
        self.ppSeq.set_definition(
            "AdcRasterTime",
            1e-07
        )
        self.ppSeq.set_definition(
            "GradientRasterTime",
            self.specs.grad_raster_time
        )
        self.ppSeq.set_definition(
            "RadiofrequencyRasterTime",
            self.specs.rf_raster_time
        )


def createCommandlineParser():
    """
        Build the parser for arguments
        Parse the input arguments.
        """
    parser = sp.ArgumentParser(prog='jstmc')
    parser.add_arguments(SequenceConfig, dest="config")
    parser.add_arguments(ScannerSpecs, dest="specs")
    parser.add_arguments(SequenceParameters, dest="params")
    args = parser.parse_args()

    return parser, args
