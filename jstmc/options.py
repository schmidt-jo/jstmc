import logging

import numpy as np
from simple_parsing import ArgumentParser, helpers
from dataclasses import dataclass
import pypulseq as pp
from pathlib import Path
import json
import pandas as pd

logModule = logging.getLogger(__name__)


@dataclass
class SequenceConfig(helpers.Serializable):
    configFile: str = ""
    outputPath: str = ""
    version: str = "1a"


@dataclass
class ScannerSpecs(helpers.Serializable):
    """
    Holding all Scanning System Parameters
    """
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


@dataclass
class SequenceParameters(helpers.Serializable):
    """
    Holding all Sequence Parameters
    """
    resolutionFovRead: float = 225  # [mm]
    resolutionFovPhase: float = 100.0  # [%]
    resolutionBase: int = 346
    resolutionSliceThickness: float = 0.65  # [mm]
    resolutionNumSlices: int = 55
    resolutionSliceGap: int = 0  # %

    numberOfCentralLines: int = 24
    accelerationFactor: int = 4
    partialFourier: float = 6/8
    useAcc: bool = False

    excitationFA: float = 90.0
    excitationRfPhase: float = 90.0  # °
    excitationDuration: int = 2500  # [us]
    excitationTimeBwProd: float = 2.0

    refocusingFA: float = 180.0
    refocusingRfPhase: float = 0.0  # °
    refocusingDuration: int = 3000  # [us]
    refocusingTimeBwProd: float = 2.0

    spoilerScaling: float = 1.1

    ESP: float = 7.6  # [ms] echo spacing
    ETL: int = 8  # echo train length
    TR: float = 4500.0  # [ms]

    bandwidth: float = 302.0  # [Hz / px]

    def __post_init__(self):
        # resolution
        self.resolutionNRead = self.resolutionBase  # number of freq encodes
        self.resolutionNPhase = int(self.resolutionBase * self.resolutionFovPhase / 100)  # number of phase encodes
        self.resolutionVoxelSizeRead = self.resolutionFovRead / self.resolutionBase  # [mm]
        self.resolutionVoxelSizePhase = self.resolutionFovRead / self.resolutionBase  # [mm]
        self.deltaK = 1e3 / self.resolutionFovRead  # cast to m
        self.TE = np.arange(1, self.ETL + 1) * self.ESP  # [ms] echo times
        # sequence
        self.acquisitionTime = 1 / self.bandwidth
        self.dwell = self.acquisitionTime / self.resolutionNRead
        logModule.info(f"Bandwidth: {self.bandwidth:.1f} Hz/px;"
                       f"Readout time: {self.acquisitionTime * 1e3:.1f} ms;"
                       f"DwellTime: {self.dwell * 1e6:.1f} us;"
                       f"Number of Freq Encodes: {self.resolutionNRead}")
        # casting
        self.excitationRadFA = self.excitationFA / 180.0 * np.pi
        self.excitationRadRfPhase = self.excitationRfPhase / 180.0 * np.pi
        self.refocusingRadFA = self.refocusingFA / 180.0 * np.pi
        self.refocusingRadRfPhase = self.refocusingRfPhase / 180.0 * np.pi
        if not self.useAcc:
            self.accelerationFactor = self.ETL
        self.get_voxel_size()

    def get_voxel_size(self):
        logModule.info(f"Voxel Size [read, phase, slice] in mm: "
                       f"{[self.resolutionVoxelSizeRead, self.resolutionVoxelSizePhase, self.resolutionSliceThickness]}")
        return self.resolutionVoxelSizeRead, self.resolutionVoxelSizePhase, self.resolutionSliceThickness

    def get_fov(self):
        fov_read = 1e-3 * self.resolutionFovRead * 64
        fov_phase = int(fov_read * self.resolutionFovPhase / 100)
        fov_slice = self.resolutionSliceThickness * 1e-3 * self.resolutionNumSlices * (1 + self.resolutionSliceGap/100)
        return fov_read, fov_phase, fov_slice

    def set_esp(self, esp: float):
        if esp < 1.0:
            self.ESP = 1e3 * esp
        else:
            self.ESP = esp
        self.TE = np.arange(1, self.ETL+1) * self.ESP


@dataclass
class Sequence:
    config: SequenceConfig = SequenceConfig()
    specs: ScannerSpecs = ScannerSpecs()
    ppSys: pp.Opts = pp.Opts()
    ppSeq: pp.Sequence = pp.Sequence(ppSys)
    params: SequenceParameters = SequenceParameters()

    @classmethod
    def load(cls, path):
        Seq = Sequence()
        path = Path(path).absolute()
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
    def from_cmd_args(cls, prog_args: ArgumentParser.parse_args):
        Seq = Sequence(specs=prog_args.specs, params=prog_args.params, config=prog_args.config)
        if prog_args.config.configFile:
            Seq = Seq.load(prog_args.config.configFile)

        system = pp.Opts(
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

    def save(self, emc_info: dict = None, sampling_pattern: list = None):
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
            save_file = path.joinpath(f"jstmc{self.config.version}.seq").__str__()
            logModule.info(f" writing file: {save_file}")
            self.ppSeq.write(save_file)

            save_dict = {
                "config": self.config.to_dict(),
                "specs": self.specs.to_dict(),
                "params": self.params.to_dict()
            }
            save_file = path.joinpath(f"jstmc{self.config.version}_config.json")
            logModule.info(f"writing file: {save_file}")
            with open(save_file, "w") as j_file:
                json.dump(save_dict, j_file, indent=2)
            if emc_info is not None:
                save_file = path.joinpath(f"jstmc{self.config.version}_emc_sequence_conf.json")
                logModule.info(f"writing file: {save_file}")
                with open(save_file, "w") as j_file:
                    json.dump(emc_info, j_file, indent=2)
            # write k_space_sampling pattern
            if sampling_pattern is not None:
                self.write_sampling_pattern(sampling_pattern=sampling_pattern)
        else:
            logModule.info("Not Saving: no Path given")

    def write_sampling_pattern(self, sampling_pattern: list):
        path = Path(self.config.outputPath).absolute()
        path.mkdir(parents=True, exist_ok=True)
        sp = pd.DataFrame(sampling_pattern)
        save_file = path.joinpath(f"jstmc{self.config.version}_sampling_pattern.csv")
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
    parser = ArgumentParser(prog='jstmc')
    parser.add_arguments(SequenceConfig, dest="config")
    parser.add_arguments(ScannerSpecs, dest="specs")
    parser.add_arguments(SequenceParameters, dest="params")
    args = parser.parse_args()

    return parser, args
