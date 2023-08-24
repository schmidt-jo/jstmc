import logging

import simple_parsing as sp
import dataclasses as dc
import typing
import pypulseq as pp
import pypsi
import pathlib as plib
log_module = logging.getLogger(__name__)


@dc.dataclass
class Config:
    # Input sequence configuration
    i: str = sp.field(default="", help="Input Sequence Configuration")
    s: str = sp.field(default="", help="Input System Specifications")
    o: str = sp.field(default="", help="Output Path for .seq file and pypsi interface")
    v: bool = sp.field(default=True, help="Visualize on/off")
    r: bool = sp.field(default=False, help="Report on/off")
    vv: str = sp.field(default="xx", help="Version")

    d: bool = sp.field(default=False, help="Debug on/off")


@dc.dataclass
class Sequence:
    pp_sys: pp.Opts = NotImplemented
    pp_seq: pp.Sequence = NotImplemented
    params: pypsi.Params = pypsi.Params()

    @classmethod
    def from_cli(cls, args: Config):
        instance = cls()
        loads = [args.i, args.s]
        msg = ["sequence configuration", "system specifications"]
        att = ["pypulseq", "specs"]
        for l_idx in range(len(loads)):
            # make plib Path
            l_file = plib.Path(loads[l_idx]).absolute()
            if l_file.is_file():
                log_module.info(f"loading {msg[l_idx]}: {l_file.as_posix()}")
                instance.params.__setattr__(att[l_idx], instance.params.__getattribute__(att[l_idx]).load(l_file))
            else:
                err = f"A {msg[l_idx]} file needs to be provided and {l_file} was not found to be a valid file."
                log_module.error(err)
                raise FileNotFoundError(err)
        # set output path
        o_path = plib.Path(args.o).absolute()
        if o_path.suffixes:
            md = o_path.parent
        else:
            md = o_path
        # check if exist
        md.mkdir(parents=True, exist_ok=True)
        instance.params.config.output_path = o_path

        # overwrite extra arguments if not default_config
        d_extra = {
            "vv": "version",
            "r": "report",
            "v": "visualize",
        }
        def_conf = Config()
        for key, val in d_extra.items():
            if def_conf.__getattribute__(key) != args.__getattribute__(key):
                instance.params.pypulseq.__setattr__(val, args.__getattribute__(key))

        instance._set_pp_sys_init_seq()
        return instance

    def _set_pp_sys_init_seq(self):
        log_module.info(f"set pypulseg system limits")
        self.pp_sys = pp.Opts(
            B0=self.params.specs.b_0,
            adc_dead_time=self.params.specs.adc_dead_time,
            gamma=self.params.specs.gamma,
            grad_raster_time=self.params.specs.grad_raster_time,
            grad_unit=self.params.specs.grad_unit,
            max_grad=self.params.specs.max_grad,
            max_slew=self.params.specs.max_slew,
            rf_dead_time=self.params.specs.rf_dead_time,
            rf_raster_time=self.params.specs.rf_raster_time,
            rf_ringdown_time=self.params.specs.rf_ringdown_time,
            rise_time=self.params.specs.rise_time,
            slew_unit=self.params.specs.slew_unit
        )
        self.pp_seq = pp.Sequence(system=self.pp_sys)

    def _set_name_fov(self) -> str:
        fov_r = int(self.params.pypulseq.resolutionFovRead)
        fov_p = int(self.params.pypulseq.resolutionFovPhase / 100 * self.params.pypulseq.resolutionFovRead)
        fov_s = int(self.params.pypulseq.resolutionSliceThickness * self.params.pypulseq.resolutionNumSlices)
        return f"fov{fov_r}-{fov_p}-{fov_s}"

    def _set_name_fa(self) -> str:
        return f"fa{int(self.params.pypulseq.refocusingFA[0])}"

    def write_seq(self, file_name: typing.Union[str, plib.Path]):
        file_name = plib.Path(file_name).absolute()
        save_file = file_name.with_suffix(".seq").__str__()
        log_module.info(f"writing file: {save_file}")
        self.pp_seq.write(save_file)

    def write_pypsi(self, output_path: typing.Union[str, plib.Path]):
        # make plib Path ifn
        output_path = plib.Path(output_path).absolute()
        # check exist
        if output_path.suffixes:
            output_path = output_path.parent
        output_path.mkdir(parents=True, exist_ok=True)
        # write
        self.params.save(output_path)

    def setDefinitions(self):
        self.pp_seq.set_definition(
            "FOV",
            [*self.params.pypulseq.get_fov()]
        )
        self.pp_seq.set_definition(
            "Name",
            f"jstmc{self.params.pypulseq.version}"
        )
        self.pp_seq.set_definition(
            "AdcRasterTime",
            1e-07
        )
        self.pp_seq.set_definition(
            "GradientRasterTime",
            self.params.specs.grad_raster_time
        )
        self.pp_seq.set_definition(
            "RadiofrequencyRasterTime",
            self.params.specs.rf_raster_time
        )


def create_cli() -> Config:
    """
        Build the parser for arguments
        Parse the input arguments.
        """
    args = sp.parse(Config)
    if args.d:
        logging.getLogger().setLevel(logging.DEBUG)
    return args


if __name__ == '__main__':
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)

    cli_args = create_cli()
    seq = Sequence.from_cli(args=cli_args)



