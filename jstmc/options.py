import logging

import simple_parsing as sp
import dataclasses as dc
log_module = logging.getLogger(__name__)


@dc.dataclass
class Config:
    # Input sequence configuration
    i: str = sp.field(default="", alias="-c", help="Input Sequence Configuration")
    s: str = sp.field(default="./default_config/system_specifications_7T_TerraX_mpi_cbs.json",
                      help="Input System Specifications")
    o: str = sp.field(default="", help="Output Path for .seq file and pypsi interface,"
                                       "reverts to input location if blank")
    v: bool = sp.field(default=True, help="Visualize on/off")
    r: bool = sp.field(default=False, help="Report on/off")
    vv: str = sp.field(default="xx", help="Version")
    n: str = sp.field(default="jstmc", help="Sequence Name")

    d: bool = sp.field(default=False, help="Debug on/off")
    t: str = sp.field(default="vespa", choices=["vespa", "mese_fidnav"], help="Sequence Type")
    p: str = sp.field(default="png", choice=["png", "pdf", "html"], help="Plot file endings")


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
