import sys
import pathlib
sources_root_path = pathlib.Path(__file__).absolute().parent.parent
sys.path.append(sources_root_path.joinpath("pypulseq").as_posix())
sys.path.append(sources_root_path.joinpath("pypulseq_interface").as_posix())
