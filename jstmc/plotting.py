import typing
import plotly.express as px
import logging
import pandas as pd
import pathlib as plib
log_module = logging.getLogger(__name__)


def plot_grad_moms(mom_df: pd.DataFrame, out_path: typing.Union[plib.Path, str], name: str):
    fig = px.line(mom_df, x="id", y="moments", color="axis")

    fig_path = plib.Path(out_path).absolute().joinpath("plots/")
    fig_path.mkdir(parents=True, exist_ok=True)
    fig_path = fig_path.joinpath(f"plot_{name}").with_suffix(".html")
    log_module.info(f"\t\t - writing file: {fig_path.as_posix()}")
    fig.write_html(fig_path.as_posix())


def plot_grad_moments(mom_df: pd.DataFrame, out_path: typing.Union[plib.Path, str], name: str):
    fig = px.scatter(mom_df, x="time", y="moments", color="id")

    fig_path = plib.Path(out_path).absolute().joinpath("plots/")
    fig_path.mkdir(parents=True, exist_ok=True)
    fig_path = fig_path.joinpath(f"plot_{name}").with_suffix(".html")
    log_module.info(f"\t\t - writing file: {fig_path.as_posix()}")
    fig.write_html(fig_path.as_posix())
