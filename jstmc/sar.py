import numpy as np
from pypulseq.SAR import SAR_calc
import pypulseq as pp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
from pathlib import Path
import typing

logModule = logging.getLogger(__name__)


def set_mpl():
    plt.style.use('ggplot')
    colors = cm.viridis(np.linspace(0, 0.9, 12))
    return colors


def calc_sar(seq: typing.Union[pp.Sequence, Path, str], visualize: bool = True):
    if isinstance(seq, (str, Path)):
        seq_pass = seq
        path = Path(seq_pass).absolute()
        path = path.with_stem(f"{path.stem}_sar_estimate")
        save_file = path.with_suffix(".png").__str__()
    elif isinstance(seq, pp.Sequence):
        seq_pass = seq
        path = Path(seq.config.outputPath).absolute()
        path.mkdir(parents=True, exist_ok=True)
        save_file = path.joinpath(f"jstmc{seq.config.version}_sar_estimate.png").__str__()
    else:
        raise AttributeError
    colors = set_mpl()
    head_tensec, head_sixmin, body_tensec, body_sixmin = SAR_calc.calc_SAR(seq_pass)
    t_total_sec = head_tensec.shape[0]

    body_limit_sixmin = 4
    body_limit_tensec = 8

    head_limit_sixmin = 3.2
    head_limit_tensec = 6.4

    if t_total_sec > 360:
        six_min_avg_head = np.sum(head_tensec[:360]) / 360
        six_min_avg_body = np.sum(body_tensec[:360]) / 360
        logModule.info(f"SAR 6 min avg: {six_min_avg_head:.2f} W/kg, head;  "
                       f"{six_min_avg_body:.2f} W/kg, body")
    else:
        six_min_avg_head = np.sum(head_tensec) / t_total_sec
        six_min_avg_body = np.sum(body_tensec) / t_total_sec
        logModule.info(f"SAR 6 min avg: {six_min_avg_head:.2f} W/kg, head;  "
                       f"{six_min_avg_body:.2f} W/kg, body")
    if six_min_avg_body > body_limit_sixmin:
        logModule.warning(f"SAR 6 min limits exceeded! - Whole body "
                          f"{six_min_avg_body:.2f} / {body_limit_sixmin:.2f} W/kg" )
    if six_min_avg_head > head_limit_sixmin:
        logModule.warning(f"SAR 6 min limits exceeded! - head "
                          f"{six_min_avg_head:.2f} / {head_limit_sixmin:.2f} W/kg")

    if visualize:
        fig = plt.figure(figsize=(10, 5.5))
        gs = fig.add_gridspec(2,1)
        ax = fig.add_subplot(gs[0])
        x_ax = np.arange(1, 1 + head_tensec.shape[0])
        ax.set_ylim(0, 5/4 * head_limit_tensec)

        # head
        # max
        ax.hlines(
            head_limit_tensec,
            0, head_tensec.shape[0],
            color=colors[0],
            label=f"10 sec limit: {head_limit_tensec:.2f} W/kg",
            linewidth=2
        )
        ax.fill_between(
            x_ax,
            np.full(head_tensec.shape, head_limit_tensec),
            alpha=0.2,
            color=colors[1]
        )

        ax.scatter(x_ax, head_tensec, color=colors[3], label=f"SAR ten second averages for head")
        ax.fill_between(x_ax, head_tensec, alpha=0.6, color=colors[4])

        # average
        avg = np.mean(head_tensec) * head_tensec.shape[0] / 360
        ax.hlines(
            avg,
            0, head_tensec.shape[0],
            color=colors[4],
            label=f"average: {avg:.2f} W/kg",
            linewidth=2
        )

        ax.set_xlabel(f"time [s]")
        ax.set_ylabel(f"SAR [W/kg]")

        ax.legend()

        # body
        ax = fig.add_subplot(gs[1])
        ax.set_ylim(0, 5 / 4 * body_limit_tensec)
        # max
        ax.hlines(
            body_limit_tensec,
            0, head_tensec.shape[0],
            color=colors[6],
            label=f"10 sec limit: {body_limit_tensec:.2f} W/kg",
            linewidth=2
        )
        ax.fill_between(
            x_ax,
            np.full(head_tensec.shape, body_limit_tensec),
            alpha=0.2,
            color=colors[7]
        )
        ax.scatter(x_ax, body_tensec, color=colors[8], label=f"SAR ten second averages for head")
        ax.fill_between(x_ax, body_tensec, alpha=0.6, color=colors[9])

        # average
        avg = np.mean(body_tensec) * head_tensec.shape[0] / 360
        ax.hlines(
            avg,
            0, head_tensec.shape[0],
            color=colors[10],
            label=f"average: {avg:.2f} W/kg",
            linewidth=2
        )

        ax.set_xlabel(f"time [s]")
        ax.set_ylabel(f"SAR [W/kg]")

        ax.legend()

        plt.tight_layout()
        plt.savefig(save_file, bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    PATH = Path("D:\\Daten\\01_Work\\04_ressources\\pulseq\\jstmc1a.seq").absolute()
    calc_sar(PATH)
