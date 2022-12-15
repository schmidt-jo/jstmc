import logging
import typing

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from jstmc import options
from pathlib import Path
import tqdm

logModule = logging.getLogger(__name__)


def pretty_plot_et(seq: options.Sequence,
                   save: typing.Union[str, Path] = "",
                   plot_blips: bool = False,
                   t_start: int = 0,
                   figsize: tuple = (12, 6)):
    linewidth=2
    t_start *= 1000     # cast to us
    logging.info(f"plot")

    # set time until which to plot (taking 1echo time more for nice plot)
    t_total = (seq.params.ETL + 2) * seq.params.ESP * 1000  # in us
    # leave first x free
    t_cum = 0
    # build x ax
    x_arr = np.arange(-t_cum, int(t_total))
    # init arrays
    arr_rf = np.zeros((2, len(x_arr)))  # [amplitude,phase]
    arr_g = np.zeros((3, len(x_arr)))  # [gx, gy, gz]
    arr_adc = np.zeros_like(x_arr)
    block_end = 0

    # for plotting
    colors = cm.viridis(np.linspace(0, 0.9, 10))

    def configure_axes_twin(ax: plt.axis,
                            y_ax_offset: bool = False,
                            y_label: str = "",
                            color: int = 0,
                            max_val: float = 1,
                            grid: bool = False):
        ax.grid(grid)
        ax.set_xlabel('time [ms]')
        ax.set_ylabel(f"{y_label}")
        ax.yaxis.label.set_color(colors[color])
        ax.spines["right"].set_edgecolor(colors[color])
        ax.tick_params(axis='y', colors=colors[color])
        if y_ax_offset:
            ax.spines.right.set_position(("axes", 1.08))
        ax.set_ylim(-max_val, max_val)
        return ax

    # find starting idx
    start_idx = 0
    for block_idx in range(len(seq.ppSeq.block_durations)):
        t_cum += 1e6 * seq.ppSeq.block_durations[block_idx]
        if t_cum > t_start:
            start_idx = block_idx
            block = seq.ppSeq.get_block(block_idx + 1)
            # check if we found excitation pulse, ie. start of echo train
            if getattr(block, "rf") is not None:
                if block.rf.use == "excitation":
                    break
    t_cum = 0
    for block_idx in np.arange(start_idx, len(seq.ppSeq.block_durations)):
        t0 = t_cum
        block = seq.ppSeq.get_block(block_idx + 1)
        if t_cum + 1e6 * seq.ppSeq.block_durations[block_idx] > t_total:
            break

        if getattr(block, 'rf') is not None:
            rf = block.rf
            delay = int(1e6 * rf.delay)
            start = t0 + delay
            end = t0 + delay + len(rf.signal)
            arr_rf[0, start:end] = np.abs(rf.signal)
            arr_rf[1, start:end] = np.angle(
                rf.signal * np.exp(1j * rf.phase_offset) * np.exp(1j * 2 * np.pi * rf.t * rf.freq_offset))
            if block_end < end:
                block_end = end

        grad_channels = ['gx', 'gy', 'gz']
        for x in range(len(grad_channels)):
            if getattr(block, grad_channels[x]) is not None:
                grad = getattr(block, grad_channels[x])
                if grad.type == 'trap':
                    amp_value = 1e3 * grad.amplitude / seq.specs.gamma
                    start = int(t0 + 1e6 * grad.delay)
                    end = start + int(1e6 * grad.rise_time)
                    arr_g[x, start:end] = np.linspace(0, amp_value, num=end - start)
                    start = end
                    end = start + int(1e6 * grad.flat_time)
                    arr_g[x, start:end] = amp_value
                    start = end
                    end = start + int(1e6 * grad.fall_time)
                    arr_g[x, start:end] = np.linspace(amp_value, 0, num=end - start)
                    if block_end < end:
                        block_end = end
                if grad.type == 'grad':
                    start = int(t0 + 1e6 * grad.delay)
                    end = int(start + 1e6 * grad.shape_dur)
                    wf = np.zeros(end - start)
                    for idx_t in range(len(grad.tt) - 1):
                        idx_start = int(grad.tt[idx_t] * 1e6)
                        idx_end = int(grad.tt[idx_t+1] * 1e6)
                        val_start = 1e3 * grad.waveform[idx_t] / seq.specs.gamma
                        val_end = 1e3 * grad.waveform[idx_t+1] / seq.specs.gamma
                        wf[idx_start:idx_end] = np.linspace(val_start, val_end, idx_end - idx_start)
                    arr_g[x, start:end] = wf
        # %%
        if getattr(block, 'adc') is not None:
            adc = block.adc
            start = int(t0 + 1e6*adc.delay)
            end = start + int(adc.num_samples * adc.dwell * 1e6)
            arr_adc[start:end] = 1
            if block_end < end:
                block_end = end
        t_cum = block_end

    plt.style.use('ggplot')
    fig = plt.figure(figsize=figsize, dpi=100)
    gs = fig.add_gridspec(2, 1)

    # ax
    ax_g = fig.add_subplot(gs[0])

    # rf phase
    ax_phase = ax_g.twinx()
    ax_phase = configure_axes_twin(
        ax_phase, y_ax_offset=True, y_label=f"rf phase [$\pi$]",
        color=-1, max_val=1.1 * np.max(arr_rf[1])
    )
    ax_phase.fill_between(x_arr / 1000, arr_rf[1] / np.pi, color=colors[-1], alpha=0.6)
    ax_phase.plot(x_arr / 1000, arr_rf[1] / np.pi, color=colors[-1])

    # gz
    gz_max = 1.1 * np.max(np.abs(arr_g[2]))
    ax_g = configure_axes_twin(ax_g, y_label=f"$g_z [mT/m]$", color=6, max_val=gz_max, grid=True)
    ax_g.fill_between(x_arr / 1000, arr_g[2], alpha=0.6, color=colors[6])
    ax_g.plot(x_arr / 1000, arr_g[2], c=colors[6])

    # rf signal
    ax_rf = ax_g.twinx()
    ax_rf = configure_axes_twin(ax_rf, y_label="rf amplitude", color=0, max_val=1.1*np.max(arr_rf[0]))
    ax_rf.plot(x_arr/1000, arr_rf[0], c=colors[0], linewidth=linewidth)

    # adc
    ax_adc = fig.add_subplot(gs[1])
    ax_adc = configure_axes_twin(ax_adc, y_label="ADC", color=2, max_val=1.1, grid=True)
    ax_adc.axes.yaxis.set_ticklabels([])
    ax_adc.fill_between(x_arr / 1000, arr_adc, alpha=0.5, color=colors[2])

    # gx
    ax_g = ax_adc.twinx()
    gx_max = 1.1*np.max(np.abs(arr_g[0]))
    ax_g = configure_axes_twin(ax_g, y_label=f"$g_x$ [mT/m]", color=8, max_val=gx_max)
    ax_g.plot(x_arr / 1000, arr_g[0], c=colors[8], linewidth=linewidth)
    ax_g.fill_between(x_arr / 1000, arr_g[0], color=colors[9], alpha=0.4, hatch='/')

    # gy
    ax_gy = ax_adc.twinx()
    gy_max = 1.1*np.max(np.abs(arr_g[1]))
    ax_gy = configure_axes_twin(ax_gy, y_ax_offset=True, y_label=f"$g_y$ [mT/m]", color=5, max_val=gy_max)
    ax_gy.plot(x_arr / 1000, arr_g[1], c=colors[5], label=f"$g_y$", zorder=1, linewidth=linewidth)

    if plot_blips:
        pos_idx = arr_g[1] > 0.8 * np.max(arr_g[1])
        neg_idx = arr_g[1] < 0.8 * np.min(arr_g[1])

        ax_gy.fill_between(x_arr / 1000, np.max(arr_g[1]), arr_g[1], where=pos_idx, color='orange',
                           alpha=0.7, hatch='/', label='grad blips')
        ax_gy.fill_between(x_arr / 1000, np.min(arr_g[1]), arr_g[1], where=neg_idx, color='orange',
                           alpha=0.7, hatch='/')
        ax_gy.legend()

    ax_gy.fill_between(x_arr / 1000, arr_g[1], color=colors[5], alpha=0.4, hatch='/')

    plt.tight_layout()

    if save:
        logModule.info(f"saving plot-file: {save}")
        plt.savefig(save, bbox_inches="tight", dpi=100)
    plt.show()


def plot_sampling_pattern(sampling_pattern: list, seq_vars: options.Sequence):
    n_read = seq_vars.params.resolutionNRead
    n_phase = seq_vars.params.resolutionNPhase
    x_ax = np.tile(np.arange(n_read), seq_vars.params.ETL)
    y_ax = np.arange(n_phase) - int(n_phase / 2)
    plot_arr = np.zeros((seq_vars.params.resolutionNPhase, n_read * seq_vars.params.ETL))
    for s_indices in tqdm.tqdm(sampling_pattern, desc="processing sampling scheme"):
        pe_num = s_indices['pe_num']
        echo_num = s_indices['echo_num']
        plot_arr[pe_num, echo_num*n_read:(echo_num+1)*n_read] = 1

    x_labels = np.arange(1, seq_vars.params.ETL + 1)
    x_pos = np.array([n_read/2 + k*n_read for k in range(seq_vars.params.ETL)])
    y_labels = [0]
    y_pos = [int(n_phase/2)]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.grid(False)
    ax.set_xlabel("# echo - freq encode")
    ax.set_ylabel("# phase encode")
    ax.set_xticks(x_pos, labels=x_labels)
    ax.set_yticks(y_pos, labels=y_labels)
    ax.imshow(plot_arr, interpolation='None', aspect=seq_vars.params.ETL)
    plt.show()


if __name__ == '__main__':
    # set up path
    seq_path = Path(
        "D:\\Daten\\01_Work\\11_owncloud\\ds_mese_cbs_js\\97_pulseq\\sequence\\seq_1a_fa180_fov_210-166-10_RL\\jstmc1a_fa180_fov210-165-14_RL.seq"
    ).absolute()
    seq = options.Sequence.load(seq_path)
    scan_time = np.sum(seq.ppSeq.block_durations)
    pretty_plot_et(seq, plot_blips=True, t_start=0, figsize=(10, 5))


