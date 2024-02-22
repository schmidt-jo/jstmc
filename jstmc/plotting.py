import typing
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as psub
import logging
import pandas as pd
import pathlib as plib
import pypulseq as pp
import numpy as np
log_module = logging.getLogger(__name__)


def create_fig_dir_ensure_exists(path: plib.Path):
    fig_path = plib.Path(path).absolute().joinpath("plots/")
    fig_path.mkdir(parents=True, exist_ok=True)
    return fig_path


def plot_grad_moms(mom_df: pd.DataFrame, out_path: typing.Union[plib.Path, str], name: str, file_suffix: str = "png"):
    fig = px.line(mom_df, x="id", y="moments", color="axis")
    fig_path = create_fig_dir_ensure_exists(out_path)
    fig_path = fig_path.joinpath(f"plot_{name}").with_suffix(f".{file_suffix}")
    log_module.info(f"\t\t - writing file: {fig_path.as_posix()}")
    if file_suffix in ["png", "pdf"]:
        fig.write_image(fig_path.as_posix())
    else:
        fig.write_html(fig_path.as_posix())


def plot_grad_moments(mom_df: pd.DataFrame, out_path: typing.Union[plib.Path, str], name: str, file_suffix: str = "html"):
    fig = px.scatter(mom_df, x="time", y="moments", color="id")
    fig_path = create_fig_dir_ensure_exists(out_path)
    fig_path = fig_path.joinpath(f"plot_{name}").with_suffix(f".{file_suffix}")
    log_module.info(f"\t\t - writing file: {fig_path.as_posix()}")
    if file_suffix in ["png", "pdf"]:
        fig.write_image(fig_path.as_posix())
    else:
        fig.write_html(fig_path.as_posix())


def plot_seq(seq: pp.Sequence, out_path: typing.Union[plib.Path, str], name: str,
             t_start_s: float = 0.0, t_end_s: float = 10.0, sim_grad_moments: bool = False, file_suffix: str = "html"):
    gamma = 42577478.518
    logging.debug(f"plot_seq")
    # transform to us
    t_start_us = int(t_start_s * 1e6)
    t_end_us = int(t_end_s * 1e6)
    t_total_us = t_end_us - t_start_us
    if t_total_us < 1:
        err = f"end time needs to be after start time"
        log_module.error(err)
        raise ValueError(err)
    # find starting idx
    start_idx = 0
    t_cum_us = 0
    # go through blocks
    for block_idx in range(len(seq.block_durations)):
        t_cum_us += 1e6 * seq.block_durations[block_idx]
        if t_cum_us > t_start_us:
            start_idx = block_idx
            break
        if block_idx == len(seq.block_durations) - 1:
            err = (f"looped through sequence blocks to get to {t_cum_us} us, "
                   f"and didnt arrive at starting time given {t_start_us} us")
            log_module.error(err)
            raise AttributeError(err)
    t_cum_us = 0
    # set up lists to fill with values
    times = []
    values = []
    labels = []
    # for simulating grad moments, track rf timings
    rf_flip_null = [[], []]

    grad_channels = ['gx', 'gy', 'gz']

    def append_to_lists(time: float | int | list, value: float | list, label: str):
        if isinstance(time, float) or isinstance(time, int):
            times.append(time)
            values.append(value)
            labels.append(label)
        else:
            times.extend(time)
            values.extend(value)
            labels.extend([label] * len(time))

    # start with first block after start time
    for block_idx in np.arange(start_idx, len(seq.block_durations)):
        # set start block
        t0 = t_cum_us
        block = seq.get_block(block_idx + 1)
        if t_cum_us + 1e6 * seq.block_durations[block_idx] > t_total_us:
            break

        if getattr(block, 'rf') is not None:
            rf = block.rf
            start = t0 + int(1e6 * rf.delay)
            # starting point at 0
            append_to_lists(start - 1e-6, 0.0, label="RF amp")
            append_to_lists(start - 1e-6, 0.0, label="RF phase")
            t_rf = rf.t * 1e6
            signal = np.abs(rf.signal)
            angle = np.angle(
                rf.signal * np.exp(1j * rf.phase_offset) * np.exp(1j * 2 * np.pi * rf.t * rf.freq_offset)
            )
            if sim_grad_moments:
                flip_angle = 2 * np.pi * np.trapz(x=rf.t, y=np.abs(signal))
                identifier = 0
                if np.pi / 3 < flip_angle < 2 * np.pi / 3:
                    identifier = 1
                if 2 * np.pi / 3 < flip_angle < 4 * np.pi / 3:
                    identifier = 2
                # assumes rf effect in the center
                append_to_lists(start + rf.shape_dur * 1e6 / 2, identifier,
                                label="RF grad moment effect")

            append_to_lists(time=(t_rf + start).tolist(), value=signal.tolist(), label="RF amp")
            append_to_lists(time=(t_rf + start).tolist(), value=angle.tolist(), label="RF phase")
            # set back to 0
            append_to_lists(t_rf[-1] + start + 1e-6, 0.0, label="RF amp")
            append_to_lists(t_rf[-1] + start + 1e-6, 0.0, label="RF phase")

        for x in range(len(grad_channels)):
            if getattr(block, grad_channels[x]) is not None:
                grad = getattr(block, grad_channels[x])
                if grad.type == 'trap':
                    amp_value = 1e3 * grad.amplitude / gamma
                elif grad.type == 'grad':
                    amp_value = 1e3 * grad.waveform / gamma
                else:
                    amp_value = 0
                start = int(t0 + 1e6 * grad.delay)
                t = grad.tt * 1e6
                append_to_lists(time=start + t, value=amp_value, label=f"GRAD {grad_channels[x]}")

        # %%
        if getattr(block, 'adc') is not None:
            adc = block.adc
            start = int(t0 + 1e6 * adc.delay)
            # set starting point to 0
            append_to_lists(time=start - 1e-6, value=0.0, label="ADC")
            end = int(start + adc.dwell * adc.num_samples * 1e6)
            dead_time = int(end + 1e6 * adc.dead_time)
            append_to_lists(time=[start, end, end + 1e-6, dead_time], value=[1.0, 1.0, 0.2, 0.2], label="ADC")
            # set end point to 0
            append_to_lists(time=dead_time + 1e-6, value=0.0, label="ADC")
        t_cum_us += int(1e6 * getattr(block, 'block_duration'))

    # build df
    df = pd.DataFrame({
        "data": values, "time": times, "labels": labels
    })
    num_rows = 2
    specs = [[{"secondary_y": True}], [{"secondary_y": True}]]

    # simulate grad moments
    if sim_grad_moments:
        df_grad_moments = simulate_grad_moments(
            df_rf_grads=df[df["labels"].isin(
                [*[f"GRAD {grad_channels[k]}" for k in range(3)], "RF grad moment effect"])]
        )
        num_rows += 1
        specs.append([{"secondary_y": False}])
    else:
        df_grad_moments = None

    fig = psub.make_subplots(
        num_rows, 1,
        specs=specs,
        shared_xaxes=True
    )

    # top axis left
    tmp_df = df[df["labels"] == "RF amp"]
    tmp_df.loc[:, "data"] = tmp_df["data"] / tmp_df["data"].max() * np.pi
    fig.add_trace(
        go.Scattergl(x=tmp_df["time"], y=tmp_df["data"], name="RF Amplitude"),
        1, 1, secondary_y=False
    )
    tmp_df = df[df["labels"] == "RF phase"]
    fig.add_trace(
        go.Scattergl(x=tmp_df["time"], y=tmp_df["data"], name="RF Phase [rad]", opacity=0.3),
        1, 1, secondary_y=False
    )
    # axes properties
    fig.update_yaxes(title_text="RF Amplitude & Phase", range=[-3.5, 3.5], row=1, col=1, secondary_y=False)
    # top axis right
    tmp_df = df[df["labels"] == "GRAD gz"]
    fig.add_trace(
        go.Scattergl(x=tmp_df["time"], y=tmp_df["data"], name="Gradient gz"),
        1, 1, secondary_y=True
    )
    fig.update_yaxes(
        title_text="Gradient Slice [mT/m]",
        range=[-1.2*np.max(np.abs(tmp_df["data"].to_numpy())), 1.2*np.max(np.abs(tmp_df["data"].to_numpy()))],
        row=1, col=1, secondary_y=True
    )
    # bottom axis left
    tmp_df = df[df["labels"] == "ADC"]
    fig.add_trace(
        go.Scattergl(x=tmp_df["time"], y=tmp_df["data"], name="ADC", fill="tozeroy", opacity=0.5),
        2, 1, secondary_y=False
    )
    fig.update_xaxes(title_text="Time [us]", row=2, col=1)
    fig.update_yaxes(title_text="ADC", range=[-1.5, 1.5], row=2, col=1, secondary_y=False)
    # bottom axis right
    max_val = 40
    for k in range(2):
        tmp_df = df[df["labels"] == f"GRAD {grad_channels[k]}"]
        fig.add_trace(
            go.Scattergl(x=tmp_df["time"], y=tmp_df["data"], name=f"Gradient {grad_channels[k]} [mT/m]"),
            2, 1, secondary_y=True
        )
        if max_val < np.max(np.abs(tmp_df["data"].to_numpy())):
            max_val = np.max(np.abs(tmp_df["data"].to_numpy()))
    fig.update_yaxes(title_text="Gradient [mT/m]", range=[-1.2 * max_val, 1.2*max_val], row=2, col=1, secondary_y=True)

    # add gradient moment simulation
    if sim_grad_moments:
        max_val = np.max(np.abs(df_grad_moments[df_grad_moments["labels"].isin(
                [*[f"GRAD {grad_channels[k]}" for k in range(3)]])]["moment"].to_numpy()))
        for k in range(3):
            tmp_df = df_grad_moments[df_grad_moments["labels"] == f"GRAD {grad_channels[k]}"].sort_values(by="time")
            fig.add_trace(
                go.Scattergl(
                    x=tmp_df["time"], y=tmp_df["moment"], name=f"Gradient Moment {grad_channels[k]} [mTs/m]"
                ),
                3, 1
            )
        fig.update_yaxes(title_text="Gradient Moment [mT s/m]", range=[-1.2 * max_val, 1.2 * max_val], row=3, col=1,
                         secondary_y=False)
    fig.update_layout(
        width=1000,
        height=800
    )
    fig_path = create_fig_dir_ensure_exists(out_path)
    fig_path = fig_path.joinpath(f"plot_{name}").with_suffix(f".{file_suffix}")
    log_module.info(f"\t\t - writing file: {fig_path.as_posix()}")
    if file_suffix in ["png", "pdf"]:
        fig.write_image(fig_path.as_posix())
    else:
        fig.write_html(fig_path.as_posix())


def simulate_grad_moments(df_rf_grads: pd.DataFrame):
    # gradient amplitudes are named after axes
    grad_channels = ['gx', 'gy', 'gz']
    # we have a dataframe with all gradient amplitudes and rf effects on moment [0=None, 1=Null, 2=Flip]
    df_rf_grads = df_rf_grads.copy().sort_values(by=["time"])
    grad_moments = {
        "GRAD gx": {"moment": [], "time": [], "last_time": 0.0, "last_amp": 0.0},
        "GRAD gy": {"moment": [], "time": [], "last_time": 0.0, "last_amp": 0.0},
        "GRAD gz": {"moment": [], "time": [], "last_time": 0.0, "last_amp": 0.0}
    }
    # we move through the time sorted entries
    for value in df_rf_grads.values:
        data, time, label = value
        # if we encounter a gradient we calculate the respective moment change
        if label in grad_moments.keys():
            # if no previous moment entry we skip
            if not grad_moments[label]["moment"]:
                # collect the time
                grad_moments[label]["time"].append(time)
                grad_moments[label]["moment"].append(0.0)
            else:
                # cummulate gradient
                grad_moments[label]["moment"].append(
                    grad_moments[label]["moment"][-1] + np.trapz(
                        x=[grad_moments[label]["last_time"], time],
                        y=[grad_moments[label]["last_amp"], data]
                    ) * 1e-6
                )
                grad_moments[label]["time"].append(time)
            grad_moments[label]["last_amp"] = data
            grad_moments[label]["last_time"] = time
        if label == "RF grad moment effect":
            # we calculate the moment until this point for all axes, saved last gradient amp for all gradients,
            # assumes no gradient changes across an RF
            data = int(data)
            for label in grad_moments.keys():
                # if accessed before
                if grad_moments[label]["moment"]:
                    # moment til rf center
                    grad_moments[label]["time"].append(time - 1e-9)
                    grad_moments[label]["moment"].append(
                        grad_moments[label]["moment"][-1] + np.trapz(
                            x=[grad_moments[label]["last_time"], time],
                            y=[grad_moments[label]["last_amp"]] * 2
                        ) * 1e-6
                    )
                    # save as last accessed time
                    grad_moments[label]["last_time"] = time
                    # RF effect
                    grad_moments[label]["time"].append(time + 1e-9)
                    if data == 1:
                        # grad mom null
                        grad_moments[label]["moment"].append(0.0)
                    if data == 2:
                        # grad mom flip
                        grad_moments[label]["moment"].append(-grad_moments[label]["moment"][-1])

    # build dataframe
    times = []
    moments = []
    labels = []
    for key in grad_moments.keys():
        times.extend(grad_moments[key]["time"])
        moments.extend(grad_moments[key]["moment"])
        labels.extend([key] * len(grad_moments[key]["time"]))
    return pd.DataFrame({
        "time": times, "moment": moments, "labels": labels
    })
