from jstmc import options, sequence, utils
import numpy as np
import logging
from pathlib import Path
logging.getLogger('matplotlib.axis').disabled = True


def main():
    parser, prog_args = options.createCommandlineParser()

    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    logging.info("Starting sequence build")
    seq = options.Sequence.from_cmd_args(prog_args)
    seq.check_output_path()

    sbe = sequence.SequenceBlockEvents(seq=seq)
    sbe.build()
    seq = sbe.get_seq()

    logging.info("Plotting")
    path = Path("jstmc/images").absolute()
    # seq.ppSeq.plot()

    # utils.pretty_plot_et(seq, t_start=seq.params.TR, save=path.joinpath("echo_train_central_semc.png"))
    scan_time = np.sum(seq.ppSeq.arr_block_durations)
    # utils.pretty_plot_et(seq, t_start=scan_time * 1e3 - 4*seq.params.TR, plot_blips=True,
    # save=path.joinpath("echo_train_acc_tse.png"))
    logging.info(f"Total Scan Time: {scan_time / 60:.1f} min")

    # verifying
    outpath = Path(seq.config.outputPath).absolute().joinpath("report.txt")
    with open(outpath, "w") as w_file:
        report = seq.ppSeq.test_report()
        ok, err_rep = seq.ppSeq.check_timing()
        log = "report \n" + report + "\ntiming_check \n" + str(ok) + "\ntiming_error \n"
        w_file.write(log)
        for err_rep_item in err_rep:
            w_file.write(f"{str(err_rep_item)}\n")

    seq.save()
    logging.info(f".seq set definitions: {seq.ppSeq.dict_definitions}")
    seq.ppSeq.plot(time_range=(0, 2e-3 * seq.params.TR), time_disp='s')


if __name__ == '__main__':
    main()
