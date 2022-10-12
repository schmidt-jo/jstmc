from jstmc import options, sequence, sar
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
    emc_info = sbe.get_emc_info()
    sampling_pattern = sbe.get_sampling_pattern()
    seq = sbe.get_seq()
    seq.write_sampling_pattern(sampling_pattern=sampling_pattern)

    scan_time = np.sum(seq.ppSeq.block_durations)
    logging.info(f"Total Scan Time: {scan_time / 60:.1f} min")

    logging.info(f"SAR calculations")
    sar.calc_sar(seq=seq)

    logging.info("Verifying and Writing Files")
    # verifying
    outpath = Path(seq.config.outputPath).absolute().joinpath("report.txt")
    with open(outpath, "w") as w_file:
        report = seq.ppSeq.test_report()
        ok, err_rep = seq.ppSeq.check_timing()
        log = "report \n" + report + "\ntiming_check \n" + str(ok) + "\ntiming_error \n"
        w_file.write(log)
        for err_rep_item in err_rep:
            w_file.write(f"{str(err_rep_item)}\n")

    seq.save(emc_info=emc_info, sampling_pattern=sampling_pattern)
    logging.info(f".seq set definitions: {seq.ppSeq.definitions}")

    logging.info("Plotting")

    # path = Path("jstmc/images").absolute()
    # seq.ppSeq.plot()

    # utils.pretty_plot_et(seq, t_start=seq.params.TR)  # , save=path.joinpath("echo_train_central_semc.png"))
    # utils.pretty_plot_et(seq, t_start=scan_time * 1e3 - 4*seq.params.TR, plot_blips=True)
    #  save=path.joinpath("echo_train_acc_tse.png"))

    seq.ppSeq.plot(time_range=(0, 2e-3 * seq.params.TR), time_disp='s')
    seq.ppSeq.plot(time_range=(scan_time - 2e-3 * seq.params.TR, scan_time - 1e-6), time_disp='s')


if __name__ == '__main__':
    main()
