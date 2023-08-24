from jstmc import options, seq_v_jstmc, seq_v_vespa, sar, utils
import numpy as np
import logging
from pathlib import Path
logging.getLogger('matplotlib.axis').disabled = True


def main():
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)

    prog_args = options.create_cli()

    logging.info("Starting sequence build")
    # build seq object from parsed args
    seq = options.Sequence.from_cli(args=prog_args)

    # setup sequence algorithm
    # jstmc_algo = seq_v_jstmc.JsTmcSequence(seq_opts=seq)
    jstmc_algo = seq_v_vespa.VespaGerdSequence(seq_opts=seq)
    # build sequence
    jstmc_algo.build()

    # get emc_info
    emc_info = jstmc_algo.get_emc_info()
    # get slice info
    z = jstmc_algo.get_z()
    # get sampling pattern
    sampling_pattern = jstmc_algo.get_sampling_pattern()
    # get pulse info
    pulse = jstmc_algo.get_pulse_rfpf()
    # get sequence info for recon
    seq_info = jstmc_algo.get_sequence_info()
    # get seq object
    seq = jstmc_algo.get_seq()

    scan_time = np.sum(seq.pp_seq.block_durations)
    logging.info(f"Total Scan Time Sum Seq File: {scan_time / 60:.1f} min")

    logging.info("Verifying and Writing Files")
    # verifying
    if seq.config.report:
        outpath = Path(seq.config.outputPath).absolute().joinpath("report.txt")
        with open(outpath, "w") as w_file:
            report = seq.pp_seq.test_report()
            ok, err_rep = seq.pp_seq.check_timing()
            log = "report \n" + report + "\ntiming_check \n" + str(ok) + "\ntiming_error \n"
            w_file.write(log)
            for err_rep_item in err_rep:
                w_file.write(f"{str(err_rep_item)}\n")

    # saving details
    seq.save(emc_info=emc_info, sampling_pattern=sampling_pattern, pulse_signal=pulse, sequence_info=seq_info)
    logging.info(f".seq set definitions: {seq.pp_seq.definitions}")

    if seq.config.visualize:
        logging.info("Plotting")
        # give z and slice thickness both with same units. here mm
        utils.plot_slice_acquisition(z*1e3, seq.params.resolutionSliceThickness)
        # utils.plot_sampling_pattern(sampling_pattern, seq_vars=seq)

        utils.pretty_plot_et(seq, t_start=1e3 * scan_time / 2 - 2 * seq.params.TR,
                             save=Path(options.Sequence.config.outputPath).absolute().joinpath("echo_train_semc"))

        seq.pp_seq.plot(time_range=(0, 2e-3 * seq.params.TR), time_disp='s')
        # seq.ppSeq.plot(time_range=(scan_time - 2e-3 * seq.params.TR, scan_time - 1e-6), time_disp='s')


if __name__ == '__main__':
    main()
