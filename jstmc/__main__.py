from jstmc import options, seq_v_jstmc, seq_v_vespa
import numpy as np
import logging
import pathlib as plib
logging.getLogger('matplotlib.axis').disabled = True
logging.getLogger('simple_parsing').setLevel(logging.WARNING)


def main():
    prog_args = options.create_cli()

    logging.info("Starting sequence build")
    # setup sequence algorithm
    if prog_args.t == "vespa":
        jstmc_seq = seq_v_vespa.SeqVespaGerd.from_cli(args=prog_args)
    elif prog_args.t == "mese_fidnav":
        jstmc_seq = seq_v_jstmc.SeqJstmc.from_cli(args=prog_args)
    else:
        err = f"sequence type ({prog_args.t}) not recognised."
        logging.error(err)
        raise ValueError(err)
    # build sequence
    jstmc_seq.build()

    # gepypulseq sequence object
    pyp_seq = jstmc_seq.get_pypulseq_seq()
    scan_time = np.sum(pyp_seq.block_durations)
    logging.info(f"Total Scan Time Sum Seq File: {scan_time / 60:.1f} min")

    logging.info("Verifying and Writing Files")
    # verifying
    if jstmc_seq.params.report:
        outpath = plib.Path(jstmc_seq.interface.config.output_path).absolute().joinpath("report.txt")
        with open(outpath, "w") as w_file:
            report = pyp_seq.test_report()
            ok, err_rep = pyp_seq.check_timing()
            log = "report \n" + report + "\ntiming_check \n" + str(ok) + "\ntiming_error \n"
            w_file.write(log)
            for err_rep_item in err_rep:
                w_file.write(f"{str(err_rep_item)}\n")

    # saving
    jstmc_seq.write_seq()
    jstmc_seq.write_pypsi()

    logging.info(f".seq set definitions: {pyp_seq.definitions}")

    if jstmc_seq.interface.config.visualize:
        logging.info("Plotting")
        # give z and slice thickness both with same units. here mm
        # utils.plot_slice_acquisition(z * 1e3, seq.interface.resolutionSliceThickness)
        # utils.plot_sampling_pattern(sampling_pattern, seq_vars=seq)

        # utils.pretty_plot_et(seq, t_start=1e3 * scan_time / 2 - 2 * seq.interface.TR,
        #                      save=Path(options.RXV_Sequence.config.outputPath).absolute().joinpath("echo_train_semc"))

        pyp_seq.plot(time_range=(0, 4e-3 * jstmc_seq.params.tr), time_disp='s')
        # seq.ppSeq.plot(time_range=(scan_time - 2e-3 * seq.params.TR, scan_time - 1e-6), time_disp='s')
        jstmc_seq.interface.visualize()


if __name__ == '__main__':
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)

    logging.info("__________________________________________________________")
    logging.info("____________ jstmc pypulseq sequence creation ____________")
    logging.info("__________________________________________________________")
    try:
        main()
    except Exception as e:
        logging.exception(e)
