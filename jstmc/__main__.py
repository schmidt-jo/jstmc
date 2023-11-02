from jstmc import options, seq_v_jstmc, seq_v_vespa, plotting
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
    outpath = plib.Path(jstmc_seq.interface.config.output_path).absolute()
    # verifying
    if jstmc_seq.params.report:
        out_file = outpath.joinpath("report.txt")
        with open(out_file, "w") as w_file:
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
        # pyp_seq.plot(time_range=(0, 4e-3 * jstmc_seq.params.tr), time_disp='s')
        plotting.plot_seq(pyp_seq, out_path=outpath, name="seq_10s")
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
