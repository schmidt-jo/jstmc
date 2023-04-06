# TMC Sequence with pypulseq
### turbo multi contrast multi-echo spin-echo sequence with auto-calibration central k-space region and echo-readout dependent outer k-space random phase encode sampling.

#### clone

`git clone --recurse-submodules`

#### env
build conda env via environments.yml

install pypulseq from dev branch fork using the recurse cloned submodule
`pip install -e <path_to_jstmc/pypulseq/>`

#### test
- Default settings are given in `options.py` for (1mm)³ isotropic voxels with 10 slices in a acquisition, using a 2 fold acceleration (40 central slices).
- Default system is our 7T system. You should! be updating those specs to whatever system you are using.
- The repo should only aid development of the MESE sequence. You should be familiar with pypulseq and take all necessary steps to check if a generated .seq file can be savely transferred to your scanner.
- Per Default a number of plots are generated showing the gradient pulse setup and echo train generation. Slice acquisition scheme and sampling scheme.
- The sampling scheme additionally is output as .csv file.

#### notes
- Correct setting of measurement headers is dependent on the system you are using. Hence reconstruction of the data could be streamlined. As is, the sequence and the output sampling_scheme.csv file would allow correct sorting of raw data and subsequent choice of custom reconstruction methods.
- At the moment we are using AC-LORAKS reconstruction to deal with the subsampling and subsequently dictionary based matching of the data using the emc approach also found on my git.
