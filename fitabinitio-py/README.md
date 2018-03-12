This is an example of how to run the fitting program 
ppfit/fitabinitio. In order to perform a potential fit, 
you must:
1) Perform reference DFT single point calculations.
2) Prepare the input files (making sure everything is in the same UNITS)
in order to launch single point calculations with pimaim/cp2k and interatomic
potentials.
3) Modify the potential_template file to include the parameters that 
you will fit. Their initial values are specified in the PARAMS file.
4) Make sure that the runtime.inpt files and restart.dat files for each configuration
have the atoms in the same order, i.e. first the anions, then the cations.
5) Provided you specify the species in each configuration in the configs.yml, 
the python program that performs the fit SHOULD generate potential.inpt files that
are appropriate for each configuration (May not apply to potential types other than
XFT).
6) Modify the options.yml file to specify the location of your executable, scaling factor 
and optimization details.
7) Have fun?
