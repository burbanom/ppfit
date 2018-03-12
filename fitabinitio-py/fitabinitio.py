#! /usr/bin/env python3

import numpy as np
import os

from ppfit.chi import chi_squared, sumOfChi
from ppfit.configuration import Configuration
from ppfit.fitting_parameter_set import Fitting_Parameter_Set
from ppfit.potential_file import Potential_File
from ppfit.training_set import Training_Set
from ppfit.inputoutput import mkdir_p
from ppfit.optimisation import optimise
from ppfit.options import read_options

options = read_options( 'options.yml' )

outfile = open('OUTPUT','w')

config_read = read_options('configs.yml')

config=[]

for dict_name, value in config_read.items():
    config.append( Configuration( options = options,
                         species = config_read[str(dict_name)]["species"],
                         directory = config_read[str(dict_name)]["directory"],
                         runtime_file = config_read[str(dict_name)]["runtime_file"],
                         restart_file = config_read[str(dict_name)]["restart_file"],
                         forces_file  = config_read[str(dict_name)]["forces_file"],
                         dipoles_file = config_read[str(dict_name)]["dipoles_file"],
                         stresses_file = config_read[str(dict_name)]["stresses_file"] ))


training_set = Training_Set(config )

fitting_parameters = Fitting_Parameter_Set.from_parameters_file( 'PARAMS' )
potential_file = Potential_File( 'potential_template', fitting_parameters )

chi_squared_scaling = { 'forces':   options[ 'scaling' ][ 'forces' ],
                        'dipoles':  options[ 'scaling' ][ 'dipoles' ],
                        'stresses': options[ 'scaling' ][ 'stresses' ] }

sum_of_chi = sumOfChi( potential_file, training_set, chi_squared_scaling )

# # evaluate sum_of_chi with our initial potential parameters, and save a .pdf plot
# sum_of_chi.evaluate( fitting_parameters.to_fit.initial_values, plot = True )
# # TODO move plotting capabilities of sum_of_chi to a .plot method, and allow the target directory to be set here
# mkdir_p('./initial-errors-pdfs')
# os.system('mv *.pdf ./initial-errors-pdfs')

optimise( sum_of_chi.evaluate, fitting_parameters, options )
