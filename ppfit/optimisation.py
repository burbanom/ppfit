import sys
import os
import numpy as np
from ppfit.inputoutput import output, mkdir_p
from ppfit.basin_hopping import MyTakeStep, WriteRestart, MyBounds
from ppfit.chi import sumOfChi
from scipy.optimize import basinhopping, minimize, OptimizeResult

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()



class LBFGSB_Minimizer:
    """Wrapper class for scipy.optimize.minimize( method = 'L-BFGS-B' )

    Attributes:
        options (dict of str: var): stores scipy.optimize.minimize() options
    """

    def __init__( self, opts ):
        """Initialise a LBFGSB_Minimizer object.

        Args:
            opts (ppfit.Options): Options objects with attributes { ftol_min, gtol_min, verbose, maxiter_min, stepsize_min } set.
        """
        self.options = { 'ftol': opts[ 'ftol' ],
                         'gtol': opts[ 'gtol' ],
                         'disp': opts[ 'verbose' ],
                         'maxiter': opts[ 'maxiter' ],
                         'eps': opts[ 'eps' ] }

    def minimize( self, function, initial_values, bounds, callback, **options): # can initial values and bounds be passed in as a Fitting_Parameter_Set object?
        """Minimize a function using scipy.optimize.minimize( method = 'L-BFGS-B' ).

        Args:
            function (function)
            initial_values (?)
            bounds (?)
        Returns:
            results_min (?)
        Notes:
        """

        if rank == 0:
           stop=[0]
           results_min = minimize( function, initial_values, args=stop, method = 'L-BFGS-B', bounds = bounds, options = self.options, callback= callback)
           stop=[1]
           function(initial_values, stop)
        else:
           stop=[0]
           results_min = OptimizeResult()
           while stop[0]==0:
              function(initial_values, stop)

        results_min = comm.bcast(results_min, root=0)

        return results_min

class Nelder_Mead_Minimizer:

    def __init__( self, opts ):

        self.options = opts

    def minimize( self, function, initial_values, callback, **options):
        if rank == 0:
           stop=[0]
           results_min = minimize( function, initial_values, args=stop, method = 'Nelder-Mead', options = self.options , callback= callback)
           stop=[1]
           function(initial_values, stop)
        else:
           stop=[0]
           results_min = OptimizeResult()
           while stop[0]==0:
              function(initial_values, stop)
          
        results_min = comm.bcast(results_min, root=0)  
        
        return results_min

class CG_Minimizer:

    def __init__( self, opts ):
        self.options = { 'gtol': opts[ 'tolerance'][ 'gtol' ],
                         'disp': opts[ 'verbose' ],
                         'maxiter': opts[ 'maxiter' ],
                         'eps': opts[ 'stepsize' ] }
        self.tol = opts[ 'tolerance' ][ 'gtol' ]

    def minimize( self, function, initial_values, callback, **options):
        if rank == 0:
           stop=[0]
           results_min = minimize( function, initial_values, args=stop, method = 'CG', tol = self.tol, options = self.options , callback= callback)
           stop=[1]
           function(initial_values, stop)
        else:
           stop=[0]
           results_min = OptimizeResult()
           while stop[0]==0:
              function(initial_values, stop)

        results_min = comm.bcast(results_min, root=0)

        return results_min

def optimise( function, fitting_parameters, opts ):
    tot_vars = ( fitting_parameters.to_fit + fitting_parameters.fixed ).strings
    pot_vars = fitting_parameters.to_fit.strings
    const_vars = fitting_parameters.fixed.strings
    const_values = np.asarray( fitting_parameters.fixed.initial_values )
    tot_values_min = ( fitting_parameters.fixed + fitting_parameters.to_fit ).min_bounds
    tot_values_max = ( fitting_parameters.fixed + fitting_parameters.to_fit ).max_bounds
    all_step_sizes = ( fitting_parameters.fixed + fitting_parameters.to_fit ).max_deltas
    step_sizes = np.asarray( fitting_parameters.to_fit.max_deltas )
    to_fit_and_not = ( fitting_parameters.fixed + fitting_parameters.to_fit ).is_fixed
    pot_values_min = fitting_parameters.to_fit.min_bounds
    pot_values_max = fitting_parameters.to_fit.max_bounds
    pot_values = np.asarray( fitting_parameters.to_fit.initial_values )
    
    ########################################################################################
    # basin hopping part using the minimization parameters as a starting guess             #
    # The temperature should be a fraction of the final function value from the minimizer  #
    ########################################################################################
    # Define variables for BH RESTART file 
    
    write_restart = WriteRestart(tot_vars,const_values,to_fit_and_not,tot_values_min,tot_values_max,all_step_sizes,'RESTART')
    #
    stop=[0]
    #if opts[ 'use_basin_hopping' ] == True:
    print( 'Basin Hopping' )
    output( 'Basin Hopping\n' )
    # Temperature parameter for BH
    if rank == 0:
       temperature =  function(pot_values,stop) * opts[ 'basin_hopping' ][ 'temperature' ]
       output( 'The temperature is set to: '+str(temperature)+'\n' )
    else: 
       function(pot_values, stop)

    # Set the options for the minimization algo in BH
    if opts[ 'basin_hopping' ][ 'method' ]['name'] == 'L-BFGS-B':

        try:
            ftol = np.float(opts['basin_hopping']['method'][ 'tolerance' ][ 'ftol' ])
        except:
            ftol = 1e-3

        try:
            gtol = np.float(opts['basin_hopping']['method']['tolerance' ][ 'gtol' ])
        except:
            gtol = 1e-3

        try:
            disp = opts[ 'verbose' ]
        except:
            disp = True

        try:
            eps = np.float(opts['basin_hopping']['method'][ 'stepsize' ])
        except:
            eps = 1e-08

        try:
            maxiter = np.int(opts['basin_hopping']['method'][ 'maxiter' ])
        except:
            maxiter = 1

        options= { 'ftol': ftol,
                        'gtol': gtol,
                        'disp': disp,
                        'eps':  eps,
                        'maxiter': maxiter }

    elif opts[ 'basin_hopping' ][ 'method' ]['name'] == 'CG':

        try:
            gtol = np.float(opts['basin_hopping']['method']['tolerance' ][ 'gtol' ])
        except:
            gtol = 1e-3

        try:
            disp = opts[ 'verbose' ]
        except:
            disp = True

        try:
            eps = np.float(opts['basin_hopping']['method'][ 'stepsize' ])
        except:
            eps = 1.4901161193847656e-08

        try:
            maxiter = np.int(opts['basin_hopping']['method'][ 'maxiter' ])
        except:
            maxiter = 1

        options= { 'gtol': gtol,
                        'disp': disp,
                        'eps':  eps,
                        'maxiter': maxiter }


    elif opts[ 'basin_hopping' ][ 'method' ]['name'] == 'Nelder-Mead':

        try:
            ftol = np.float(opts['basin_hopping']['method'][ 'tolerance' ][ 'ftol' ])
        except:
            ftol = 1e-3

        try:
            xtol = np.float(opts['basin_hopping']['method']['tolerance' ][ 'xtol' ])
        except:
            xtol = 1e-3

        try:
            disp = opts[ 'verbose' ]
        except:
            disp = True

        try:
            maxfev = np.int(opts['basin_hopping']['method'][ 'maxfev' ])
        except:
            maxfev = 1

        try:
            maxiter = np.int(opts['basin_hopping']['method'][ 'maxiter' ])
        except:
            maxiter = 1

        options= { 'ftol': ftol,
                        'xtol': xtol,
                        'disp': disp,
                        'maxfev': maxfev,
                        'maxiter': maxiter }

    else:
        sys.exit( 'minimization method {} not supported'.format( opts[ 'method' ][ 'name' ]) )

    # Bounds for BH
    mybounds = MyBounds(pot_values_max,pot_values_min)
    if opts[ 'basin_hopping' ][ 'method' ]['name'] in ('L-BFGS-B'):
        stop=[0]
        minimizer_kwargs = { 'method': opts[ 'basin_hopping' ][ 'method' ]['name'],
                             'bounds': fitting_parameters.to_fit.bounds,
                             'options': options,
                             'callback':  write_restart.write_local_restart,
                             'args': stop}
    else:
        minimizer = Nelder_Mead_Minimizer( options )
        stop=[0]
        minimizer_kwargs = { 'method': opts['basin_hopping']['method']['name'],
                             'options': options,
                             'callback': write_restart.write_local_restart,
                             'args': stop} 

    # Step sizes for BH
    if rank == 0:

       results_BH = basinhopping( function,
                               x0 = pot_values,
                               niter = opts[ 'basin_hopping' ][ 'niter' ], # TODO check this
                               T = temperature,
                               stepsize = opts[ 'basin_hopping'][ 'timestep' ],
                               minimizer_kwargs = minimizer_kwargs,
                               take_step = MyTakeStep( step_sizes ),
                               disp = opts[ 'verbose' ],
                               accept_test = mybounds,
                               callback = write_restart.write_bh_restart,
                               niter_success = opts[ 'basin_hopping' ][ 'niter_success' ] )

       stop=[1]
       function(pot_values, stop)
    else:
       stop=[0]
       results_BH = OptimizeResult()
       while stop[0]==0:
          function(pot_values, stop)
    
    results_BH = comm.bcast(results_BH, root=0)

    if rank ==0:
       output( results_BH.message[0] )
       tot_values = np.concatenate((const_values,results_BH.x),axis=0)
       ## Write a results file for the BH part 
       write_Results_BH = WriteRestart(tot_vars,const_values,to_fit_and_not,tot_values_min,tot_values_max,all_step_sizes,'RESULTS_BH')
       write_Results_BH.write_bh_restart( results_BH.x, results_BH.fun, accepted = 1 )

       # plot should take target filenames as arguments to save having to move afterwards
   
    stop=[0]
    function( results_BH.x, stop, plot = True )
    
    if rank == 0:
       mkdir_p('./BH-plots-pdfs')
       os.system('mv *.pdf ./BH-plots-pdfs')
