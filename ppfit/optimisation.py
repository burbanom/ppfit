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
        self.options = { 'ftol': opts[ 'tolerance' ][ 'ftol' ],
                         'gtol': opts[ 'tolerance' ][ 'gtol' ],
                         'disp': opts[ 'verbose' ],
                         'maxiter': opts[ 'maxiter' ]
                           }#,
                         #'eps': opts[ 'stepsize' ] }

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
        self.options= { 'ftol': opts[ 'tolerance' ][ 'ftol' ],
                        'xtol': opts[ 'tolerance' ][ 'xtol' ],
                        'disp': opts[ 'verbose' ],
                        'maxfev': opts[ 'maxfev' ],
                        'maxiter': opts['maxiter'] }

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
    
 
    write_restart = WriteRestart(tot_vars,const_values,to_fit_and_not,tot_values_min,tot_values_max,all_step_sizes,'RESTART')

    # Choose the calculation order
    if ( 'use_basin_hopping' not in opts.keys() or
         opts[ 'use_basin_hopping' ] == False or
         opts[ 'basin_hopping' ][ 'calc_order' ] == 0 ): 
    # what happens if we are not using basin hopping?
        if opts[ 'method' ] == 'L-BFGS-B':
            minimizer = LBFGSB_Minimizer( opts )
            results_min = minimizer.minimize( function, pot_values, bounds = fitting_parameters.to_fit.bounds, callback =  write_restart.write_local_restart)
            if rank == 0:
              output( results_min.message.decode("utf-8") )
        elif opts[ 'method' ] == 'CG':
            minimizer = CG_Minimizer( opts )
            results_min = minimizer.minimize( function, pot_values, callback =  write_restart.write_local_restart)
            if rank ==0:
               output( results_min.message )
        elif opts[ 'method' ] == 'Nelder-Mead':
            minimizer = Nelder_Mead_Minimizer( opts )
            results_min = minimizer.minimize( function, pot_values, callback =  write_restart.write_local_restart)
            if rank ==0:
               output( results_min.message)
            
        else:
            sys.exit( 'minimization method {} not supported'.format( opts[ 'method' ] ) )
       
        # Write a results file
        if rank == 0: 
           tot_values = np.concatenate((const_values,results_min.x),axis=0)
           write_Results_min = WriteRestart(tot_vars,const_values,to_fit_and_not,tot_values_min,tot_values_max,all_step_sizes,'RESULTS_min')
           write_Results_min.write_bh_restart(results_min.x,results_min.fun,accepted=1)
        
        stop=[0]
        function(results_min.x, stop, plot = True)

        if rank == 0:
           mkdir_p('./min-errors-pdfs')
           os.system('mv *.pdf ./min-errors-pdfs')
 
########################################################################################
# basin hopping part using the minimization parameters as a starting guess             #
# The temperature should be a fraction of the final function value from the minimizer  #
########################################################################################
# Define variables for BH RESTART file 
    
    write_restart = WriteRestart(tot_vars,const_values,to_fit_and_not,tot_values_min,tot_values_max,all_step_sizes,'RESTART')
#
    stop=[0]
    if opts[ 'use_basin_hopping' ] == True:
        print( 'Basin Hopping' )
        if opts[ 'basin_hopping' ]['calc_order' ] == 0:
        # Pass the optimized values to BH
            pot_values = results_min.x
        # Temperature parameter for BH
            if rank == 0:
               temperature = results_min.fun * opts[ 'basin_hopping']['temperature' ]
        elif opts[ 'basin_hopping' ]['calc_order' ] == 1:
        # Temperature parameter for BH
            if rank == 0:
               temperature =  function(pot_values,stop) * opts[ 'basin_hopping' ][ 'temperature' ]
            else: 
               function(pot_values, stop)
        else:
            exit( 'not recognised as basin hopping calculation order: {}'.format( opts[ 'basin_hopping' ][ 'calc_order' ] ) )
            # Step sizes for BH
        if rank == 0:
           output( 'The temperature is set to: '+str(temperature)+'\n' )
    # Set the options for the minimization algo in BH
        if opts[ 'basin_hopping' ][ 'method' ] == 'L-BFGS-B':
            options = { 'ftol': opts[ 'basin_hopping' ][ 'tolerance' ][ 'ftol' ],
                        'gtol': opts[ 'basin_hopping' ][ 'tolerance' ][ 'gtol' ],
                        'disp': opts[ 'verbose' ],
                        'maxiter': opts[ 'basin_hopping' ][ 'maxiter' ]}
        elif opts[ 'basin_hopping' ][ 'method' ] == 'CG':
            options = {  'gtol': opts[ 'basin_hopping' ][ 'tolerance'][ 'gtol' ],
                         'disp': opts[ 'basin_hopping' ][ 'verbose' ],
                         'maxiter': opts[ 'basin_hopping' ][ 'maxiter' ],
                         'eps': opts[ 'basin_hopping' ][ 'stepsize' ] }
        elif opts[ 'basin_hopping' ][ 'method' ] == 'Nelder-Mead':
            options = { 'ftol': opts[ 'basin_hopping' ][ 'tolerance' ][ 'ftol' ],
                        'xtol': opts[ 'basin_hopping' ][ 'tolerance' ][ 'xtol' ],
                        'disp': opts[ 'verbose' ],
                        'maxfev': opts[ 'basin_hopping' ][ 'maxfev' ],
                        'maxiter': opts[ 'basin_hopping' ]['maxiter'] }
        else:
            sys.exit( 'minimization method {} not supported'.format( opts[ 'method' ] ) )

    # Bounds for BH
        mybounds = MyBounds(pot_values_max,pot_values_min)
        if opts[ 'basin_hopping' ][ 'method' ] in ('L-BFGS-B'):
            stop=[0]
            minimizer_kwargs = { 'method': opts[ 'basin_hopping' ][ 'method' ],
                                 'bounds': fitting_parameters.to_fit.bounds,
                                 'options': options,
                                 'callback':  write_restart.write_local_restart,
                                 'args': stop}
        else:
            minimizer = Nelder_Mead_Minimizer( opts )
            stop=[0]
            minimizer_kwargs = { 'method': opts['basin_hopping']['method'],
                                 'options': options,
                                 'callback': write_restart.write_local_restart,
                                 'args': stop} 

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
           mkdir_p('./BH-errors-pdfs')
           os.system('mv *.pdf ./BH-errors-pdfs')
         
        #
