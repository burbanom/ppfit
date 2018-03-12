import numpy as np
import matplotlib
matplotlib.use( 'Agg' )
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from ppfit.inputoutput import output
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()



fmt="{0:.7f}"

def plot( ai_plot, ff_plot, sqDiff, filename, title ):
    with PdfPages( '{}.pdf'.format( filename ) ) as pdf:
        f, axarr = plt.subplots(2, sharex=True)
        att1 = {'label':'ai','color': 'black', 'markerfacecolor': None, 'markersize': 2.5,
        'markeredgewidth': 0.5, 'alpha': 1.0, 'marker': 'o',
        'markeredgecolor': 'black','linestyle' : ':'}
        att2 = {'label':'ff','color': 'blue', 'markerfacecolor': None, 'markersize': 2.5,
        'markeredgewidth': 0.5, 'alpha': 1.0, 'marker': 'o',
        'markeredgecolor': 'blue','linestyle' : 'None'}
        axarr[0].plot(ai_plot,**att1)
        axarr[0].plot(ff_plot,**att2)
        axarr[0].legend(loc='best')
        axarr[0].set_title( title )
        axarr[1].plot( sqDiff, 'r' )
        plt.ylabel('ERRORS')
        plt.xlabel('index')
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

def chi_squared( ai_vals, ff_vals, genplot, filename ):
    ai_s2 = ai_vals.var( axis = 1 )
    genplot = bool(genplot)
    sqDiff = []
    ai_plot = []
    ff_plot = []
    for i in range(np.shape(ai_vals)[-1]):
        numerator = 0.0
        denominator = 0.0
        ai_val = 0.0
        ff_val = 0.0
        for j,k,l in zip(ai_vals,ff_vals,ai_s2):
            if np.shape(ai_vals)[0] < 4:
                numerator += ((k[i]-j[i])**2)
                denominator += l
            else:
                numerator += ((k[i]-j[i])**2)
                denominator += (j[i])**2
            if genplot:
                ai_val += (j[i])**2
                ff_val += (k[i])**2
        sqDiff.append(numerator/denominator)
        if genplot:
            ai_plot.append(ai_val)
            ff_plot.append(ff_val)
    sqDiff = np.array(sqDiff)
    chiSq = np.sum(sqDiff)/len(sqDiff)
    if genplot:
        ai_plot = np.sqrt(ai_plot)
        ff_plot = np.sqrt(ff_plot)
        plot(  ai_plot, ff_plot, sqDiff, filename, title = 'Difference = {}'.format( str( chiSq ) ) )
    return chiSq

# This is the objective function evaluated by the minimization algorithms 
class sumOfChi:
  def __init__( self, potential_file, training_set, scaling ):
    self.potential_file = potential_file
    self.scaling = scaling
    self.training_set = training_set
    self.ai_forces = training_set.forces.T
    self.ai_dipoles = training_set.dipoles.T
    self.ai_stresses = training_set.stresses.T
    self.times_called = 0

  def evaluate( self,  test_values, stop, plot = False ):
    stop[0] = comm.bcast(stop[0], root=0)

    if stop[0] == 0:
      if type( test_values ) is not np.ndarray:
          raise TypeError
      if rank == 0:
         self.potential_file.write_with_parameters( test_values )

      ran_okay = self.training_set.run()
      
      if rank == 0:
        
        if not ran_okay:
            totalChi = 1E10
            output('Error: likely due to unphysical parameter value\n') 
            return totalChi
        
        ff_forces = self.training_set.new_forces.T
        ff_dipoles = self.training_set.new_dipoles.T
        ff_stresses = self.training_set.new_stresses.T
        self.times_called += 1

        chiSq = {}
        chiSq[ 'forces' ]   = chi_squared( self.ai_forces, ff_forces, plot, 'forces-errors' )
        chiSq[ 'dipoles' ]  = chi_squared( self.ai_dipoles, ff_dipoles, plot, 'dipoles-errors' )
        chiSq[ 'stresses' ] = chi_squared( self.ai_stresses, ff_stresses, plot, 'stresses-errors' )
        factorTot = sum( self.scaling.values() )
        totalChi = sum( [ self.scaling[ k ] * chiSq[ k ] for k in chiSq.keys() ] ) / factorTot
 
        output( 'Function call number: ' + str( self.times_called ) + '\n' )
        output( 'Forces chi sq: ' + fmt.format(chiSq[ 'forces' ]) + '\n')
        output( 'Dipoles chi sq: ' + fmt.format(chiSq[ 'dipoles' ]) +'\n')
        output( 'Stresses chi sq: ' + fmt.format(chiSq[ 'stresses' ]) + '\n')
        output( 'Total chi sq (no factors): ' + fmt.format( np.mean( list( chiSq.values() ) ) ) + '\n' )
        output('Total chi sq: '+fmt.format(totalChi)+'\n')
        output('\n')
        return totalChi
    else :
       output( 'function call finished on rank '+ str(rank) + '\n')

