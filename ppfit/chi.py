import numpy as np
import matplotlib
matplotlib.use( 'Agg' )
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from ppfit.inputoutput import output

fmt="{0:.7f}"

def plot( ai_vals, ff_vals, ai_plot, ff_plot, diff_plot, sqDiff, sqDiff_s2, filename, title ):
    with PdfPages( '{}.pdf'.format( filename ) ) as pdf:
        f, axarr = plt.subplots(3, sharex=True)
        att1 = {'color': 'black', 'markerfacecolor': None, 'markersize': 2.5, 
        'markeredgewidth': 0.5, 'alpha': 1.0, 'marker': 'o', 
        'markeredgecolor': 'black','linestyle' : ':','label' : 'AI'} 
        att2 = {'color': 'blue', 'markerfacecolor': None, 'markersize': 2.5, 
        'markeredgewidth': 0.5, 'alpha': 1.0, 'marker': 'o', 
        'markeredgecolor': 'blue','linestyle' : 'None','label' : 'FF'} 
        axarr[0].plot(ai_plot,**att1)
        axarr[0].plot(ff_plot,**att2)
        axarr[0].legend(loc='best')
        axarr[0].set_title( title )
        axarr[0].set_ylabel('Norm')
        axarr[1].plot(diff_plot,'go',markersize=2.5,markeredgewidth=0.5,label='Diff.')
        axarr[1].legend(loc='best')
        axarr[1].set_ylabel('Diff.')
        axarr[2].plot(sqDiff,'ro',markersize=2.5,markeredgewidth=0.5,label='/AI^2')
        axarr[2].plot(sqDiff_s2,'co:',markersize=2.5,markeredgewidth=0.5,label='/S^2')
        axarr[2].legend(loc='best')
        axarr[2].set_ylabel('ERRORS')
        axarr[2].set_xlabel('index')
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        maxFF = np.max(np.abs(ff_vals))
        maxAI = np.max(np.abs(ai_vals))
        limit = np.max([maxFF,maxAI])
        for i,j in zip(ai_vals,ff_vals):
            plt.plot(i,j,'o',markersize=2.5,markeredgewidth=0.5)
            plt.ylabel('FF')
            plt.xlabel('AI')
            plt.ylim([-limit-0.1*limit,limit+0.1*limit])
            plt.xlim([-limit-0.1*limit,limit+0.1*limit])
            plt.axes().set_aspect('equal')
        plt.plot([-limit,limit],[-limit,limit],'k--')
        pdf.savefig()
        plt.close()

def chi_squared( ai_vals, ff_vals, genplot, filename ):
    ai_s2 = ai_vals.var( axis = 1 )
    genplot = bool(genplot)
    sqDiff = []
    sqDiff_s2 = []
    ai_plot = []
    ff_plot = []
    diff_plot = []
    denominator_s2 = np.sum(ai_s2)
    for ai_val,ff_val  in zip( ai_vals.T, ff_vals.T ):
        numerator = np.sum(( ff_val - ai_val )**2)
        denominator = np.sum(ai_val**2)
        if ( denominator == 0.000 ):
            sqDiff.append( numerator )
        else:
            sqDiff.append( numerator / denominator )
        if ( denominator_s2 == 0.000 and denominator != 0.000 ):
            sqDiff_s2.append( numerator / denominator )
        elif( denominator_s2 == 0.000 and denominator == 0.000 ):
            sqDiff_s2.append( numerator )
        else:
            sqDiff_s2.append( numerator / denominator_s2 )
        if genplot:
            diff_plot.append(numerator)
            ai_plot.append(denominator)
            ff_plot.append(np.sum(ff_val**2))
    sqDiff = np.array(sqDiff)
    sqDiff_s2 = np.array(sqDiff_s2)
    chiSq = np.sum(sqDiff) / len(sqDiff)
    chiSq_s2 = np.sum(sqDiff_s2) / len(sqDiff_s2)
    if genplot:
        diff_plot = np.array(diff_plot)
        ai_plot = np.sqrt(ai_plot)
        ff_plot = np.sqrt(ff_plot)
        plot(  ai_vals, ff_vals, ai_plot, ff_plot, diff_plot, sqDiff, sqDiff_s2, 
                filename, title = 'Difference = {}'.format( str( chiSq_s2 ) ) )
    return chiSq, chiSq_s2

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

  def evaluate( self, test_values, plot = False ):
    if type( test_values ) is not np.ndarray:
        raise TypeError
    self.potential_file.write_with_parameters( test_values )
    ran_okay = self.training_set.run()
    if ran_okay:
        ff_forces = self.training_set.new_forces.T
        ff_dipoles = self.training_set.new_dipoles.T
        ff_stresses = self.training_set.new_stresses.T
        self.times_called += 1

        chiSq = {}
        chiSq_s2 = {}
        chiSq[ 'forces' ], chiSq_s2[ 'forces' ] = chi_squared( self.ai_forces, ff_forces, plot, 'forces-errors' )
        chiSq[ 'dipoles' ], chiSq_s2[ 'dipoles' ]  = chi_squared( self.ai_dipoles, ff_dipoles, plot, 'dipoles-errors' )
        chiSq[ 'stresses' ], chiSq_s2[ 'stresses' ] = chi_squared( self.ai_stresses, ff_stresses, plot, 'stresses-errors' )
        factorTot = sum( self.scaling.values() )
        totalChi = sum( [ self.scaling[ k ] * chiSq[ k ] for k in chiSq.keys() ] ) / factorTot
        totalChi_s2 = sum( [ self.scaling[ k ] * chiSq_s2[ k ] for k in chiSq_s2.keys() ] ) / factorTot
     
        output( '\nIteration: ' + str( self.times_called ) + '\n' )
        output( '                          \t' + 'Divided by AI' + '\t' + 'Divided by S' + '\n' )
        output( 'Forces chi sq:            \t' + fmt.format(chiSq[ 'forces' ]) + '\t' + fmt.format(chiSq_s2[ 'forces' ]) + '\n')
        output( 'Dipoles chi sq:           \t' + fmt.format(chiSq[ 'dipoles' ]) + '\t' + fmt.format(chiSq_s2[ 'dipoles' ]) + '\n')
        output( 'Stresses chi sq:          \t' + fmt.format(chiSq[ 'stresses' ]) + '\t' + fmt.format(chiSq_s2[ 'stresses' ]) + '\n')
        output( 'Total chi sq (no factors):\t ' + fmt.format( np.mean( list( chiSq.values() ) ) ) 
                + '\t' + fmt.format( np.mean( list( chiSq_s2.values() ) ) ) + '\n' )
        output('Total chi sq:              \t' + fmt.format(totalChi) + '\t' + fmt.format(totalChi_s2) + '\n')
        output('\n')
    else:
        totalChi = 1E10
        totalChi_s2 = 1E10
        self.times_called += 1
        output( '\nIteration: ' + str( self.times_called ) + '\n' )
        output('Error: likely due to unphysical parameter value\n') 

    return totalChi_s2
