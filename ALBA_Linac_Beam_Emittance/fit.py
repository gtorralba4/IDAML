import numpy as np
import matplotlib.pyplot as plt
import ultranest
from ultranest.plot import PredictionBand
from ultranest.plot import cornerplot
from ultranest import stepsampler
import time
import uncertainties.unumpy as unumpy
from astropy import units as u
def wrapped_obj_function(x_exp1, y_exp1,y_err):
    def objective_function(parameters):
        a, b, c = parameters
        y_prime = a*(x_exp1-b)**2+c
        
        log1 = -0.5 * np.sum(((y_exp1 - y_prime) / (y_err))**2)
        
        logL = log1
        return logL
    return objective_function

def prior_transform1(cube):
    # the argument, cube, consists of values from 0 to 1
    # we have to convert them to physical scales
    
    params = np.empty_like(cube)
    #a
    params[0] = cube[0]*1.5
    #b
    params[1] = cube[1]*1.6+0.4
    #c
    params[2] = cube[2]*0.15
    return params

def main():
    calib_factorF = 387.65*u.MeV/u.A/u.m**2 #MeV/A/m^2
    A = 0.7058*u.MeV/u.A
    A_0 = 0.2448*u.MeV
    #m
    L = 0.126
    #m
    d = 1.235
    I1 = 113.1*u.A
    I2 = 99.3*u.A
    E1 = A*I1+A_0
    E2 = A*I2+A_0
    param_1caustic = ["a","b","c"]  
    data = np.loadtxt('data_simple.txt',skiprows=4)
    X = data[:, 0]*u.A
    #beam size in mm^2
    Y = data[:, 1]**2
    #10% error on Y sigma = sqrt(y)->d(sigma^2) = 5%*2*sigma^2
    sigma_y = 2 * Y * 0.05 
    lines = len(X)
    X1 = int(np.ceil(len(X)/2))
    X2 = int(np.ceil(len(X)))
    sqrtIFE1 = np.sqrt(abs(X[0:X1])*calib_factorF/E1)
    sqrtIFE2 = np.sqrt(abs(X[X1:X2])*calib_factorF/E2)
    k = np.zeros(len(X))*sqrtIFE1.unit
    Q_strength = np.zeros(len(X))*k.unit
    k[0:X1] = np.sign(X[0:X1])*sqrtIFE1
    k[X1:X2] = np.sign(X[X1:X2])*sqrtIFE2
    Q_strength[0:X1] = k[0:X1]*np.sin(k[0:X1].value*L*u.rad)
    Q_strength[X1:X2] = k[X1:X2]*np.sin(k[X1:X2].value*L*u.rad)
    k_thin = np.zeros(len(X))
    k_thin[0:X1] = k[0:X1]**2*L
    k_thin[X1:X2] = k[X1:X2]**2*L
    #just rename to make it easier (in m^-1)
    X = Q_strength.value

    plt.rcParams["font.family"] = "Times New Roman"
    fontname = {'fontname': 'Times New Roman', 'size': 14}
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)    
    for i in range(0,len(X),6):
        model = f"fit_{i}"
        true_objective_function=wrapped_obj_function(X[i:i+6], Y[i:i+6],sigma_y[i:i+6])
        sampler = ultranest.ReactiveNestedSampler(param_1caustic, true_objective_function, prior_transform1,log_dir=f"ultranest-{model}")
        sampler.stepsampler = stepsampler.SliceSampler(nsteps=400,generate_direction=stepsampler.generate_mixture_random_direction,adaptive_nsteps='move-distance')
        result = sampler.run(viz_callback=False,min_num_live_points=400,max_ncalls=10000000)
        sampler.print_results()
        mean_params = np.mean(result['samples'], axis=0)
        std_err = np.std(result['samples'], axis=0)
        mean_params = np.append(mean_params,result['logz'])
        std_err = np.append(std_err,result['logzerr'])
  
        mean_a, mean_b, mean_c = mean_params[:3]
        y_pred = mean_a*(X[i:i+6]-mean_b)**2 + mean_c
        chi2 = np.sum(((Y[i:i+6] - y_pred) / sigma_y[i:i+6])**2)
        reduced_chi2 = chi2 / (len(X)/4 - len(param_1caustic))
        mean_params = np.append(mean_params, [chi2, reduced_chi2])
        std_err = np.append(std_err, [0, 0])
        print("chi2=", chi2, "  reduced chi2=", chi2 / (6 - 3))
        print(lines/4-len(param_1caustic), "+/-",f"{np.sqrt(2*(lines/4-len(param_1caustic))):.2f}")
        np.savetxt(f'{model}_fit.dat', np.column_stack((mean_params,std_err)),delimiter=' ',fmt='%.8g')
        cornerplot(result)
 
        #Twiss parameters
        #m^2*mm^2
        A = unumpy.uarray(mean_params[0],std_err[0])/10**6
        #m*mm^2
        B = unumpy.uarray(mean_params[1],std_err[1])/10**6
        #mm^2
        C = unumpy.uarray(mean_params[2],std_err[2])/10**6

        alpha = unumpy.sqrt(A/C)*(B+1/d)
        beta = unumpy.sqrt(A/C)
        epsilon = unumpy.sqrt(A*C)/d**2
        gamma = (1+alpha**2)/beta
        print("alpha= ",alpha)
        print("beta= ",beta)
        print("gamma= ",gamma)
        print("emittance= ",epsilon)
        # proton rest mass energy
        m_e = 0.511  * u.MeV
        
        # relativistic betagamma for each energy
        betagamma_1 = np.sqrt(((E1+m_e)/m_e)**2 - 1)   # for fits 0, 12 (horizontal)
        betagamma_2 = np.sqrt(((E2+m_e)/m_e)**2 - 1)   # for fits 6, 18 (vertical)
        if i == 0:
            print("normalised emmittance= ",epsilon*betagamma_1)
        else:
            print("normalised emmittance= ",epsilon*betagamma_2)
        print("-"*40)
        #------------------------------------------------------------------------------------------------------------------------------------
        #graph

        fig,ax = plt.subplots(figsize=(10, 6))
        
        band = PredictionBand(X[i:i+6])
        for a,b,c in sampler.results['samples'][:,:3]:
            Y_model = a*(X[i:i+6]-b)**2+c
            band.add(Y_model)
            
        band.line(color='k',label='Model')
        band.shade(color='k', alpha=0.3,label="1 sigma quantile")
        
        ax.plot(X[i:i+6], Y[i:i+6],
                     marker='o', ls=' ', color='darkorange', 
                     markerfacecolor='none', markersize=10,
                     label='data')
        
        ax.set_xlabel('Integrated quadrupole strength [m$^{-1}$]', fontdict=fontname)
        ax.set_ylabel('Beam size [mm$^2$]', fontdict=fontname)
        ax.legend(loc='upper center', fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        
        png_name = f'{model}_fit.png'
        fig.savefig(png_name, dpi=150)
        plt.show()
        plt.close()
        
        fig2,ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(k_thin[i:i+6], Y[i:i+6],marker='o', ls=' ', color='darkorange', 
                 markerfacecolor='none', markersize=10,
                 label='Integrated quadrupole intensity')
        ax2.plot(X[i:i+6], Y[i:i+6],marker='^', ls=' ', color='green', 
                 markerfacecolor='none', markersize=10,
                 label='Thin lens approximation')
        ax2.set_xlabel('Integrated quadrupole strength [m$^{-1}$]', fontdict=fontname)
        ax2.set_ylabel('Beam size [mm$^2$]', fontdict=fontname)
        ax2.legend(loc='upper center', fontsize=10)
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        
        png_name = f'{model}_thinlens.png'
        fig2.savefig(png_name, dpi=150)
        plt.show()
        plt.close()
        
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
