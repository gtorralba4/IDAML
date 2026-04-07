import numpy as np
import matplotlib.pyplot as plt
import ultranest
from ultranest.plot import PredictionBand
from ultranest.plot import cornerplot
from ultranest import stepsampler
import time
import uncertainties.unumpy as unumpy
from astropy import units as u
def wrapped_obj_function(x_exp,y_exp,y_err):
    def objective_function(parameters):
        a, b, c, d = parameters
        y_1 = y_exp[0:int(np.ceil(len(y_exp)/2))]
        y_2 = y_exp[int(np.ceil(len(y_exp)/2)):int(np.ceil(len(y_exp)))]
        x_1 = x_exp[0:int(np.ceil(len(x_exp)/2))]
        x_2 = x_exp[int(np.ceil(len(x_exp)/2)):int(np.ceil(len(x_exp)))]
        y_err1 = y_exp[0:int(np.ceil(len(y_err)/2))]
        y_err2 = y_exp[int(np.ceil(len(y_err)/2)):int(np.ceil(len(y_err)))]
        y_prime = a*y_1+b
        x_prime = c*x_1+d
        
        log1 = -0.5 * np.sum(((y_2 - y_prime) / (y_err2))**2)
        log2 = -0.5 * np.sum(((x_2 - x_prime) / (y_err2))**2)
        logL = log1+log2
        return logL
    return objective_function

def prior_transform1(cube):
    # the argument, cube, consists of values from 0 to 1
    # we have to convert them to physical scales
    
    params = np.empty_like(cube)
    #a
    params[0] = cube[0]*2.5-1.1
    #b
    params[1] = cube[1]*1.5-0.5
    #c
    params[2] = cube[2]*2
    #d
    params[3] = cube[3]
    
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
    param_1caustic = ["a_beamsize","b_beamsize","c_intensity","d_intensity"]  
    data = np.loadtxt('data_simple.txt',skiprows=4)
    X = data[:, 0]*u.A
    #beam size in mm^2
    Y = data[:, 1]**2
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
    #just rename to make it easier (in m^-1)
    X = Q_strength.value
    X_pos = np.concatenate((X[0:6],X[12:18]))
    Y_pos = np.concatenate((Y[0:6],Y[12:18]))
    X_neg = np.concatenate((X[6:12],X[18:24]))
    Y_neg = np.concatenate((Y[6:12],Y[18:24]))
    X = np.concatenate((X_pos,X_neg))
    Y = np.concatenate((Y_pos,Y_neg))
    #10% error on Y sigma = sqrt(y)->d(sigma^2) = 5%*2*sigma^2
    sigma_y = 2 * Y * 0.01 
    plt.rcParams["font.family"] = "Times New Roman"
    fontname = {'fontname': 'Times New Roman', 'size': 14}
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    for i in range(0,len(X),12):
        model = f"fit_{i}_energy"
        true_objective_function=wrapped_obj_function(X[i:i+12], Y[i:i+12],sigma_y[i:i+12])
        sampler = ultranest.ReactiveNestedSampler(param_1caustic, true_objective_function, prior_transform1,log_dir=f"ultranest-{model}")
        sampler.stepsampler = stepsampler.SliceSampler(nsteps=800,generate_direction=stepsampler.generate_mixture_random_direction,adaptive_nsteps='move-distance')
        result = sampler.run(viz_callback=False,min_num_live_points=400,max_ncalls=10000000)
        sampler.print_results()
        mean_params = np.mean(result['samples'], axis=0)
        std_err = np.std(result['samples'], axis=0)
        mean_params = np.append(mean_params,result['logz'])
        std_err = np.append(std_err,result['logzerr'])
  
        mean_a, mean_b, mean_c, mean_d = mean_params[:4]
        y_pred = mean_a*Y[i:i+6]+ mean_b
        x_pred = mean_c*X[i:i+6] + mean_d
        chi2 = np.sum(((Y[i+6:i+12] - y_pred) / sigma_y[i+6:i+12])**2+((X[i+6:i+12] - x_pred) / sigma_y[i+6:i+12])**2)
        reduced_chi2 = chi2 / (len(X)/2 - len(param_1caustic))
        mean_params = np.append(mean_params, [chi2, reduced_chi2])
        std_err = np.append(std_err, [0, 0])
        print("chi2=", chi2, "  reduced chi2=", chi2 / (12-len(param_1caustic)))
        print(lines/2-len(param_1caustic), "+/-",f"{np.sqrt(2*(lines/2-len(param_1caustic))):.2f}")
        np.savetxt(f'{model}_fit.dat', np.column_stack((mean_params,std_err)),delimiter=' ',fmt='%.8g')
        cornerplot(result)
        #------------------------------------------------------------------------------------------------------------------------------------
        #graph
        
        fig,ax = plt.subplots(figsize=(10, 6))
        bands = [[] for _ in range(6)]
        
        #forced to do double loops in order to plot bands and not straight lines
        for a, b, c, d in sampler.results['samples'][:, :4]:
            Y_transformed = a * Y[i:i+6] + b
            for j in range(6):
                bands[j].append((Y[i+j], Y_transformed[j]))
        X_transformed = mean_c * X[i:i+6] + mean_d
        for j in range(6):
            band = PredictionBand((X[i+j], X_transformed[j]))
            for y_pair in bands[j]:
                band.add(y_pair)
            if j==0:
                band.line(color='k',label="Model")
                band.shade(color='k', alpha=0.3,label="1 sigma quantile")
            else:
                band.line(color='k',)
                band.shade(color='k', alpha=0.3)
        
        ax.plot(X[i:i+6], Y[i:i+6],
                     marker='o', ls=' ', color='darkorange', 
                     markerfacecolor='none', markersize=10,
                     label='Dipole intensity = 113.1 A')
        ax.plot(X[i+6:i+12], Y[i+6:i+12],
                     marker='^', ls=' ', color='darkgreen', 
                     markerfacecolor='none', markersize=10,
                     label='Dipole intensity = 99.3 A')
        
        ax.set_xlabel('Integrated quadrupole strength [m$^{-1}$]', fontdict=fontname)
        ax.set_ylabel('Beam size [mm$^2$]', fontdict=fontname)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        png_name = f'{model}_fit.png'
        fig.savefig(png_name, dpi=150)
        plt.show()
        plt.close()
            
        
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
