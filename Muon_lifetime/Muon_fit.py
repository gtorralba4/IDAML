import numpy as np
import matplotlib.pyplot as plt
import time
import ultranest
from ultranest.plot import PredictionBand
from astropy import units as u
from astropy.constants import c,hbar
import uncertainties.unumpy as unumpy



def muondecay(parameters):
    tau, = parameters
    #bins = np.linspace(min(histogram), max(histogram), int(n)+1)
    #counts,edges = np.histogram(histogram,bins=bins)
    #midpoints = (edges[:-1] + edges[1:]) / 2
    #bin_width = edges[1]-edges[0]
    #y_model = np.sum(counts) * bin_width * (1/tau) * np.exp(-midpoints/tau)
    #chi2 = -np.sum(((y_model - counts))**2)
    
    #ln(\Prod(1/tau*e^(-t_i/tau))) = Nln(1/tau)-1/tau \Sum (t_i)  
    
    #truncated exponential due to \exists t_min P(t|tau) = 1/tau*exp(-(t-tmin)/tau)
    #tmin = min(histogram)
    #logL = np.sum(-np.log(tau) - histogram/tau)
    t_min, t_max = histogram.min(), 40000
    logL = np.sum(-np.log(tau) - histogram/tau) - len(histogram) * np.log(np.exp(-t_min/tau) - np.exp(-t_max/tau))
    return logL

def prior(cube):
    # the argument, cube, consists of values from 0 to 1
    # we have to convert them to physical scales
    
    params = np.empty_like(cube)
    params[0] = cube[0]*1200+1900
    return params

def main():
    global histogram
    datafiles = ["26-03-12-13-11.data","26-03-18-12-39.data"]
    histogram = np.concatenate([np.loadtxt(f)[:, 0] for f in datafiles])
    histogram = histogram[histogram < 40000]
    np.savetxt("muon_filtered.txt",histogram,fmt='%d',newline='\n')
    param_names = ["tau"]
    sampler = ultranest.ReactiveNestedSampler(param_names, muondecay, prior,log_dir="ultranest")
    result = sampler.run(viz_callback=False,min_num_live_points=500)
    sampler.print_results()
    mean_params = np.mean(result['samples'], axis=0)
    std_err = np.std(result['samples'], axis=0)
    mean_params = np.append(mean_params,result['logz'])
    std_err = np.append(std_err,result['logzerr'])
    
    
    
    tau_obs = unumpy.uarray(mean_params[0],std_err[0])
    muon_mass = unumpy.uarray(105.6583755,0.0000023) #MeV/c^2 from https://pdglive.lbl.gov/Particle.action?node=S004&init=0
    rho_lit = unumpy.uarray(1.266,0.001) #Measurement of the underground atmospheric muon charge ratio using the MINOS Near Detector
    
    c_unc = unumpy.uarray(c.value,0)
    hbar_unc = unumpy.uarray(hbar.value,hbar.uncertainty)
    hbar_GeVs_val = hbar.to(u.GeV * u.s).value
    hbar_GeVs_unc = hbar.uncertainty / hbar.value * hbar_GeVs_val
    hbar_GeVs = unumpy.uarray(hbar_GeVs_val,hbar_GeVs_unc)
    G_Fermi = unumpy.sqrt(192*np.pi**3 / ((tau_obs*1e-9)/hbar_GeVs * (muon_mass*1e-3)**5))
    G_Fermi_SI = G_Fermi*(hbar_unc*c_unc)**3
    print("G_F = ",G_Fermi,"GeV^-2 =", G_Fermi_SI,"Jm^3")
    print("--"*30)
    tau_muon = unumpy.uarray(2.19703,0.00004)*10**3 #ns
    tau_muon_neg_medium = unumpy.uarray(2.043,0.003)*10**3 #ns
    expected_tau = tau_muon*(1+rho_lit)/(tau_muon/tau_muon_neg_medium+rho_lit)
    rho = -tau_muon/tau_muon_neg_medium *((tau_muon_neg_medium-tau_obs)/(tau_muon-tau_obs))
    print("Literature rho =", rho_lit)
    print("rho = ",rho)
    print("--"*30)
    print("expected tau = ", expected_tau,"ns")
    #maximum likelihood estimation (d/dtau(lnL = 0) -> tau_mle = mean(tau))
    print(f"Sample mean (analytical MLE): {np.mean(histogram):.1f} ns")
    print("obtained tau = ",tau_obs,"ns")
    
    
    plt.figure()
    plt.rcParams["font.family"] = "Times New Roman"
    fontname={'fontname':'Times New Roman','size':14}
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Time [ns]', fontdict=fontname)
    plt.ylabel('Counts',fontdict=fontname)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    
    counts, edges, _ = plt.hist(histogram, bins="auto", label='Data')
    bin_width = edges[1] - edges[0]
    midpoints = (edges[:-1] + edges[1:]) / 2
    
    x= midpoints
    band = PredictionBand(x)    
    for (a,) in sampler.results['samples'][:,:1]:
        y_model = len(histogram)*bin_width*1/a*np.exp(-x/a)
        band.add(y_model)

    band.line(color='k',label='Fit',zorder=5)
    #plt.xlim(0, 20000)
    #plt.ylim(-0.3,0.8)
    #plt.yscale('log')
    png_name = 'fitted_histogram.png'
    plt.legend(loc='upper right')
    plt.savefig(png_name)
    plt.show()
    plt.close()
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
