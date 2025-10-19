import pickle
import torch
import numpy as np
import math
from scipy.stats import expon, uniform, multinomial
import scipy.integrate
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

class HawkesProcessGenerator:
    def __init__(self, mu, alpha, beta, T):
        self.mu = mu        # Background intensity
        self.alpha = alpha  # Excitation parameter
        self.beta = beta    # Decay parameter
        self.T = T          # Time horizon
        self.events = []    # List to store event times
        self.dataset_root = 'data/synth'

    def intensity(self, t, points_hawkes):
        mu = self.mu
        alpha = self.alpha
        beta = self.beta
        # Calculate the intensity function at time t
        points_hawkes = np.array(points_hawkes)
        intensity = 0
        history = points_hawkes[points_hawkes < t]
        intensity = mu + np.sum(alpha * np.exp(-beta * (t-history)))
        return intensity
    
    def hawkes_simulation(self): 
        t = 0
        jump = 0
        points_hawkes = []
        while(t<self.T):
            intensity_sup = self.intensity(t, points_hawkes) + jump
            r = expon.rvs(scale=1/intensity_sup) #scale=1/lamda
            t += r
            d = uniform.rvs(loc=0,scale=1)
            if (t < self.T) & (d * intensity_sup <= self.intensity(t, points_hawkes)):
                points_hawkes.append(t)
                jump = self.alpha
            else:
                jump = 0
        return np.array(points_hawkes)
    
    def nll_hawkes(self, points_hawkes): # fast: the complexity is O(N)
        mu = self.mu
        alpha = self.alpha
        beta = self.beta
        T = self.T
        
        time_difference = T - points_hawkes #T-t0...T-t(n-1) 
        time_exponential = np.exp(-beta * time_difference) - 1
        second_sum=alpha / beta * sum(time_exponential)
        
        R = np.zeros(len(points_hawkes))
        for i in range(1, len(points_hawkes)):
            R[i] = np.exp(-beta * (points_hawkes[i]-points_hawkes[i-1])) * (1 + R[i-1])
        first_sum = sum(np.log(mu+alpha*R))  #left-continuous
        
        logl = first_sum - mu * T + second_sum
        return -logl
    
    def plot_hawkes(self, points_hawkes):
        T = self.T
        plt.figure(1,figsize=(6,2))
        plt.subplot(111)
        plt.plot(np.arange(0,T,0.1),[self.intensity(t, points_hawkes) for t in np.arange(0,T,0.1)],'r-',lw=1,alpha=0.6)
        plt.plot(points_hawkes, np.zeros(len(points_hawkes)), linestyle='None', marker='|', color='b', markersize=10,label='points')
        plt.title('Hawkes Process')
        plt.ylim(-0.5,11)
        plt.legend(loc='best',frameon=0)
        plt.savefig('HawkesProcess.png', dpi=300)

    def dataset_pkl(self, dataset_size=100):
        whole_data = {}
        sequences = []
        for i in range(dataset_size):
            one_data = {}
            arrival_times = self.hawkes_simulation()
            nll = self.nll_hawkes(arrival_times)
            one_data['arrival_times'] = arrival_times.tolist()
            one_data['t_start'] = 0.0
            one_data['t_end'] = self.T
            one_data['nll'] = nll
            sequences.append(one_data)
        whole_data['sequences'] = sequences
        torch.save(whole_data, f"{self.dataset_root}/hawkes_{self.mu}_{self.alpha}_{self.beta}_{self.T}.pkl")

        print(f"Dataset saved to {self.dataset_root}/hawkes_{self.mu}_{self.alpha}_{self.beta}_{self.T}.pkl")
        print(f"Number of sequences: {len(sequences)}")
        print(f"mu: {self.mu}, alpha: {self.alpha}, beta: {self.beta}, T: {self.T}")

class InhomogeneousPoissonGenerator:
    def __init__(self, intensity_func=None, upper_bound=None, T=None):
        self.T = T
        self.intensity_func = intensity_func
        self.upper_bound = upper_bound
        self.dataset_root = 'data/synth'

    def intensity(self, t):
        return self.intensity_func(t) if self.intensity_func else 1.0

    def inhomo_poisson_simulation(self):
        t = 0
        T = self.T
        UB = self.upper_bound
        intensity = self.intensity
        points_inhomo = []
        while t < T:
            r = expon.rvs(scale=1/UB)
            t += r
            d = uniform.rvs(loc=0, scale=1)
            if (t<T) & (d * UB <= intensity(t)):
                points_inhomo.append(t)

        return np.array(points_inhomo)

    def nll_inhomo_poisson(self, points_inhomo):
        intensity = self.intensity
        T = self.T
        sum_log_intensity = sum([np.log(intensity(t)) for t in points_inhomo])
        integrate = scipy.integrate.quad(intensity, 0, T)[0]
        return integrate - sum_log_intensity

    def plot_inhomo_poisson(self, points_inhomo):
        T = self.T
        plt.figure(1, figsize=(6, 2))
        plt.subplot(111)
        plt.plot(np.arange(0, T, 0.1), [self.intensity(t) for t in np.arange(0, T, 0.1)], 'r-', lw=1, alpha=0.6)
        plt.plot(points_inhomo, np.zeros(len(points_inhomo)), linestyle='None', marker='|', color='b', markersize=10, label='points')
        plt.title('Inhomogeneous Poisson Process')
        plt.ylim(-0.5, 11)
        plt.legend(loc='best', frameon=0)
        plt.savefig('InhomogeneousPoissonProcess.png', dpi=300)

    def dataset_pkl(self, dataset_size=100):
        whole_data = {}
        sequences = []
        for i in range(dataset_size):
            one_data = {}
            arrival_times = self.inhomo_poisson_simulation()
            nll = self.nll_inhomo_poisson(arrival_times)
            one_data['arrival_times'] = arrival_times.tolist()
            one_data['t_start'] = 0.0
            one_data['t_end'] = self.T
            one_data['nll'] = nll
            sequences.append(one_data)
        whole_data['sequences'] = sequences
        torch.save(whole_data, f"{self.dataset_root}/inhomogeneous_poisson_{self.upper_bound}.pkl")

        print(f"Dataset saved to {self.dataset_root}/inhomogeneous_poisson_{self.upper_bound}.pkl")
        print(f"Number of sequences: {len(sequences)}")
        print(f"upper_bound: {self.upper_bound}, T: {self.T}")

class MultiHawkesProcessGenerator:
    def __init__(self, mu, alpha, beta, T):
        self.mu = np.array(mu)       
        self.alpha = np.array(alpha) 
        self.beta = np.array(beta)  
        self.T = T                   
        self.M = len(mu)             
        self.dataset_root = 'data/synth'
        
    def intensity(self, t, m, points_hawkes):
        mu_m = self.mu[m]
        intensity_value = mu_m
        
        for n in range(self.M):
            history = np.array(points_hawkes[n])
            history = history[history < t]
            if len(history) > 0:
                influence = np.sum(self.alpha[m][n] * np.exp(-self.beta[m][n] * (t - history)))
                intensity_value += influence
                
        return intensity_value
    
    def total_intensity(self, t, points_hawkes):
        return sum(self.intensity(t, m, points_hawkes) for m in range(self.M))
    
    def multi_hawkes_simulation(self):
        t = 0
        jumps = np.zeros(self.M)
        points_hawkes = [[] for _ in range(self.M)]
        
        while t < self.T:
            intensity_sup = sum(self.intensity(t, m, points_hawkes) + jumps[m] for m in range(self.M))
            r = expon.rvs(scale=1/intensity_sup)
            t += r
            
            if t >= self.T:
                break
                
            D = uniform.rvs(loc=0, scale=1)
            current_total_intensity = self.total_intensity(t, points_hawkes)
            
            if D * intensity_sup <= current_total_intensity:
                probs = [self.intensity(t, m, points_hawkes) / current_total_intensity for m in range(self.M)]
                k = list(multinomial.rvs(1, probs)).index(1)
                points_hawkes[k].append(t)
                for m in range(self.M):
                    jumps[m] = self.alpha[m][k]
            else:
                jumps = np.zeros(self.M)
        
        for m in range(self.M):
            points_hawkes[m] = np.array(points_hawkes[m])
            
        return points_hawkes
    
    def nll_multi_hawkes(self, points_hawkes):
        log_likelihood = 0

        for m in range(self.M):
            for t in points_hawkes[m]:
                log_likelihood += np.log(self.intensity(t, m, points_hawkes))
        
        for m in range(self.M):
            log_likelihood -= self.mu[m] * self.T
            for n in range(self.M):
                for t in points_hawkes[n]:
                    time_remaining = self.T - t
                    if time_remaining > 0:
                        log_likelihood -= (self.alpha[m][n] / self.beta[m][n]) * (1 - np.exp(-self.beta[m][n] * time_remaining))
        
        return -log_likelihood
    
    def plot_multi_hawkes(self, points_hawkes, time_points=None):
        if time_points is None:
            time_points = np.linspace(0, self.T, 1000)
        
        plt.figure(figsize=(10, 2 * self.M))
        
        for m in range(self.M):
            plt.subplot(self.M, 1, m+1)

            intensities = [self.intensity(t, m, points_hawkes) for t in time_points]
            plt.plot(time_points, intensities, 'r-', lw=1, alpha=0.6)

            plt.plot(points_hawkes[m], np.zeros(len(points_hawkes[m])), 
                     linestyle='None', marker='|', color='b', markersize=10)
            
            plt.title(f'Dimension {m+1}')
            plt.ylim(-0.5, max(intensities) * 1.1)
            
            if m == self.M - 1:
                plt.xlabel('Time')
            
            plt.ylabel('Intensity')
        
        plt.tight_layout()
        plt.savefig('MultiHawkesProcess.png', dpi=300)
        plt.show()
        
    def dataset_pkl(self, dataset_size=100):
        whole_data = {}
        sequences = []
        
        for i in range(dataset_size):
            one_data = {}
            points_hawkes = self.multi_hawkes_simulation()
            
            arrival_times = []
            event_types = []
            
            for m in range(self.M):
                for t in points_hawkes[m]:
                    arrival_times.append(float(t))
                    event_types.append(int(m))
            
            sorted_indices = np.argsort(arrival_times)
            arrival_times = [arrival_times[i] for i in sorted_indices]
            event_types = [event_types[i] for i in sorted_indices]
            
            nll = self.nll_multi_hawkes(points_hawkes)
            one_data['arrival_times'] = arrival_times
            one_data['marks'] = event_types
            one_data['t_start'] = 0.0
            one_data['t_end'] = self.T
            one_data['nll'] = float(nll)
            
            sequences.append(one_data)
        
        whole_data['sequences'] = sequences
        whole_data['num_marks'] = self.M
        
        mu_str = "_".join(map(str, self.mu))
        torch.save(whole_data, f"{self.dataset_root}/multi_hawkes_{(mu[0], mu[1])}_{(alpha[0], alpha[1])}_{(beta[0], beta[1])}_{self.T}.pkl")
        
        print(f"Dataset saved to {self.dataset_root}/multi_hawkes_{mu_str}_{self.T}.pkl")
        print(f"Number of sequences: {len(sequences)}")
        print(f"mu: {self.mu}, alpha: {self.alpha}, beta: {self.beta}, T: {self.T}")

class SelfCorrectingProcessGenerator:
    def __init__(self, mu, alpha, T):
        self.mu = mu
        self.alpha = alpha
        self.T = T
        self.dataset_root = 'data/synth'
    
    def intensity(self, t, points_history):
        points_history = np.array(points_history)
        N_t = np.sum(points_history < t)
        intensity = np.exp(self.mu * t - self.alpha * N_t)
        return intensity
    
    def self_correcting_simulation(self):
        t = 0
        points_history = []
        
        while t < self.T:
            current_intensity = self.intensity(t, points_history)
            delta = 1.0  # time step size
            intensity_sup = np.exp(self.mu * (t + delta) - self.alpha * len(points_history)) * 5
            
            assert intensity_sup > current_intensity, f"Intensity upper bound: {intensity_sup:.4f} should be greater than current intensity: {current_intensity:.4f}"

            r = expon.rvs(scale=1/intensity_sup)
            t += r
            
            if t >= self.T:
                break
            
            d = uniform.rvs(loc=0, scale=1)
            
            true_intensity = self.intensity(t, points_history)
           

            if d * intensity_sup <= true_intensity:
                points_history.append(t)
            
        return np.array(points_history)
    
    def nll_self_correcting(self, points_history):

        points_history = np.array(points_history)
        log_intensity_sum = 0
        for i, t in enumerate(points_history):
            history_before = points_history[:i]
            intensity_value = self.intensity(t, history_before)
            log_intensity_sum += np.log(intensity_value)

        integral_value = 0

        last_time = 0
        for i, t in enumerate(points_history):
            if t > self.T:
                break

            N_last = i
            # Integral from last_time to t
            # ∫exp(μs - αN)ds from last_time to t
            # = (exp(μt) - exp(μ*last_time)) * exp(-αN) / μ
            integral_value += (np.exp(self.mu * t) - np.exp(self.mu * last_time)) * np.exp(-self.alpha * N_last) / self.mu
            
            last_time = t
        
        N_last = len(points_history[points_history <= last_time])
        integral_value += (np.exp(self.mu * self.T) - np.exp(self.mu * last_time)) * np.exp(-self.alpha * N_last) / self.mu
        
        return -(log_intensity_sum - integral_value)
    
    def plot_self_correcting(self, points_history, save_name='SelfCorrectingProcess.png'):
        
        T = self.T
        t_points = np.linspace(0, T, 1000)
        intensities = [self.intensity(t, points_history) for t in t_points]
        
        plt.figure(figsize=(10, 4))
        plt.subplot(111)
        plt.plot(t_points, intensities, 'r-', lw=1, alpha=0.6)
        plt.plot(points_history, np.zeros(len(points_history)), 
                 linestyle='None', marker='|', color='b', markersize=10, label='Events')
        plt.title('Self-Correcting Process')
        plt.xlabel('Time')
        plt.ylabel('Intensity')
        plt.ylim(-0.5, max(1.1*max(intensities), 1))
        plt.legend(loc='best', frameon=0)
        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
    
    def dataset_pkl(self, dataset_size=100):
        
        whole_data = {}
        sequences = []
        
        for i in range(dataset_size):
            one_data = {}
            arrival_times = self.self_correcting_simulation()
            nll = self.nll_self_correcting(arrival_times)
            # self.plot_self_correcting(arrival_times, save_name=f'gen_plot/self_correcting_{i}.png')

            one_data['arrival_times'] = arrival_times.tolist()
            one_data['t_start'] = 0.0
            one_data['t_end'] = self.T
            one_data['nll'] = float(nll)
            
            sequences.append(one_data)
        
        whole_data['sequences'] = sequences
        
        filename = f"{self.dataset_root}/self_correcting_{self.mu}_{self.alpha}_{self.T}.pkl"
        torch.save(whole_data, filename)
        
        print(f"Dataset saved to {filename}")
        print(f"Number of sequences: {len(sequences)}")
        print(f"mu: {self.mu}, alpha: {self.alpha}, T: {self.T}")




if __name__ == "__main__":
    #### Hawkes Process
    # mu = 2.5
    # alpha = 1.0
    # beta = 2.0
    # T = 100.0

    # generator = HawkesProcessGenerator(mu, alpha, beta, T)
    # generator.dataset_pkl(dataset_size=100)


    #### Inhomogeneous Poisson Process
    # intensity = lambda t: 5*(1+np.sin(2*np.pi/100*t))
    # generator = InhomogeneousPoissonGenerator(intensity_func=intensity, upper_bound=10, T=100.0)
    # generator.dataset_pkl(dataset_size=100)

    # ### Multi Hawkes Process
    # mu = [0.5, 0.5]
    # alpha = [[1, 0.5],[0.1, 1]]
    # beta = [[2, 2],[2, 2]]
    # T = 100
    # generator = MultiHawkesProcessGenerator(mu, alpha, beta, T)
    # generator.dataset_pkl(dataset_size=100)

    #3. Instantiate generator
    mu = [0.5, 0.5]
    alpha = [[1, 0.5],[0.1, 1]]
    beta = [[2, 2],[2, 2]]
    T = 100
    generator = MultiHawkesProcessGenerator(mu, alpha, beta, T)

    # 4. Record start time
    start_time = time.time()

    # 5. Call simulation once
    points_hawkes = generator.multi_hawkes_simulation()

    # 6. Record end time
    end_time = time.time()

    # 7. Calculate and print duration
    duration = end_time - start_time
    print(f"Single call to multi_hawkes_simulation took: {duration:.4f} seconds")
