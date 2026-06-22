import numpy as np
from scipy.linalg import expm

class Transmon:

    def __init__(self,E_C,E_J,N):

        self.E_C=E_C
        self.E_J=E_J
        self.N=N   

        self.phi = np.linspace(-np.pi, np.pi, N, endpoint=False) #vaiheoperaattori diskretisoituna välillä [-pi,pi] 1000 pisteeseen
        self.delta_phi = self.phi[1]-self.phi[0] #vaiheoperaattorin diskresitoitu askelväli

        M = np.eye(N,k=1)+np.eye(N,k=-1)-2*np.eye(N) #Muodostetaan numero-operaattorin neliön matriisiesitys differenssimenetelmällä
        M[0,-1]=1
        M[-1,0]=1
        
        n_squared = -M/(self.delta_phi**2)

        self.H_0 = 4*E_C*n_squared-E_J*np.diag(np.cos(self.phi)) #Hamiltonin operaattori

        energies, eigenstates = np.linalg.eigh(self.H_0) # Transmonin ominaisenergiat ja -tilat

        self.energies = energies
        self.eigenstates = eigenstates
        self.frequency = energies[1]-energies[0]

    def H_D_phibasis(self): #Ajettu Hamiltonin operaattori phi-kannassa (n_g:n muutos)

        M=np.eye(self.N,k=1)-np.eye(self.N,k=-1)

        def n_g(A): return A/(8*self.E_C)
        def offset_term(A): return 1j * n_g(A) * M / self.delta_phi

        def H_t(A): return self.H_0 + 4 * self.E_C * offset_term(A)

        return H_t
    
    def H_D_eigbasis(self,dim): #Ajettu Hamiltonin operaattori ominaiskannassa
 
        H_0 = np.diag(self.energies[0:dim]-self.energies[0]) #Diagonaalimatriisi ominaisenergioista (lukumäärä ensimmäiset dim)

        M = np.zeros((dim, dim))
        vals = np.sqrt(np.arange(1, dim))
        M[np.arange(dim-1), np.arange(1, dim)] = vals      
        M[np.arange(1, dim), np.arange(dim-1)] = vals

        def H_t(A): return H_0 + A*M

        return H_t
    

class time_evolution:

    def __init__(self,generator):
        self.generator = generator
    
    def U(self, A, dt): return expm(-1j * self.generator(A) * dt) #Aikaevoluutio-operaattori

    def U_floquet(self, A_vals, dt): #Floquet-operaattori

        U_f = self.U(0, 0) #Aikakehitysoperaattori yhdelle Floquet-jaksolle
        for A in A_vals: 
            U_f = self.U(A, dt) @ U_f
        return U_f