import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

class Pulse:

    def __init__(self,frequency,f_supp,pulse_funcs,pulse_times,time_unit=1,drag=False,offset_input=None,second_frequency=None):

        #Tarkistetaan, että on annettu sama määrä pulssifunktiota ja -aikoja
        if len(pulse_times)!=len(pulse_funcs):
            raise Exception("Pulse function and time interval arrays must have the same size") 

        self.time_unit = time_unit
        self.frequency = frequency
        self.f_supp = f_supp

        #Määritellään ajan arvot eri pulssin osille
        self.piecewise_time_vals=[(np.arange(time)*time_unit) for time in pulse_times]

        #Ajan arvot koko pulssin ajalle
        self.time_vals = np.arange(np.sum(pulse_times))*time_unit

        #Pulssin amplitudi määriteltynä kolmessa osassa
        self.piecewise_envelope_vals=[np.array([pulse_funcs[i](t) for t in self.piecewise_time_vals[i]]) for i in range(0,len(self.piecewise_time_vals))]

        #Pulssin amplitudi
        self.envelope = np.concatenate(self.piecewise_envelope_vals)

        if drag:
            #Derivaatta amplitudifunktiosta
            self.envelope_derivative = np.gradient(self.envelope,time_unit)

        if offset_input == None:
            #Määritellään signaalin vaihe siten, että amplitudi on nolla toisen jakson alussa
            #offset = frequency*pulse_times[0]
            self.offset=0
        else: self.offset = offset_input

        #Aikapisteiden määrä
        N=len(self.time_vals)

        if second_frequency==None:
            signal=np.zeros(N)
            cosine_signal=np.zeros(N)
            self.drag_component=np.zeros(N)

            if callable (frequency):
                #Määritellään signaali sini- ja kosinifunktioiden avulla, käyttäen taajuutta amplitudin funktiona
                signal[0] = np.sin(frequency(self.envelope[0])*self.time_vals[0]-self.offset)
                cosine_signal[0] = np.cos(frequency(self.envelope[0])*self.time_vals[0]-self.offset)

                phase=self.offset

                for i in range(1,N):
                    phase=frequency(self.envelope[i-1])*self.time_vals[i-1]-frequency(self.envelope[i])*self.time_vals[i-1]+phase

                    signal[i] = np.sin(frequency(self.envelope[i])*self.time_vals[i]+phase)
                    if drag:
                        #DRAG-komponentti pulssille
                        cosine_signal[i] = np.cos(frequency(self.envelope[i])*self.time_vals[i]+phase)
                        self.drag_component[i] = self.envelope_derivative[i] * cosine_signal[i] / (f_supp-frequency(self.envelope[i]))

            else:
                #Määritellään signaali sini- ja kosinifunktioiden avulla, käyttäen taajuutta amplitudin funktiona
                signal = np.sin(frequency*self.time_vals-self.offset)
                if drag:
                    #DRAG-komponentti pulssille
                    cosine_signal = np.cos(frequency*self.time_vals-self.offset)
                    self.drag_component = self.envelope_derivative * cosine_signal / (f_supp-frequency)

        else:      
            signal=np.zeros(N)
            cosine_signal_1=np.zeros(N)
            cosine_signal_2=np.zeros(N)
            self.drag_component=np.zeros(N)

            if callable(frequency):
                #Määritellään signaalin vaihe siten, että amplitudi on nolla toisen jakson alussa
                #offset=0

                #Määritellään signaali sini- ja kosinifunktioiden avulla, käyttäen taajuutta amplitudin funktiona
                signal[0] = np.sin(frequency(self.envelope[0])*self.time_vals[0]-self.offset)-np.sin(second_frequency(self.envelope[0])*self.time_vals[0]-self.offset)
                cosine_signal_1[0] = np.cos(frequency(self.envelope[0])*self.time_vals[0]-self.offset)
                cosine_signal_2[0] = np.cos(second_frequency(self.envelope[0])*self.time_vals[0]-self.offset)

                phase1=self.offset
                phase2=self.offset

                for i in range(1,N):
                    phase1=frequency(self.envelope[i-1])*self.time_vals[i-1]-frequency(self.envelope[i])*self.time_vals[i-1]+phase1
                    phase2=frequency(self.envelope[i-1])*self.time_vals[i-1]-frequency(self.envelope[i])*self.time_vals[i-1]+phase2

                    signal[i] = np.sin(frequency(self.envelope[i])*self.time_vals[i]+phase1)-np.sin(second_frequency(self.envelope[i])*self.time_vals[i]+phase2)

                    if drag:
                        cosine_signal_1[i] = np.cos(frequency(self.envelope[i])*self.time_vals[i]+phase1)
                        cosine_signal_2[i] = np.cos(second_frequency(self.envelope[i])*self.time_vals[i]+phase2)

                        #DRAG-komponentti pulssille
                        #self.drag_component[i] = self.envelope_derivative[i] * (2*cosine_signal_1[i] / (f_supp-2*frequency+second_frequency) + cosine_signal_2[i] / (f_supp-2*frequency+second_frequency))
                        self.drag_component[i] = self.envelope_derivative[i] * (cosine_signal_1[i] / (f_supp[0]-frequency[i]) + cosine_signal_2[i] / (f_supp[1]-second_frequency[i]))

            else:
                #Määritellään signaalin vaihe siten, että amplitudi on nolla toisen jakson alussa
                #offset=0

                #Määritellään signaali sini- ja kosinifunktioiden avulla, käyttäen taajuutta amplitudin funktiona
                signal = np.sin(frequency*self.time_vals-self.offset)-np.sin(second_frequency*self.time_vals-self.offset)
                if drag:
                    #DRAG-komponentti pulssille
                    cosine_signal_1 = np.cos(frequency*self.time_vals-self.offset)
                    cosine_signal_2 = np.cos(second_frequency*self.time_vals-self.offset)
                    #self.drag_component = self.envelope_derivative * (2*cosine_signal_1 / (f_supp-2*frequency+second_frequency) + cosine_signal_2 / (f_supp-2*frequency+second_frequency))
                    self.drag_component = self.envelope_derivative * (cosine_signal_1 / (f_supp[0]-frequency) - cosine_signal_2 / (f_supp[1]-second_frequency))

        #Signaalin arvot moduloituna amplitudilla
        if drag:
            self.raw = self.envelope*signal + self.drag_component
        else:
            self.raw = self.envelope*signal  

    def second_drag(self,f_supp_second):
        envelope_second_derivative = np.gradient(self.envelope_derivative,self.time_unit)/(self.f_supp-self.frequency)

        sine_signal = np.sin(self.frequency*self.time_vals-self.offset)
        cosine_signal = np.cos(self.frequency*self.time_vals-self.offset)

        drag_component = (self.envelope_derivative*cosine_signal - envelope_second_derivative*sine_signal)/ (f_supp_second-self.frequency)
        self.raw = self.raw + drag_component

        
def sim_3_part_pulses(f_d,f_supp,A,evolution,dim,f_rabi,t_ramp=5,K=20,f_2=None,U_f=0,drag=False,verbose=False,use_avg=False,s2=None):
    #Floquet-jakson pituus
    if f_2 == None:
        if callable(f_d):
            T_floquet = 2*np.pi/(f_d(1))
        else:
            T_floquet = 2*np.pi/(f_d)
    else:
        if callable(f_d):
            T_floquet = 10*np.pi/(2*f_d(1)-f_2(1)) 
        else:
            T_floquet = 10*np.pi/(2*f_d-f_2) 

    pulse_shape=[gaussian(t_ramp*2,t_ramp/2),one,gaussian_opp(0,t_ramp/2)]

    dt = T_floquet/K #Diskretoidaan aika-avaruus siten, että Floquet-jakson pituus on K aikapistettä

    N_edge=int(t_ramp*2/dt) #Määritellään reunojen pituudeksi 2x rampin pituus, jotta amplitudi menee lähemmäs nollaa
    N_pulse=int(2*np.pi/(f_rabi*dt)) #Määritellään pulssin pituus arvatun Rabi-taajuuden perusteella

    #Kokonainen pulssi rampin aikakehityksen sekä Floquet-operaattorin laskemista varten
    pulse = Pulse(f_d,f_supp,pulse_shape,[N_edge,N_pulse,N_edge],time_unit=dt,second_frequency=f_2,drag=drag)
    if s2 != None:
        pulse.second_drag(s2)

    #Plotataan ajopulssi mikäli pyydetty
    if verbose:
        plt.figure()
        plt.plot(pulse.time_vals,pulse.raw)
        plt.xlabel("Aika (ns)")
        plt.ylabel("Normalisoitu amplitudi")
        plt.title("Ajopulssi")

        freqs = np.fft.rfftfreq(len(pulse.time_vals), d=dt)
        fourier = np.fft.rfft(pulse.raw)
        
        plt.figure()
        plt.semilogy(freqs*2*np.pi, np.abs(fourier)/np.max(np.abs(fourier)))
        plt.xlabel("Kulmataajuus (10^9 rad/s)")
        plt.ylabel("Normalisoitu amplitudi")
        plt.show()
    

    #Alustetaan kubitin tila perustilaan
    psi_eig_accum = np.zeros(dim, dtype=complex)
    psi_eig_accum[0] = 1

    #Lopulliset todennäköisyydet eripituisten pulssien jälkeen
    last_vals=[]

    for i in range(0, N_edge): #Aikakehitys nousevalle rampille
        psi_eig_accum = evolution.U(A*pulse.raw[i], dt) @ psi_eig_accum

    #Simulaatio pulsseille
    if U_f == 0:
        #Pulssien pituudet. Oletuksena, että rampin pituus on puolet simuloidusta reunojen pituuksista
        pulse_times=(np.arange(0,N_pulse))*dt + 2*t_ramp
        for k in range(0,N_pulse):
            psi_eig = psi_eig_accum

            #Laskevan rampin simulaatio
            pulse = Pulse(f_d,f_supp,pulse_shape,[N_edge,k,N_edge],time_unit=dt,second_frequency=f_2,drag=drag)
            if s2 != None:
                pulse.second_drag(s2)
            for i in range(0,N_edge): #Aikakehitys laskevalle rampille
                psi_eig = evolution.U(A*pulse.raw[N_edge+k+i], dt) @ psi_eig

            last_vals.append(np.abs(psi_eig)**2)
            psi_eig_accum = evolution.U(A*pulse.raw[N_edge+k], dt) @ psi_eig_accum #Kehitetään kubitin tilaa


    else:
        #Simuloitujen pulssien määrä
        N_sim=int(N_pulse/(K*U_f))
        #Pulssien pituudet. Oletuksena, että rampin pituus on puolet simuloidusta reunojen pituuksista
        pulse_times=np.arange(0,N_pulse)*K*U_f*dt + 2*t_ramp

        #Floquet operaattori potenssiin U_f
        U_floquet=evolution.U_floquet(A*pulse.raw[N_edge:N_edge+K],dt)
        U_floquet=np.linalg.matrix_power(U_floquet,U_f)

        for k in range(0,N_sim):
            psi_eig = psi_eig_accum

            #Laskevan rampin simulaatio
            pulse = Pulse(f_d,f_supp,pulse_shape,[N_edge,K*U_f*k,N_edge],time_unit=dt,second_frequency=f_2,drag=drag)
            for i in range(0,N_edge): #Aikakehitys laskevalle rampille
                psi_eig = evolution.U(A*pulse.raw[N_edge+K*U_f*k+i], dt) @ psi_eig
            last_vals.append(np.abs(psi_eig)**2)
            psi_eig_accum = U_floquet @ psi_eig_accum #Kehitetään kubitin tilaa U_f Floquet-jaksolla

    #e- ja f-tasojen lopulliset todennäköisyydet
    e_vals = np.array([a[1] for a in last_vals])
    f_vals = np.array([a[2] for a in last_vals])

    #Pulssin pituus määritettynä maksimiarvon sijainnin perusteella
    index = np.argmax(e_vals)
    T_pulse = pulse_times[index]
    
    if use_avg:
        #e-tason keskiarvo
        e_avg=np.mean(e_vals)
        #f-tason keskiarvo
        f_avg=np.mean(f_vals)
        return (e_avg,f_avg,T_pulse,last_vals,pulse_times)
    else:
        #e-tason suurin todennäköisyys
        e_max=max(e_vals)
        #f-tason suurin todennäköisyys pulssille
        f_max=max(f_vals)
        return (e_max,f_max,T_pulse,last_vals,pulse_times)

# Funktioita pulssimuotojen muodostamiseen

def zero(t): #Nollafunktio
    return 0

def one(t): #Ykkösfunktio
    return 1

def linear(k,t_0,c): #Suora
    return lambda t: k*(t-t_0)+c

def linear_opp(k,t_0,c): #Suora
    return lambda t: k*(t_0-t)+c

def arctan(t_0,c): #Arkustangentti skaalattuna välille [-1,1], siirrettynä t_0 verran ajassa
    return lambda t: np.arctan(c*(t-t_0))/np.pi+0.5

def arctan_opp(t_0,c): #Arkustangentti -t-funktiona, siirrettynä -t_0 verran ajassa. Pulssin laskevaa reunaa varten
    return lambda t: np.arctan(c*(t_0-t))/np.pi+0.5

def gaussian(t_0,sigma): #Normalisoitu Gaussinen funktio keskihajonnalla sigma, ja keskipisteellä t_0.
    return lambda t: np.exp(-((t-t_0)**2)/(2*sigma**2))
    
def gaussian_opp(t_0,sigma): #Normalisoitu Gaussinen funktio keskihajonnalla sigma, ja keskipisteellä -t_0. Pulssin laskevaa reunaa varten
    return lambda t: np.exp(-((t+t_0)**2)/(2*sigma**2))