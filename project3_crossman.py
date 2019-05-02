# -*- coding: utf-8 -*-
"""
@Title:     Physics 660 Project Three
@Author     Andrew Crossman
@Date       Mar. 22nd, 2019
"""
###############################################################################
# IMPORTS
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random as rand
###############################################################################
# HELPER FUNCTION DEFINITIONS
###############################################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Helper function for calculating the Lyapunov fit
def func(t, b):
     return np.exp(b*t)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Creates a [length X length] lattice that is aligned with the Magnetic Field (B).
#length is the size of the square lattice
def create_aligned_lattice(length,B):
    array = []
    if B>0:
        tmp = 1
    else:
        tmp = -1
    for i in list(range(0,length)):
        row = []
        for j in list(range(0,length)):
            row.append(tmp)
        array.append(row)
    return array
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Creates a random [128 X 128] lattice. Position (i,j) has value -1 or 1
#length is the size of the square lattice
def create_random_lattice(length):
    array = []
    for i in list(range(0,length)):
        row = []
        for j in list(range(0,length)):
            tmp = 2*rand.randint(0,1)-1
            row.append(tmp)
        array.append(row)
    return array
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Calculates the Energy of the entire lattice particular state with periodic boundary conditions
def lattice_energy(length,lat,J,B):
    i,j = 0,0
    total_energy = 0
    while i < length:
        while j < length:
            s = lat[i][j]
            #Ensure periodic boundary condition is met on BOTTOM side
            if i+1==length:
                b = lat[0][j]*s
            else:
                b = lat[i+1][j]*s
            #Ensure periodic boundary condition is met on RIGHT side
            if j+1==length:
                r = lat[i][0]*s
            else:
                r = lat[i][j+1]*s
            t,l = lat[i-1][j]*s,lat[i][j-1]*s
            total_energy += -J(t+r+l+b)-B*s
    return total_energy
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Calculates the Energy of the a current spin and the energy if it were flipped
def calculate_delta_energy(v,h,lat,length,J,B):
    current_energy = 0
    flipped_energy = 0
    current = lat[v][h]
    flipped = -1*lat[v][h]
    t,r,l,b = 0,0,0,0
    #Ensure periodic boundary condition is met on BOTTOM side
    if v+1==length:
        b = lat[0][h]
    else:
        b = lat[v+1][h]
    #Ensure periodic boundary condition is met on RIGHT side
    if h+1==length:
        r = lat[v][0]
    else:
        r = lat[v][h+1]
    #Periodic boundries will be met otherwise
    t,l = lat[v-1][h],lat[v][h-1]
    current_energy = -J*(t+r+l+b)*current-B*current
    flipped_energy = -J*(t+r+l+b)*flipped-B*flipped
    return flipped_energy-current_energy
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Calculates the Energy of the a current spin and the energy if it were flipped
def calculate_energy(lat,length,J,B,v,h):
    current = lat[v][h]
    t,r,l,b = 0,0,0,0
    #Ensure periodic boundary condition is met on BOTTOM side
    if v+1==length:
        b = lat[0][h]
    else:
        b = lat[v+1][h]
    #Ensure periodic boundary condition is met on RIGHT side
    if h+1==length:
        r = lat[v][0]
    else:
        r = lat[v][h+1]
    #Periodic boundries will be met otherwise
    t,l = lat[v-1][h],lat[v][h-1]
    current_energy = -J*(t+r+l+b)*current-B*current
    return current_energy
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Calculates mean magnitization per spin by summing over all the dipole spins 
#and then divinding by the total number of dipoles
def mean_dipole_magnetization(lattice, length):
    total_dipoles = length**2
    magnetization = 0
    i=0
    while i<length:
        j=0
        while j<length:
            magnetization+=lattice[i][j]
            j+=1
        i+=1
    return magnetization/total_dipoles
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Calculates the mean energy per dipole by calculating the energy stored in each
#dipoleand then dividing by the total number of dipoles
def mean_dipole_energy(lattice,length,J,B):
    total_energy = 0
    i=0
    while i<length:
        j=0
        while j<length:
            b,r = 0,0 #bottom energy coupling and right energy coupling
            if i+1==length:
                b = lattice[0][j]
            else:
                b = lattice[i+1][j]
            if j+1==length:
                r = lattice[i][0]
            else:
                r = lattice[i][j+1]
            total_energy += -J*lattice[i][j]*(r+b)
            j+=1
        i+=1
    return total_energy/(length**2)
###############################################################################
# MAIN FUNCTIONS
###############################################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Question One Part A and B
def QuestionOne(J,T,B_o):
    k_b = 1 #boltzmann constant
    E_o = J #Normalized energy
    T = T/(E_o/k_b) #Normalized Temperature
    totMCS = 1000 #number of Monte Carlo Sweeps
    discard = 50  # how many MC sweeps to discard before writing data
    length = 128 #length of square lattice
    N = length**2 #total number of dipoles
    lattice = create_aligned_lattice(length,B_o) #lattice in array form
    m_B = [] #this will store the mean dipole magnetization per increment of B
    up = np.linspace(-5,5,51)
    down = np.linspace(4.8,-5,50)
    Fields=np.concatenate([up,down])
    print(Fields)
    for B in Fields:
        m = [] #this will store the mean dipole magnetization across a number of MCs for a specific B
        i = 0
        print(B)
        while i <= totMCS:
            #track the total number of up spins for computing enery, magnetization, etc
            j = 0
            while j < N: #N is the total number of spins to be randomly iterated over
                v = rand.randint(0,length-1)
                h = rand.randint(0,length-1)
                delta_energy = calculate_delta_energy(v,h,lattice,length,J,B)
                #if delta H less than 0 or if delta H is greater than 0 and it randomly gains energy then flip curent dipole
                if (delta_energy<=0) or (rand.uniform(0, 1)<np.exp(-delta_energy/(k_b*T))):
                    lattice[v][h]=-1*lattice[v][h]
                j+=1
            #check whether it is time to write some data
            if i > discard:
                m.append(mean_dipole_magnetization(lattice,length))
            i+=1
        m_B.append(sum(m)/(len(m)))
#        #CHECK if we make an auto-correlation plot for this value of B
#        if (B<.8+10**-6 and B>.8-10**-6) or B==1 or (B<1.2+10**-6 and B>1.2-10**-6):
#            autocorrelation(m,B,T)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Creates two graphs with the same data: one is a plot, the other is a scatter plot
    f,ax = plt.subplots()
    f.tight_layout()
    ax.set_xticks(np.linspace(-5,5,11))
    ax.plot(up,m_B[0:len(up)],color='k',label="B Field Increase")
    ax.plot(down,m_B[len(up):],color='r',label="B Field Decrease")
    ax.set_title("Magentic Field vs Mean Magnetization of Lattice",style='italic')
    ax.set_xlabel('B Field',style='italic')
    ax.set_ylabel('Mean Magnetization',style='italic')
    ax.legend(loc='upper left')
    f.show()
    f.savefig("FirstOrderPhaseT="+str(T)+"Plot.png",dpi=600)
    
    f1,ax1 = plt.subplots()
    f1.tight_layout()
    ax1.set_xticks(np.linspace(-5,5,11))
    ax1.scatter(up,m_B[0:len(up)],color='k',label="B Field Increase")
    ax1.scatter(down,m_B[len(up):],color='r',label="B Field Decrease")
    ax1.set_title("Magentic Field vs Mean Magnetization of Lattice",style='italic')
    ax1.set_xlabel('B Field',style='italic')
    ax1.set_ylabel('Mean Magnetization',style='italic')
    ax1.legend(loc='upper left')
    f1.show()
    f1.savefig("FirstOrderPhaseT="+str(T)+"Scatter.png",dpi=600)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Question Two Part A  and B
def QuestionTwo(J,T,B):
    k_b = 1 #boltzmann constant
    E_o = J #Normalized energy
    T = T/(E_o/k_b) #Normalized Temperature
    totMCS = 1000 #number of Monte Carlo Sweeps
    discard = 100
    length = 128 #length of square lattice
    N = length**2 #total number of dipoles
    lattice = create_random_lattice(length) #lattice in array form
    m = [] #this will store the mean dipole magnetization across a number of MCs for a specific B
    m2 = [] #stores mean dipole magneziation squared
    e = [] #this will store the mean energy per dipole per MCs
    sweeps = [] #holds the current sweep number
    i = 0
    while i <= totMCS:
        #print(i)
        #track the total number of up spins for computing enery, magnetization, etc
        j = 0
        while j < N: #N is the total number of spins to be randomly iterated over
            v = rand.randint(0,length-1)
            h = rand.randint(0,length-1)
            delta_energy = calculate_delta_energy(v,h,lattice,length,J,B)
            #if delta H less than 0 or if delta H is greater than 0 and it randomly gains energy then flip curent dipole
            if (delta_energy<=0) or (rand.uniform(0, 1)<np.exp(-delta_energy/(k_b*T))):
                lattice[v][h]=-1*lattice[v][h]
            j+=1
        M = mean_dipole_magnetization(lattice,length)
        E = mean_dipole_energy(lattice,length,J,B)
        m.append(M)
        m2.append(M**2)
        e.append(E)
        sweeps.append(i)
        i+=1
    autocorrelation(m[discard:],m2[discard:],B,T)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Plots the energy per dipole and magnetization per dipole versus the number
    #of Monte Carlo Sweeps
    f,ax = plt.subplots()
    f.tight_layout()
    ax.plot(sweeps,m,color='r',label="Mean")
    ax.set_title('Mean Magnetization Per Dipole'+" T="+str(T)+" B="+str(B),style='italic')
    ax.set_xlabel(r'$Monte-Carlo Sweeps$',style='italic')
    ax.set_ylabel(r'$Magnetization$',style='italic')
    f.show()
    f.savefig("MagnetizationT="+str(T)+"B="+str(B)+".png",dpi=600,bbox_inches='tight')
    
    f1,ax1 = plt.subplots()
    f1.tight_layout()
    ax1.plot(sweeps,e,color='r',label="Mean")
    ax1.set_title('Mean Energy Per Dipole'+" T="+str(T)+" B="+str(B),style='italic')
    ax1.set_xlabel(r'$Monte-Carlo Sweeps$',style='italic')
    ax1.set_ylabel(r'$Energy$',style='italic')
    f1.show()
    f1.savefig("EnergyT="+str(T)+"B="+str(B)+".png",dpi=600,bbox_inches='tight')
    
    f2,ax2 = plt.subplots()
    ax2.imshow(lattice, cmap='gray', interpolation='nearest')
    ax2.set_title('Spin Direction'+" T="+str(T)+" B="+str(B),style='italic')
    ax2.set_xlabel(r'$X$'+" Position",style='italic')
    ax2.set_ylabel(r'$Y$'+" Position",style='italic')
    f2.show()
    f2.savefig("SpinsT="+str(T)+"B="+str(B)+".png",dpi=600,bbox_inches='tight')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Question One Part C, Operates in Question 2
def autocorrelation(m,m2,B,T):
    #m is the array holding all of the magnetizations of the lattice for a certain B
    #B is the magnetic field that the lattice is under the influence of
    print("correlating...")
    autocorrelation = np.zeros(len(m))
    lag = []
    m_avg_sqrt = (sum(m)/len(m))**2
    m_sqrt_avg = sum(m2)/len(m2)
    t = 0
    while t < len(autocorrelation):
        suma = 0
        i = 0
        while i + t < len(m):
            suma += m[i+t]*m[i]
            i+=1
        suma = suma/i
        autocorrelation[t] = (suma - m_avg_sqrt)/(m_sqrt_avg-m_avg_sqrt)
        lag.append(t)
        t+=1
    print(T,B,np.std(m))
    #popt, pcov = curve_fit(func, lag, autocorrelation, maxfev=10000)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Plots autocorrelation for field B
    f,ax = plt.subplots()
    f.tight_layout()
    ax.scatter(lag,autocorrelation,color='r')
    #ax.plot(lag,func(lag,*popt),color='r',linestyle='dotted',label='fit: b='+str(round(popt[0],4)))
    ax.set_title(r'$Auto-Correlation$'+" T="+str(T)+" B="+str(round(B,2)),style='italic')
    ax.set_xlabel('Lag in Monte Carlo Sweeps',style='italic')
    ax.set_ylabel(r'$Auto-Correlation$',style='italic')
    f.show()
    f.savefig("AutocorrelationT="+str(T)+"B="+str(B)+"Scatter.png",dpi=600,bbox_inches='tight')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Question 2 part C
def QuestionTwoC(J,random,length):
    k_b, E_o = 1, np.abs(J) #boltzmann constant, Normalized Energy
    T_o = (E_o/k_b) #Normalized Temperature
    B = 0 #No magnetic Field
    totMCS, discard = 1000,400 #number of Monte Carlo Sweeps #number of MCS to discard
    N = length**2 #total number of dipoles
    lattice = []
    if random:
        lattice = create_random_lattice(length) 
    else:
        lattice = create_aligned_lattice(length,1)
    mm, specific_heat, susceptibility = [], [], [] #holds mean magentization per dipole per temp, specific_heat per temp and, mag per temp
    Temps = np.linspace(1.769,2.769,41)/T_o
    print(Temps)
    for T in Temps:
        i = 0
        MMM,E,E2,M,M2 = [],[],[],[],[] #stores total energy, total energy Squared, total mag, and total mag sqaured
        print(T)
        while i <= totMCS:
            #track the total number of up spins for computing enery, magnetization, etc
            j = 0
            while j < N: #N is the total number of spins to be randomly iterated over
                v = rand.randint(0,length-1)
                h = rand.randint(0,length-1)
                delta_energy = calculate_delta_energy(v,h,lattice,length,J,B)
                #if delta H less than 0 or if delta H is greater than 0 and it randomly gains energy then flip curent dipole
                if (delta_energy<=0) or (rand.uniform(0, 1)<np.exp(-delta_energy/(k_b*T))):
                    lattice[v][h]=-1*lattice[v][h]
                j+=1
            
            m = mean_dipole_magnetization(lattice,length)*N
            e = mean_dipole_energy(lattice,length,J,B)*N
            MMM.append(m/N)
            M.append(m)
            M2.append(m**2)
            E.append(e)
            E2.append(e**2)
            i+=1
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #Calculates the specific heat per spin and the magnetic susceptibility per spin
        e_sqrt_avg = sum(E2[discard:])/(len(E2[discard:]))
        e_avg_sqrt = (sum(E[discard:])/(len(E[discard:])))**2
        m_sqrt_avg = sum(M2[discard:])/(len(M2[discard:]))
        m_avg_sqrt = (sum(M[discard:])/(len(M[discard:]))) **2
        mm_avg = sum(MMM[discard:])/len(MMM[discard:])
        mm.append(mm_avg)
        specific_heat.append((e_sqrt_avg-e_avg_sqrt)/((length**2)*(T**2)*k_b))
        susceptibility.append((m_sqrt_avg-m_avg_sqrt)/((length**2)*T*k_b))
        
#        if random==False and T<2.35:
#            f2,ax2 = plt.subplots()
#            ax2.imshow(lattice, cmap='gray', interpolation='nearest')
#            ax2.set_title('Spin Direction'+" J="+str(J)+" T="+str(T)+" B="+str(B),style='italic')
#            ax2.set_xlabel(r'$X$'+" Position",style='italic')
#            ax2.set_ylabel(r'$Y$'+" Position",style='italic')
#            f2.show()
#            f2.savefig("AntiFerromagnetismT="+str(T)+".png",dpi=600,bbox_inches='tight')
        

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Plots specific heat per spin and the magnetic susceptibility per spin as a
    #function of T/T_o
    f,ax = plt.subplots()
    f.tight_layout()
    ax.plot(Temps,specific_heat,color='r')
    ax.set_title('Specific Heat '+r'$(C_v)$',style='italic')
    ax.set_xlabel(r'$Temperature$'+" "+r'$\left(\frac{Tk_B}{J}\right)$',style='italic')
    ax.set_ylabel('Specific Heat',style='italic')
    f.show()
    if random==False:
        f.savefig("AntiSpecificHeat"+str(length)+".png",dpi=600,bbox_inches='tight')
    else:
        f.savefig("SpecificHeat"+str(length)+".png",dpi=600,bbox_inches='tight')
    
    f1,ax1 = plt.subplots()
    f1.tight_layout()
    ax1.plot(Temps,susceptibility,color='r')
    ax1.set_title('Magentic Susceptibility '+r'$(X)$',style='italic')
    ax1.set_xlabel(r'$Temperature$'+" "+r'$\left(\frac{Tk_B}{J}\right)$',style='italic')
    ax1.set_ylabel('Susceptibility',style='italic')
    f1.show()
    if random==False:
        f1.savefig("AntiMagSuscept"+str(length)+".png",dpi=600,bbox_inches='tight')
    else:
        f1.savefig("MagSuscept"+str(length)+".png",dpi=600,bbox_inches='tight')
        
    f3,ax3 = plt.subplots()
    f3.tight_layout()
    ax3.plot(Temps,mm,color='r',label="Mean")
    ax3.set_title('Mean Magnetization Per Dipole'+" T="+str(T)+" B="+str(B),style='italic')
    ax3.set_xlabel(r'$Temperature$'+" "+r'$\left(\frac{Tk_B}{J}\right)$',style='italic')
    ax3.set_ylabel(r'$Magnetization$',style='italic')
    f3.show()
    if random==False:
        f3.savefig("AntiMeanMag"+str(length)+".png",dpi=600,bbox_inches='tight')
    else:
        f3.savefig("MeanMag"+str(length)+".png",dpi=600,bbox_inches='tight')

###############################################################################
# MAIN CODE
###############################################################################
#Question One~~~~~~~~~~~~~~~~~~~~~~
#J,T,B_o
#QuestionOne(1,1,-5)
#QuestionOne(1,4,-5)
    
#QuestionTwo(1,1,-1)
#QuestionTwo(1,1,1)
#QuestionTwo(1,1,3)
#Question Two~~~~~~~~~~~~~~~~~~~~~~
#QuestionTwo(1,1,0)
#QuestionTwo(1,2.269,0)
#QuestionTwo(1,4,0)

#Question Two Part C~~~~~~~~~~~~~~~
#QuestionTwoC(1,True,32)
#QuestionTwoC(1,True,64)
#QuestionTwoC(1,True,128)
#QuestionTwoC(1,True,256)

#Question Three
QuestionTwoC(-1,False,128)

