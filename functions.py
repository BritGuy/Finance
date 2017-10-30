import math, random
import numpy as np

#normal distribution function
def py_NormalDistribution(x):
    ND = 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)
    return ND;
    
#cumulative normal distribution - Hart 1968
def py_CND(x):
    y = np.absolute(x)
    if y > 37:
        CND = 0
    else:
        Exponential = np.exp(-y**2/2)
        if y < 7.07106781186547:
            SumA = 3.52624965998911E-02 * y + 0.700383064443688
            SumA = SumA * y + 6.37396220353165
            SumA = SumA * y + 33.912866078383
            SumA = SumA * y + 112.079291497871
            SumA = SumA * y + 221.213596169931
            SumA = SumA * y + 220.206867912376
            SumB = 8.83883476483184E-02 * y + 1.75566716318264
            SumB = SumB * y + 16.064177579207
            SumB = SumB * y + 86.7807322029461
            SumB = SumB * y + 296.564248779674
            SumB = SumB * y + 637.333633378831
            SumB = SumB * y + 793.826512519948
            SumB = SumB * y + 440.413735824752
            CND = Exponential * SumA / SumB
        else:
            SumA = y + 0.65
            SumA = y + 4 / SumA
            SumA = y + 3 / SumA
            SumA = y + 2 / SumA
            SumA = y + 1 / SumA
            CND = Exponential / (SumA * 2.506628274631)
    if x > 0:
        CND = 1 - CND
            
    return CND;
    
#converts rate to continuous compounding rate
def py_ConvertToCCRate(r, compoundings):
    if compoundings == 0:
        ccRate = r
    else:
        ccRate = compoundings * np.log(1+r/compoundings)
    return ccRate;
    
#Generalized Black Scholes Merton Option Price
def py_BSM(CallPutFlag,S,x,T,r,b,v):
    d1 = (np.log(S/x) + (b+v**2 / 2) * T)/(v*np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)
    
    if CallPutFlag is "c" or "C":
        Output = S * np.exp((b-r)*T)*py_CND(d1) - x * np.exp(-r * T) * py_CND(d2)
    else:
        Output = x * np.exp(-r * T) * py_CND(-d2) - S * np.exp((b - r)* T) * py_CND(-d1)
    return Output;

#Generalized Black Scholes Merton with Greek Outputs
def py_EuropeanOption(OutputFlag,CallPutFlag,S,x,T,r,b,v,dS = 0.01):
    if OutputFlag is "price":
        Output = py_BSM(CallPutFlag,S,x,T,r,b,v)
    elif OutputFlag is "delta":
        Output = (py_BSM(CallPutFlag,S + dS,x,T,r,b,v) - py_BSM(CallPutFlag,S - dS,x,T,r,b,v))/(2*dS)
    elif OutputFlag is "gamma":    
        Output = (py_BSM(CallPutFlag,S + dS,x,T,r,b,v) - 2 * py_BSM(CallPutFlag,S,x,T,r,b,v) + py_BSM(CallPutFlag,S - dS,x,T,r,b,v))/ ds**2 
    elif OutputFlag is "vega":
        Output = (py_BSM(CallPutFlag,S,x,T,r,b,v + 0.01) - py_BSM(CallPutFlag,S,x,T,r,b,v-0.01))/2
    elif OutputFlag is "rho":
        Output = (py_BSM(CallPutFlag,S,x,T,r + 0.01,b + 0.01,v) - py_BSM(CallPutFlag,S,x,T,r - 0.01,b - 0.01,v))/2
    elif OutputFlag is "theta":
        if T <= 1/365:
            Output = py_BSM(CallPutFlag,S,x,0.00001,r,b,v) - py_BSM(CallPutFlag,S,x,T,r,b,v)
        else:
            Output = py_BSM(CallPutFlag,S,x,t - 1/365,r,b,v) - py_BSM(CallPutFlag,S,x,T,r,b,v)
    elif OutputFlag is "zomma" or "dGamma_dVol":
        Output = (py_BSM(CallPutFlag,S + dS,x,T,r,b,v + 0.01) - 
                  2*py_BSM(CallPutFlag,S,x,T,r,b,v + 0.01)
                  +py_BSM(CallPutFlag,S - dS,x,T,r,b,v + 0.01)
                  -py_BSM(CallPutFlag,S + dS,x,T,r,b,v - 0.01)
                  +2*py_BSM(CallPutFlag,S,x,T,r,b,v - 0.01)
                  -py_BSM(CallPutFlag,S - dS,x,T,r,b,v - 0.01))/(2*0.01 * dS**2)/100
    
    #Unexpected Flag Error Handling
    else:
        Output = "Unknown Output Flag"
    
    return Output;
    
#Cash or Nothing Digital - Base Pricer
#variables descriptions:
#CallPutFlag - Anything other than C or c will default to Put
#S = Asset Spot Rate
#x = Asset Strike Price
#k = Cash Delivery
#T = Time to Maturity
#r = Risk-free rate
#b = Cost of Carry
#v = Volatility

def py_Digital_CashOrNothing(CallPutFlag,S,x,k,T,r,b,v):
    d = (np.log(S/x) + (b - v**2 / 2) * T) / (v * np.sqrt(T))
    
    if CallPutFlag is "C" or "c" :
        Output = k * np.exp(-r * T) * py_CND(d)
    else:
        Output = k * np.exp(-r * T) * py_CND(-d)
    return Output;       
    
def py_EuropeanDigital_CashOrNothing(OutputFlag,CallPutFlag,S,x,k,T,r,b,v,dS = 0.01):
    if OutputFlag is "price":
        Output = py_Digital_CashOrNothing(CallPutFlag,S,x,k,T,r,b,v)
    elif OutputFlag is "delta":
        Output = (py_Digital_CashOrNothing(CallPutFlag,S + dS,x,k,T,r,b,v) - py_Digital_CashOrNothing(CallPutFlag,S - dS,x,k,T,r,b,v))/(2*dS)
    elif OutputFlag is "gamma":    
        Output = (py_Digital_CashOrNothing(CallPutFlag,S + dS,x,k,T,r,b,v) - 2 * py_Digital_CashOrNothing(CallPutFlag,S,x,k,T,r,b,v) + py_Digital_CashOrNothing(CallPutFlag,S - dS,x,k,T,r,b,v))/ ds**2 
    elif OutputFlag is "vega":
        Output = (py_Digital_CashOrNothing(CallPutFlag,S,x,k,T,r,b,v + 0.01) - py_Digital_CashOrNothing(CallPutFlag,S,x,k,T,r,b,v-0.01))/2
    elif OutputFlag is "rho":
        Output = (py_Digital_CashOrNothing(CallPutFlag,S,x,k,T,r + 0.01,b + 0.01,v) - py_Digital_CashOrNothing(CallPutFlag,S,x,k,T,r - 0.01,b - 0.01,v))/2
    elif OutputFlag is "theta":
        if T <= 1/365:
            Output = py_Digital_CashOrNothing(CallPutFlag,S,x,k,0.00001,r,b,v) - py_Digital_CashOrNothing(CallPutFlag,S,x,k,T,r,b,v)
        else:
            Output = py_Digital_CashOrNothing(CallPutFlag,S,x,k,t - 1/365,r,b,v) - py_Digital_CashOrNothing(CallPutFlag,S,x,k,T,r,b,v)
    elif OutputFlag is "zomma" or "dGamma_dVol":
        Output = (py_Digital_CashOrNothing(CallPutFlag,S + dS,x,k,T,r,b,v + 0.01) - 
                  2*py_Digital_CashOrNothing(CallPutFlag,S,x,k,T,r,b,v + 0.01)
                  +py_Digital_CashOrNothing(CallPutFlag,S - dS,x,k,T,r,b,v + 0.01)
                  -py_Digital_CashOrNothing(CallPutFlag,S + dS,x,k,T,r,b,v - 0.01)
                  +2*py_Digital_CashOrNothing(CallPutFlag,S,x,k,T,r,b,v - 0.01)
                  -py_Digital_CashOrNothing(CallPutFlag,S - dS,x,k,T,r,b,v - 0.01))/(2*0.01 * dS**2)/100
    #Unexpected Flag Error Handling
    else:
        Output = "Unknown Output Flag"
    return Output;

def py_MonteCarloOption(CallPutFlag,S,x,T,r,b,v,nSim):
    #parameters for monte carlo path
    Drift = (b-v**2 / 2)*T
    vSqrdt = v * np.sqrt(T)
    
    #initilize monte carlo counter and sum
    MonteCarlo =0
    RunSim = 0
    SumMC = 0
    myList = [0]
    #Call/Put option selection multiplier
    if CallPutFlag is "c":
        z = 1
    else:
        z = -1
    
    #Monte Carlo Simulation
    while RunSim < nSim:
        ST = S*np.exp(Drift + vSqrdt * sct.norm.ppf(random.random()))
        SumMC = SumMC + max(z *(ST - x),0)
        RunSim = RunSim + 1

    #Discounting Monte Carlo result
        MonteCarlo = np.exp(-r * T)*(SumMC / nSim)
    
    return MonteCarlo;    
    
