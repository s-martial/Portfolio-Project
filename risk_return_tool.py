import pandas as pd
import numpy as np
import scipy.stats


def index_return():
    indexr = pd.read_csv("C:\\Users\Stephane\RR-Index\index_close.csv",
                    header=0, index_col=1, parse_dates=True)
    idxr = indexr[['CAC40GROSSTR', 'DAX', 'IndustrialAverageIndexTotalReturn']]
    idxr.columns = ['CAC', 'DAX', 'DJIA']
    idxr.index = pd.to_datetime(idxr.index, format="%Y%m").to_period('M')
    idxr = idxr.pct_change().dropna()
    return idxr



def vol(r):
    """
    std() use a numerator of -1 so we have to get the nb of elements/observation with shape[0]
    Take a time series of returns, get the deviation, squared deviation and variance to calculate the volatility with the same base as the series
    """
    nb_months = r.shape[0]
    deviation = r - r.mean()
    squared_deviation = deviation**2
    variance = squared_deviation.sum()/(nb_months - 1)
    volatility = np.sqrt(variance)
    return volatility



def annualized_vol(r):
    """
    Taking a time series of returns to calculate the volatility and scaling it to an annualized value
    """
    an_vol = vol(r)*np.sqrt(12)
    return an_vol



def compounded_return(r):
    com_return = (((r+1).prod()-1)*100).round(2)
    return com_return



def annualized_return(r):
    nb_months = r.shape[0]
    an_return = (r+1).prod()**(12/nb_months) - 1
    return an_return
    
    
    
def rr_ratio(r):
    """
    Risk-Return Ratio
    """
    rr = annualized_return(r)/annualized_vol(r)
    return rr



def drawdown(data_series: pd.Series):
    """
    take a time series of asset returns, computes and returns a DataFrame that contains:
    the wealth index : Equity value day by day/month by month/Year ...
    the previous peak : High
    the drawdown percentage
    """
    wealth_index = 1000*(1 + data_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    #now we gonna send back everything as a DataFrame
    return pd.DataFrame({
        "Equity": wealth_index,
        "High": previous_peaks,
        "Drawdown": drawdowns})




def skewness(r):
    """alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or Dataframe
    Returns a float or a series
    """
    demeaned_r = r - r.mean()
    #use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0) #ddof=0 mean the degree of freedom as been set to zero = don't make that in -1 correction
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3



def kurtosis(r):
    """alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or Dataframe
    Returns a float or a series
    """
    demeaned_r = r - r.mean()
    #use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0) #ddof=0 mean the degree of freedom as been set to zero = don't make that in -1 correction
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4




def is_normal(r, level=0.01):
    """fix the issue for JB that it treat all the data like one massive set of returns, we want to run it on each data
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is apploed at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level




def semideviation(r):
    """Returns the semideviation aka negative semideviation of r
    r must be a Series or DataFrame
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)



def sharpe_ratio(r, riskfree_rate):
    """r = Asset or Portfolio return
    riskfree_rate = RF Rate or Treasury yield
    /!\ annualized_return & annualized_vol to be defined"""
    excess_return = r - riskfree_rate
    annualized_excess_return = annualized_return(excess_return)
    annualized_volatility = annualized_vol(r)
    sharpe_r = annualized_excess_return / annualized_volatility
    return sharpe_r

