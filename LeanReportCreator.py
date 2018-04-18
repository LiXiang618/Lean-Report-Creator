# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 09:26:20 2018

@author: Li Xiang
"""

import os
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
font = {'family': 'Open Sans Condensed'}
matplotlib.rc('font',**font)
la = matplotlib.font_manager.FontManager()
lu = matplotlib.font_manager.FontProperties(family = "Open Sans Condensed")
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
from datetime import date
import re
import math

class LeanReportCreator(object):
    
    def __init__(self, jsonfile, outdir = "outputs"):
        self.outdir = outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        with open(jsonfile) as data_file:    
            try:
                data = json.load(data_file)    
            except ValueError:
                data = {"Charts": []}

        self.is_drawable = False
        if "Strategy Equity" in data["Charts"] and "Benchmark" in data["Charts"]:
            strategySeries = data["Charts"]["Strategy Equity"]["Series"]["Equity"]["Values"] 
            self.initStrategyValue = strategySeries[0]['y']
            benchmarkSeries = data["Charts"]["Benchmark"]["Series"]["Benchmark"]["Values"] 
            self.initBenchmarkValue = benchmarkSeries[0]['y']
            df_strategy = pd.DataFrame(strategySeries).set_index('x')
            df_benchmark = pd.DataFrame(benchmarkSeries).set_index('x')
            df_strategy = df_strategy[~df_strategy.index.duplicated(keep='first')]
            df_benchmark = df_benchmark[~df_benchmark.index.duplicated(keep='first')]
            df = pd.concat([df_strategy,df_benchmark],axis = 1)
            df.columns = ['Strategy','Benchmark']
            df = df.set_index(pd.to_datetime(df.index, unit='s'))
            self.df = df.fillna(method = 'ffill')
            self.df = df.fillna(method = 'bfill')
            self.is_drawable = True
    
    def cumulative_return(self, name = "cumulative-return.png", width = 11.5, height = 2.5):
        if self.is_drawable:
            df_this = self.df.copy()
            df_this["Strategy"] = (df_this["Strategy"]/self.initStrategyValue-1)*100
            df_this["Benchmark"] = (df_this["Benchmark"]/self.initBenchmarkValue-1)*100
            
            plt.figure()
            ax = df_this.plot(color = ["#F5AE29","grey"])
            fig = ax.get_figure()
            plt.xticks(rotation = 0,ha = 'center')
            plt.xlabel("")
            plt.ylabel('Cumulative Return(%)',size = 12,fontweight='bold')
            ax.legend(["Strategy","Benchmark"],prop = {'weight':'bold'},frameon=False, loc = "upper left")
            ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
            plt.axhline(y = 0, color = 'black')
            ax.grid()
            fig.set_size_inches(width, height)
            #plt.show()
            fig.savefig(self.outdir + "/" + name)
            plt.cla()
            plt.clf()
            plt.close('all')
        return True
        
    def daily_returns(self, name = "daily-returns.png", width = 11.5, height = 2.5):
        if self.is_drawable:
            df_this = self.df.copy()
            df_this.drop("Benchmark",1,inplace = True)
            df_this = df_this.groupby([df_this.index.date]).apply(lambda x: x.tail(1))
            df_this.index = df_this.index.droplevel(1)
            ret_strategy = np.array([self.initStrategyValue] + df_this["Strategy"].tolist())
            ret_strategy = ret_strategy[1:]/ret_strategy[:-1] - 1
            df_this["Strategy"] = ret_strategy*100
            df_this.rename(columns = {"Strategy":"Above Zero"},inplace = True)
            df_this["Below Zero"] = [min(0,x) for x in df_this["Above Zero"]]
            df_this["Above Zero"] = [max(0,x) for x in df_this["Above Zero"]]
            
            plt.figure()
            ax = df_this.plot.bar(color = ["#F5AE29","grey"])
            fig = ax.get_figure()
            plt.xticks(rotation = 0,ha = 'center')
            plt.xlabel("")
            plt.ylabel('Daily Return(%)',size = 12,fontweight='bold')
            ax.legend(["Above Zero","Below Zero"],prop = {'weight':'bold'},frameon=False, loc = "upper left")
            if len(df_this) > 10:
                nticks = min(len(df_this),5)
                step = int(len(df_this)/nticks)
                tickId = [x for x in range(0, step*nticks,step)]
                plt.xticks(tickId,[df_this.index.values[x].strftime('%b %Y') for x in tickId])
            else:
                tickerlabels = [x.strftime('%b %Y') for x in df_this.index]
                ax.xaxis.set_major_formatter(ticker.FixedFormatter(tickerlabels))
            plt.axhline(y = 0, color = 'black')
            ax.grid()
            fig.set_size_inches(width, height)
            #plt.show()
            fig.savefig(self.outdir + "/" + name)
            plt.cla()
            plt.clf()
            plt.close('all')                
        return True
        
    def drawdown(self,name = "drawdowns.png",width = 11.5, height = 2.5):
        if self.is_drawable:
            df_this = self.df.copy()
            df_this.drop("Benchmark",1,inplace = True)
            df_this["Drawdown"] = 1
            lastPeak = self.initStrategyValue
            for i in range(len(df_this)):
                if df_this.iloc[i,0] < lastPeak:
                    df_this.iloc[i,1] = df_this.iloc[i,0]/lastPeak
                else:
                    lastPeak = df_this.iloc[i,0]        
            df_this["DDGroup"] = 0
            tmp = 0
            for i in range(1,len(df_this)):
                if df_this.iloc[i,1] != 1:
                    df_this.iloc[i,2] = tmp
                else:
                    continue
                if df_this.iloc[i-1,1] == 1:
                    tmp += 1
                    df_this.iloc[i,2] = tmp       
            df_this["index"] = [i for i in range(len(df_this))]
            tmp_df = pd.DataFrame.from_dict({'MDD':df_this.groupby([df_this["DDGroup"]])['Drawdown'].min(),
                                          'Offset':df_this.groupby([df_this["DDGroup"]])['Drawdown'].apply(lambda x: np.where(x == min(x))[0][0]),
                                          'Start':df_this.groupby([df_this["DDGroup"]])['index'].first(),
                                          'End':df_this.groupby([df_this["DDGroup"]])['index'].last()})
            tmp_df.drop(tmp_df.index[[0]],inplace = True)
            tmp_df.sort_values("MDD",inplace = True)
            df_this = (df_this["Drawdown"] - 1)*100
            
            plt.figure()
            tmp_colors = ["#FFCCCCCC","#FFE5CCCC","#FFFFCCCC","#E5FFCCCC","#CCFFCCCC"]
            tmp_texts = ["1st Worst","2nd Worst","3rd Worst","4th Worst","5th Worst"]
            ax = df_this.plot(color = "grey",zorder = 2)
            ax.fill_between(df_this.index.values,df_this,0, color = "grey",zorder = 3)
            for i in range(min(len(tmp_df),5)):
                tmp_start = df_this.index.values[int(tmp_df.iloc[i]["Start"])]
                tmp_end = df_this.index.values[int(tmp_df.iloc[i]["End"])]
                tmp_mid = df_this.index.values[int(tmp_df.iloc[i]["Offset"])+int(tmp_df.iloc[i]["Start"])]
                plt.axvspan(tmp_start, tmp_end,0,0.95, color = tmp_colors[i],zorder = 1)
                plt.axvline(tmp_mid, 0,0.95, ls = "dashed",color ="black", zorder = 4)
                plt.text(tmp_mid,min(df_this)*0.75,tmp_texts[i], rotation = 90, zorder = 4)
            fig = ax.get_figure()
            plt.xticks(rotation = 0,ha = 'center')
            plt.xlabel("")
            plt.ylabel('Drawdown(%)',size = 12,fontweight='bold')
            ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
            plt.axhline(y = 0, color = 'black')
            ax.grid()
            fig.set_size_inches(width, height)
            #plt.show()
            fig.savefig(self.outdir + "/" + name)
            plt.cla()
            plt.clf()
            plt.close('all')
        return True
   
    def monthly_returns(self, name = "monthly-returns.png",width = 3.5*2, height = 2.5*2):
        if self.is_drawable:
            df_this = self.df.copy()
            df_this.drop("Benchmark",1,inplace = True)
            df_this1 = df_this.groupby([df_this.index.year,df_this.index.month]).apply(lambda x: x.head(1))
            df_this2 = df_this.groupby([df_this.index.year,df_this.index.month]).apply(lambda x: x.tail(1))
            df_this1.index = df_this1.index.droplevel(2)
            df_this2.index = df_this2.index.droplevel(2)
            df_this = pd.concat([df_this1,df_this2],axis = 1)
            df_this["Return"] = (df_this.iloc[:,1] / df_this.iloc[:,0] - 1) * 100
            df_this = df_this.iloc[:,2]
            df_this = df_this.unstack()
            df_this = df_this.iloc[::-1]
            
            def make_colormap(seq):
                seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
                cdict = {'red': [], 'green': [], 'blue': []}
                for i, item in enumerate(seq):
                    if isinstance(item, float):
                        r1, g1, b1 = seq[i - 1]
                        r2, g2, b2 = seq[i + 1]
                        cdict['red'].append([item, r1, r2])
                        cdict['green'].append([item, g1, g2])
                        cdict['blue'].append([item, b1, b2])
                return mcolors.LinearSegmentedColormap('CustomMap', cdict)
            c = mcolors.ColorConverter().to_rgb
            c_map = make_colormap([c('#CC0000'),0.1,c('#FF0000'),0.2,c('#FF3333'),
                                     0.3,c('#FF9933'),0.4,c('#FFFF66'),0.5,c('#FFFF99'),
                                           0.6,c('#B2FF66'),0.7,c('#99FF33'),0.8,
                                                 c('#00FF00'),0.9, c('#00CC00')])   
            
            plt.figure()
            ax = plt.imshow(df_this, aspect='auto',cmap=c_map, interpolation='none',vmin = -10, vmax = 10)
            fig = ax.get_figure()
            fig.set_size_inches(3.5*2,2.5*2)
            plt.xlabel('Month',size = 12,fontweight='bold')
            plt.ylabel('Year',size = 12,fontweight='bold')
            plt.yticks(range(len(df_this.index.values)),df_this.index.values)
            plt.xticks(range(12),["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
            for (j,i),label in np.ndenumerate(df_this):   
                plt.text(i,j,round(label,1),ha='center',va='center')
            fig.set_size_inches(width, height)
            fig.savefig(self.outdir + "/" + name)
            plt.cla()
            plt.clf()
            plt.close('all')
        return True
    
    def annual_returns(self, name = "annual-returns.png",width = 3.5*2, height = 2.5*2):
        if self.is_drawable:
            df_this = self.df.copy()
            df_this.drop("Benchmark",1,inplace = True)
            df_this1 = df_this.groupby([df_this.index.year]).apply(lambda x: x.head(1))
            df_this2 = df_this.groupby([df_this.index.year]).apply(lambda x: x.tail(1))
            df_this1.index = df_this1.index.droplevel(1)
            df_this2.index = df_this2.index.droplevel(1)
            df_this = pd.concat([df_this1,df_this2],axis = 1)
            df_this["Return"] = (df_this.iloc[:,1] / df_this.iloc[:,0] - 1) * 100
            df_this = df_this.iloc[:,2]
            
            plt.figure()
            ax = df_this.plot.barh(color = ["#428BCA"])
            fig = ax.get_figure()
            plt.xticks(rotation = 0,ha = 'center')
            plt.xlabel("Returns(%)")
            plt.ylabel('Year',size = 12,fontweight='bold')
            plt.axvline(x = 0, color = 'black')
            vline = plt.axvline(x = np.mean(df_this),color = "red", ls = "dashed", label = "mean")
            plt.legend([vline],["mean"],loc='upper left')
            ax.grid()
            fig.set_size_inches(width, height)
            fig.savefig(self.outdir + "/" + name)
            plt.cla()
            plt.clf()
            plt.close('all')
        return True
    
    def monthly_return_distribution(self, name = "distribution-of-monthly-returns.png",width = 3.5*2, height = 2.5*2):
        if self.is_drawable:
            df_this = self.df.copy()
            df_this.drop("Benchmark",1,inplace = True)
            df_this1 = df_this.groupby([df_this.index.year,df_this.index.month]).apply(lambda x: x.head(1))
            df_this2 = df_this.groupby([df_this.index.year,df_this.index.month]).apply(lambda x: x.tail(1))
            df_this1.index = df_this1.index.droplevel(2)
            df_this2.index = df_this2.index.droplevel(2)
            df_this = pd.concat([df_this1,df_this2],axis = 1)
            df_this["Return"] = (df_this.iloc[:,1] / df_this.iloc[:,0] - 1) * 100
            df_this["Group"] = np.floor(df_this["Return"])
            tmp_mean = np.mean(df_this["Return"])
            tmp_mean = 11 if tmp_mean > 10 else -11 if tmp_mean < -10 else tmp_mean
            df_this = df_this.iloc[:,[2,3]]
            df_this["Group"] = [x if x<=10 and x>=-10 else float("-Inf") if x<-10 else float("Inf") for x in df_this["Group"]]
            df_this = df_this.groupby([df_this["Group"]]).count()
            tmp_min = int(min(max(min(df_this.index.values),-11),0))
            tmp_max = int(max(min(max(df_this.index.values), 11),0))
            for i in range(max(tmp_min,-10), min(tmp_max,10)+1):
                if i not in df_this.index.values:
                    tmp = df_this.iloc[0].copy()
                    tmp[0] = 0
                    tmp.name = np.float64(i)
                    df_this = df_this.append(tmp,ignore_index = False)
            df_this.sort_index(inplace = True)
            df_this.index = [">10" if x == float("Inf") else "<-10" if x == float("-Inf") else int(x) for x in df_this.index]

            plt.figure()
            ax = df_this.plot.bar(color = ["#F5AE29"])
            fig = ax.get_figure()
            plt.xticks(rotation = 0,ha = 'center')
            plt.xlabel("Returns(%)")
            plt.ylabel('Number of Months',size = 12,fontweight='bold')
            plt.axvline(x = -tmp_min, color = 'black')
            vline = plt.axvline(x = tmp_mean-tmp_min,color = "red", ls = "dashed", label = "mean")    
            plt.legend([vline],["mean"],loc='upper left')
            ax.grid()
            fig.set_size_inches(width, height)
            fig.savefig(self.outdir + "/" + name)
            plt.cla()
            plt.clf()
            plt.close('all')
        return True
    
    def crisis_events(self, width = 3.5*2, height = 2.5*2):
        if self.is_drawable:
            df_this = self.df.copy()
            start_date = ["2000-03-10","2001-09-11","2003-01-08","2008-08-01","2010-05-05",
                                  "2007-08-01","2008-03-01","2008-09-01","2009-01-01","2009-03-01",
                                  "2011-08-05","2011-03-16","2012-09-10",
                                  "2014-04-01","2014-10-01","2015-08-15",
                                  "2005-01-01","2007-08-01","2009-04-01","2013-01-01"]
            end_date = ["2000-09-10","2001-10-11","2003-02-07","2008-09-30","2010-05-10",
                                "2007-08-31","2008-03-31","2008-09-30","2009-02-28","2009-05-31",
                                "2011-09-05","2011-04-16","2012-10-10",
                                "2014-04-30","2014-10-31","2015-09-30",
                                "2007-07-31","2009-03-31","2012-12-31",str(date.today())]
            titles = ["Dotcom","9-11","US Housing Bubble 2003","Lehman Brothers","Flash Crash",
                       "Aug07","Mar08","Sept08","2009Q1","2009Q2",
                       "US Downgrade-European Debt Crisis","Fukushima Melt Down 2011","ECB IR Event 2012",
                       "Apr14","Oct14","Fall2015",
                       "Low Volatility Bull Market","GFC Crash","Recovery","New Normal"]
            
            for i in range(len(start_date)):    
                df_this_tmp = df_this[start_date[i]:end_date[i]].copy()
                if not len(df_this_tmp):
                    continue
                df_this_tmp["Strategy"] = (df_this_tmp["Strategy"]/df_this_tmp["Strategy"][0]-1)*100
                df_this_tmp["Benchmark"] = (df_this_tmp["Benchmark"]/df_this_tmp["Benchmark"][0]-1)*100
                plt.figure()
                ax = df_this_tmp.plot(color = ["#F5AE29","grey"])
                fig = ax.get_figure()
                plt.xticks(ha = 'center')
                plt.xlabel("")
                plt.ylabel('Return(%)',size = 12,fontweight='bold')
                ax.legend(["Strategy","Benchmark"],prop = {'weight':'bold'},frameon=False, loc = "upper left")
                ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
                plt.axhline(y = 0, color = 'black')
                ax.grid()
                fig.set_size_inches(width, height) 
                fig.savefig(self.outdir + "/crisis" +re.sub(r' ','-',titles[i].lower())+".png")
                plt.cla()
                plt.clf()
                plt.close('all')
        return True
    
    def rolling_beta(self, name = "rolling-portfolio-beta-to-equity.png",width = 11.5, height = 2.5):
        if self.is_drawable:
            days_L = 252
            days_S = 126
            if len(set(self.df.index.date)) > days_L:
                df_this = self.df.copy()
                df_this = df_this.groupby([df_this.index.date]).apply(lambda x: x.tail(1))
                df_this.index = df_this.index.droplevel(1)    
                ret_strategy = np.array([self.initStrategyValue] + df_this["Strategy"].tolist())
                ret_strategy = ret_strategy[1:]/ret_strategy[:-1] - 1
                df_this["Strategy"] = ret_strategy*100
                ret_benchmark = np.array([self.initBenchmarkValue] + df_this["Benchmark"].tolist())
                ret_benchmark = ret_benchmark[1:]/ret_benchmark[:-1] - 1
                df_this["Benchmark"] = ret_benchmark*100                    
                df_this["Beta6mo"] = float("nan")
                df_this["Beta12mo"] = float("nan")    
                for i in range(days_L, len(df_this)):
                    cov_matrix = np.cov(df_this["Strategy"][(i-days_L):i],df_this["Benchmark"][(i-days_L):i])
                    df_this.iloc[[i],[3]] = cov_matrix[0,1]/cov_matrix[1,1]
                for i in range(days_S, len(df_this)):
                    cov_matrix = np.cov(df_this["Strategy"][(i-days_S):i],df_this["Benchmark"][(i-days_S):i])
                    df_this.iloc[[i],[2]] = cov_matrix[0,1]/cov_matrix[1,1]
                df_this.drop(["Benchmark","Strategy"],1,inplace = True)    
                df_this["Empty"] = 0
                
                plt.figure()
                ax = df_this.plot(color = ["#CCCCCC","#428BCA"])
                fig = ax.get_figure()
                plt.xticks(rotation = 0,ha = 'center')
                plt.xlabel("")
                plt.ylabel('Beta',size = 12,fontweight='bold')
                ax.legend(["Beta6mo","Beta12mo"],prop = {'weight':'bold'},frameon=False, loc = "upper left")
                ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
                plt.axhline(y = 0, color = 'black')
                ax.grid()
                fig.set_size_inches(width, height)
                fig.savefig(self.outdir + "/" + name)
                plt.cla()
                plt.clf()
                plt.close('all')
        return True
    
    def rolling_sharpe(self, name = "rolling-sharpe-ratio(6-month).png",width = 11.5, height = 2.5):
        if self.is_drawable:
            days_S = 126
            days_in_one_year = 252
            if len(set(self.df.index.date)) > days_S:
                df_this = self.df.copy()
                df_this.drop("Benchmark",1,inplace = True)
                df_this = df_this.groupby([df_this.index.date]).apply(lambda x: x.tail(1))
                df_this.index = df_this.index.droplevel(1) 
                ret_strategy = np.array([self.initStrategyValue] + df_this["Strategy"].tolist())
                ret_strategy = ret_strategy[1:]/ret_strategy[:-1] - 1
                df_this["Strategy"] = ret_strategy*100
                df_this["SharpeRatio"] = float("nan")
                for i in range(days_S, len(df_this)):
                    tmp_ret = np.mean(df_this["Strategy"][(i-days_S):i]) * days_in_one_year
                    tmp_std = max(np.std(df_this["Strategy"][(i-days_S):i]) * math.sqrt(days_in_one_year),0.0001)
                    df_this.iloc[[i],[1]] = tmp_ret/tmp_std
                df_this.drop("Strategy",1,inplace = True)    
                df_this["mean"] = np.mean(df_this["SharpeRatio"])
                
                plt.figure()
                ax = df_this["SharpeRatio"].plot(color = "#F5AE29")
                ax = df_this["mean"].plot(color = "red", linestyle = "dashed")          
                fig = ax.get_figure()
                plt.xticks(rotation = 0,ha = 'center')
                plt.xlabel("")
                plt.ylabel('SharpeRatio',size = 12,fontweight='bold')
                plt.legend(["SharpeRatio","mean"],prop = {'weight':'bold'},frameon=False, loc = "upper left")
                ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
                plt.axhline(y = 0, color = 'black')
                ax.grid()
                fig.set_size_inches(width, height)
                fig.savefig(self.outdir + "/" + name)
                plt.cla()
                plt.clf()
                plt.close('all')
        return True
    
    def net_holdings(self, name = "net-holdings.png",width = 11.5, height = 2.5):
        if self.is_drawable:
            pass
        return False
    
    def leverage(self, name = "leverage.png",width = 11.5, height = 2.5):
        if self.is_drawable:
            pass
        return False
    
    def asset_allocation(self,width = 11.5, height = 2.5):
        if self.is_drawable:
            pass
        return False
    
    def return_prediction(self, name = "return-prediction.png",width = 11.5, height = 2.5):
        if self.is_drawable:
            pass
        return False

