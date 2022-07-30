# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 18:07:11 2022

@author: chong
"""

from swmm_api import swmm5_run,read_inp_file,read_rpt_file
import geopandas as gpd
from datetime import timedelta,datetime
from numpy import nan
import os.path
# from pyswmm import Simulation

def eval_pipes(pipe_file,diam):
    pipes = gpd.read_file(pipe_file)
    pipes['length'] = pipes.length
    quanti = pipes.groupby(diam)['length'].sum()
    quanti = gpd.pd.DataFrame(quanti).reset_index()
    return quanti
    
def get_inp(inp_file):
    inp = read_inp_file(inp_file)
    return inp

def get_simulate_file(inp,inp_file,rain_ts,kind,runoff_co,field):
    inp['RAINGAGES']['RG'].Timeseries = rain_ts
    inp['RAINGAGES']['RG'].Format = kind
    
    inp['RAINGAGES']['RG'].Interval = field['interval']
    
    st = (inp['OPTIONS']['START_DATE'],inp['OPTIONS']['START_TIME'])
    st_time = datetime(st[0].year,st[0].month,st[0].day,st[1].hour,st[1].minute)
    end_time = st_time + timedelta(minutes = field['duration'] + 120)   # Extra 2-hour simulation
    inp['OPTIONS']['END_DATE'] = end_time.date()
    inp['OPTIONS']['END_TIME'] = end_time.time()
    
    if field['is_tidal']:
        for k,v in inp['OUTFALLS'].items():
            v.Type = 'TIDAL'
            v.Data = rain_ts
    else:
        for k,v in inp['OUTFALLS'].items():
            v.Type = 'FREE'
            v.Data = nan            
            
    for k,v in inp['SUBCATCHMENTS'].items():
        v.Imperv = runoff_co*100

    file = os.path.splitext(inp_file)[0] + '_' + rain_ts +'.inp'
    inp.write_file(file)
    return file

    # rpt_file,_ = swmm5_run(file,init_print=True,create_out=False)

def eval_rpt(rpt_file,inp):
    rpt = read_rpt_file(rpt_file)
    lf = rpt.link_flow_summary
    nf = rpt.node_flooding_summary
    
    conds = inp['CONDUITS'].frame
    fulls = lf[lf['Max/_Full_Depth']==1].index
    full_length_perc = round(conds.loc[fulls,'Length'].sum()/conds['Length'].sum()*100,2)
    
    if nf is not None:
        flood_perc = round(len(nf)/len(inp['JUNCTIONS'])*100,2)
        flood_high_perc = round(len(nf[nf['Maximum_Ponded_Depth_Meters']>0.15])/len(inp['JUNCTIONS'])*100,2)
        
        flood_dura_avg = round(nf['Hours_Flooded'].mean(),2)
        flood_dura_max = round(nf['Hours_Flooded'].max(),2)
    else:
        flood_perc,flood_high_perc,flood_dura_avg,flood_dura_max = 0,0,'-','-'
    
    res = gpd.pd.DataFrame(columns=['满载管道长度占比/%','内涝节点占比/%',
                                    '积水深度超过15 cm节点占比/%','平均积水时间/hr',
                                    '最大积水时间/hr'])
    ts = os.path.split(rpt_file)[-1].split('_')[-1].split('.')[0]
    res.loc[ts]=[full_length_perc,flood_perc,flood_high_perc,
                      flood_dura_avg,flood_dura_max]
    return res