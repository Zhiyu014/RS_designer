# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 11:09:31 2021

@author: MOMO
"""
from swmm_api.input_file import  SwmmInput, section_labels as sections
from swmm_api.input_file.sections import *
from swmm_api.input_file.sections.others import TimeseriesData

import json
import geopandas as gpd
from shapely.geometry import MultiPoint
from math import log10
from os.path import join,split,dirname
from numpy import diff,array

# This hyetograph is correct in a continous function, but incorrect with 5-min block.
def Chicago_Hyetographs(para_tuple):
    A,C,n,b,r,P,delta,dura = para_tuple
    a = A*(1+C*log10(P))
    ts = []
    for i in range(dura//delta):
        t = i*delta
        key = str(t//60).zfill(2)+':'+str(t % 60).zfill(2)
        if t <= r*dura:
            ts.append([key, (a*((1-n)*(r*dura-t)/r+b)/((r*dura-t)/r+b)**(1+n))*60])
        else:
            ts.append([key, (a*((1-n)*(t-r*dura)/(1-r)+b)/((t-r*dura)/(1-r)+b)**(1+n))*60])
    # tsd = TimeseriesData(Name = name,data = ts)
    return ts

# Generate a rainfall intensity file from a cumulative values in ICM
def Chicago_icm(para_tuple):
    A,C,n,b,r,P,delta,dura = para_tuple
    a = A*(1+C*log10(P))
    HT = a*dura/(dura+b)**n
    Hs = []
    for i in range(dura//delta+1):
        t = i*delta
        if t <= r*dura:
            H = HT*(r-(r-t/dura)*(1-t/(r*(dura+b)))**(-n))
        else:
            H = HT*(r+(t/dura-r)*(1+(t-dura)/((1-r)*(dura+b)))**(-n))
        Hs.append(H)
    tsd = diff(array(Hs))*12
    ts = []
    for i in range(dura//delta):
        t = i*delta
        key = str(t//60).zfill(2)+':'+str(t % 60).zfill(2)
        ts.append([key,tsd[i]])
    return ts

def insert_rainfall(inp,name,tss,level):
    
    inp[sections.TIMESERIES].add_obj(TimeseriesData(Name = name,data = tss))
    if level != []:
        inp[sections.CURVES] = Curve.create_section()
        inp[sections.CURVES].add_obj(Curve(Name=name,
                                            Type = 'TIDAL',points=level))
        # for out in inp[sections.OUTFALLS]:
        #     inp[sections.OUTFALLS][out].Type = 'TIDAL'
        #     inp[sections.OUTFALLS][out].Data = name
    # inp[sections.RAINGAGES]['RG'].Format = typee
    # inp[sections.RAINGAGES]['RG'].Interval = delta
    # # inp[sections.RAINGAGES]['RG'].Timeseries = name
    return inp


def long_pattern(pttn_file):
    with open(pttn_file,'r') as f:
        lines = f.readlines()
    lines = [line.strip('\n').split(',') for line in lines]
    tss = {p+'y':[[line[0],eval(line[1])*eval(lines[1][idx+1])*0.01] 
              for line in lines[2:]]
             for idx,p in enumerate(lines[0][1:])}
    tidal = [[eval(line[0].split(':')[0]),eval(line[2])] 
              for line in lines[2:]]
    return tss,tidal

def create_inp(poly,nodes,pipes):
    inp = SwmmInput()
    inp[sections.OPTIONS] = OptionSection()
    with open(join(dirname(__file__),'OPTIONS.json'),'r',encoding='utf8')as fp:
        json_data = json.load(fp)
    option = inp[sections.OPTIONS]
    for k,v in json_data.items():
        option[k] = v
    
    inp[sections.RAINGAGES] = RainGage.create_section()
    inp[sections.RAINGAGES].add_obj(RainGage('RG','INTENSITY','0:05',1,'TIMESERIES','1y'))
    # inp[sections.RAINGAGES].add_obj(RainGage('RG','VOLUME','01:00',1,'TIMESERIES',split(pttn_file)[-1].split('.')[0]+'_50y'))
    

    inp[sections.SUBCATCHMENTS] = SubCatchment.create_section()
    sub = poly.copy()
    sub['Name'],sub['RainGage'],sub['Outlet'],sub['Area'],sub['Imperv'] = sub['id'],'RG',sub['node'],sub['geometry'].area/1e4,90
    sub['Width'] = sub['area'] / sub.apply(lambda row:max([nodes.loc[row['node'],'geometry'].distance(po) for po in MultiPoint(row['geometry'].convex_hull.boundary.coords)]),axis=1)
    sub['Slope'],sub['CurbLen'] = 0.5,0.0
    sub2 = sub.set_index('Name',drop = False).drop([co for co in sub.columns if co not in ['Name','RainGage','Outlet','Area','Imperv','Width','Slope','CurbLen']],axis=1)
    subdic = sub2.to_dict('index')
    for k,v in subdic.items():
        inp[sections.SUBCATCHMENTS].add_obj(SubCatchment(**v))
        
    inp[sections.SUBAREAS] = SubArea.create_section()
    subareadic = {name:{'Subcatch':name,'N_Imperv':0.01,'N_Perv':0.1,'S_Imperv':0.0,'S_Perv':0.0,'PctZero':100} for name in sub['Name']}
    for k,v in subareadic.items():
        inp[sections.SUBAREAS].add_obj(SubArea(**v))
    
    inp[sections.INFILTRATION] = Infiltration.create_section()
    infildic = {name:{'Subcatch':name,'MaxRate':10000,'MinRate':10000,'Decay':4.0,'DryTime':7.0,'MaxInf':0.0} for name in sub['Name']}
    for k,v in infildic.items():
        inp[sections.INFILTRATION].add_obj(InfiltrationHorton(**v))
    
    inp[sections.JUNCTIONS] = Junction.create_section()
    node = nodes[nodes['node_type']=='node'].copy()
    node['Name'] = node['name']
    node['Elevation'] = node['invelev']
    node['MaxDepth'] = node['ctrlelev'].astype('float') - node['invelev'].astype('float')  if 'ctrlelev' in node.columns else node['elev'].astype('float') - node['invelev'].astype('float')
    node['InitDepth'], node['SurDepth'], node['Aponded'] = 0,0,500
    juncdict = node.drop([co for co in node.columns if co not in ['Name' ,'Elevation' , 'MaxDepth'  , 'InitDepth' , 'SurDepth',   'Aponded']],axis=1).to_dict('index')
    for k,v in juncdict.items():
        inp[sections.JUNCTIONS].add_obj(Junction(**v))
        
    
    inp[sections.OUTFALLS] = Outfall.create_section()
    outfall = nodes[nodes['node_type'] == 'outfall'].copy()
    outfall['Name'],outfall['Elevation'] = outfall['name'],outfall['invelev']
    # outfall['Type'],outfall['Data'],outfall['FlapGate'] = 'TIDAL', split(pttn_file)[-1].split('.')[0], True
    # outfall['Type'],outfall['Data'],outfall['FlapGate'] = 'FIXED', 3.75, True
    outfall['Type'] = 'FREE'
    outfalldict = outfall.drop([co for co in outfall.columns if co not in ['Name' , 'Elevation' , 'Type',   'Data'  ,  'FlapGate']],axis=1).to_dict('index')
    for k,v in outfalldict.items():
        inp[sections.OUTFALLS].add_obj(Outfall(**v))
    
    
    inp[sections.CONDUITS] = Conduit.create_section()
    conds = pipes.copy()
    conds['Name'] = conds['us_node'].astype(str) +'-'+ conds['ds_node'].astype(str)
    conds['FromNode'],conds['ToNode'] = conds['us_node'],conds['ds_node']
    conds['Length'],conds['Roughness'] = conds['length'],0.022
    conds['InOffset'] = conds.apply(lambda row:row['us_depth']-nodes.loc[row['us_node'],'invelev'],axis=1)
    conds['InitFlow'],conds['MaxFlow'] = 0.0,0
    conds['OutOffset'] = conds.apply(lambda row:row['ds_depth']-nodes.loc[row['ds_node'],'invelev'],axis=1)
    condict = conds.drop([co for co in conds.columns if co not in ['Name' ,   'FromNode' ,  'ToNode'   ,  'Length'  ,   'Roughness',  'InOffset' ,  'OutOffset' , 'InitFlow' ,  'MaxFlow']],axis=1).to_dict('index')
    for k,v in condict.items():
        inp[sections.CONDUITS].add_obj(Conduit(**v))
        
    
    inp[sections.XSECTIONS] = CrossSection.create_section()
    xsec = pipes.copy()
    xsec['Link'],xsec['Shape'],xsec['Geom1'] = xsec['us_node'].astype(str) + '-' + xsec['ds_node'].astype(str),'CIRCULAR',xsec['diameter']/1000
    xsecdict = xsec.drop([co for co in xsec.columns if co not in ['Link','Shape','Geom1']],axis=1).to_dict('index')
    for k,v in xsecdict.items():
        inp[sections.XSECTIONS].add_obj(CrossSection(**v))
    
    
    inp[sections.TIMESERIES] = Timeseries.create_section()
    # para_tuple = (9.5981,0.846,0.656,7,5,120,0.405) #(A,c,n,b,delta,dura,r)
    # P = [1,2,5,10,20,50,100]
    # for p in P:
    #     ts = Chicago_Hyetographs(para_tuple,p)
    #     inp[sections.TIMESERIES].add_obj(TimeseriesData(Name = str(p)+'y',data = ts))
    
    # TODO Long-duration rainfall pattern
    # tss,tidal = long_pattern(pttn_file)
    # for k,v in tss.items():        
        # inp[sections.TIMESERIES].add_obj(TimeseriesData(Name = split(pttn_file)[-1].split('.')[0]+'_'+k,
                                                    # data = v))
    
    # inp[sections.CURVES] = Curve.create_section()
    # inp[sections.CURVES].add_obj(Curve(Name=split(pttn_file)[-1].split('.')[0],
    #                                     Type = 'TIDAL',points=tidal))
    
    
    inp[sections.REPORT] = ReportSection()
    inp[sections.REPORT]['SUBCATCHMENTS'] = 'ALL'
    inp[sections.REPORT]['NODES'] = 'ALL'
    inp[sections.REPORT]['LINKS'] = 'ALL'
    
    
    inp[sections.MAP] = MapSection()
    bounds = (sub.bounds['minx'].min(),sub.bounds['miny'].min(),sub.bounds['maxx'].max(),sub.bounds['maxy'].max())
    inp[sections.MAP] = {'DIMENSIONS':list(bounds),'UNITS':'Meters'}
    
    inp[sections.COORDINATES] = Coordinate.create_section()
    coord = gpd.GeoDataFrame()
    coord['Node'],coord['x'],coord['y'] = nodes['name'],nodes['geometry'].x,nodes['geometry'].y
    coordict = coord.to_dict('index')
    for k,v in coordict.items():
        inp[sections.COORDINATES].add_obj(Coordinate(**v))
           
    
    inp[sections.POLYGONS] = Polygon.create_section()
    polygons = gpd.GeoDataFrame(geometry = sub.boundary)
    polygons['Subcatch'] = sub['id']
    polygons = polygons.explode().reset_index(drop=True,level=None)
    polygons['polygon'] = polygons['geometry'].apply(lambda poly: [list(node[:2]) for node in poly.coords])
    polydict = polygons.drop('geometry',axis=1).to_dict('index')
    for k,v in polydict.items():
        inp[sections.POLYGONS].add_obj(Polygon(**v))
    
    inp[sections.SYMBOLS] = Symbol.create_section()
    minx,maxy = sub.bounds['minx'].min(),sub.bounds['maxy'].max()
    sym = {'Gage':'RG','x':minx,'y':maxy}
    inp[sections.SYMBOLS].add_obj(Symbol(**sym))
    return inp

# def add_river(inp,riverpoly,riverouts):

#     outfall = riverouts.copy()
#     outfall['Name'],outfall['Elevation'] = outfall['name'],outfall['invelev']
#     outfall['Type'],outfall['Data'],outfall['FlapGate'] = 'TIDAL', pttn_file.split('/')[-1].split('.')[0], True
#     # outfall['Type'],outfall['Data'],outfall['FlapGate'] = 'FIXED', 3.75, True
#     # outfall['Type'] = 'FREE'
#     outfalldict = outfall.drop([co for co in outfall.columns if co not in ['Name' , 'Elevation' , 'Type',   'Data'  ,  'FlapGate']],axis=1).to_dict('index')
#     for k,v in outfalldict.items():
#         inp[sections.OUTFALLS].add_obj(Outfall(**v))    
    

def create_model(nodes,pipes,sub,inp_path,
                 river_poly = None, riverpoints = None):
    # sub = gpd.read_file(subcatch_file)
    nodes = nodes.set_index('name',drop = False)
    # pipes = gpd.read_file(pipe_file)
    # riverpoly = gpd.read_file(river_poly) if river_poly != None else None
    # riverouts = gpd.read_file(riverpoints) if riverpoints != None else None
    inp = create_inp(sub, nodes, pipes)
    inp.write_file(inp_path)



