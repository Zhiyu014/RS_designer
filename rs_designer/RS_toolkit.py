# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 15:15:41 2022

@author: chong
"""
from os import getcwd,mkdir,system
from os.path import join,exists,split as osplit
from osgeo import gdal
import geopandas as gpd
import networkx as nx
from numpy import array,std,sqrt,argmin
from pandas import concat
from shapely.geometry import Point,MultiPoint,LineString,Polygon
from shapely.ops import split,nearest_points,unary_union
from scipy.spatial import Voronoi
from scipy.special import comb
from itertools import combinations
import math
from numpy import piecewise
import random
import json
# from SWMM_writer2 import main as write_inp


def read_resample(pipes,river,dis):
    '''
    Extract drainage nodes and pipes from the road network.
    
    Parameters
    ----------
    pipes : GeoDataFrame
        GeoPandas file of the road center line.
    river : GeoDataFrame
        GeoPandas file of the river polygon file.
    dis : int
        Pipe length between nodes. (m)

    Returns
    -------
    nodes : GeoDataFrame
        GeoPandas file of the extracted nodes.
    pipes : GeoDataFrame
        GeoPandas file of the extracted pipes.
    '''
    # pipes = gpd.read_file(pipe_file)
    pipes = pipes.drop([co for co in pipes.columns if co !='geometry'],axis=1)
    if river is not None:
        riverpolys = unary_union(river['geometry'])
        pipes['geometry'] = pipes['geometry'].apply(lambda pipe:pipe.difference(riverpolys)
                                                    if pipe.intersects(riverpolys) else pipe)
            
    pipes = pipes.drop(pipes[pipes['geometry'].is_empty].index)
    pipes = pipes.explode().reset_index(drop=True,level=None)
        
    pipe_shape = [re for pipe in pipes['geometry'] 
                  for re in split(pipe,MultiPoint(pipe.coords))]
    dis = sum([pipe.length for pipe in pipe_shape])/len(pipe_shape) if dis is None else dis
    pipes['coords'] = pipes['geometry'].apply(lambda pipe: 
                                              MultiPoint([Point(pipe.interpolate(x).coords[0][:2])
                                                          for x in range(0,int(pipe.length),int(dis))]
                                                         +[Point(pipe.coords[-1][:2])]))
    pipes = pipes.drop(pipes[pipes['coords'].apply(len)==1].index)
    pipes['geometry'] = pipes['coords'].apply(lambda cors:LineString(cors))
    pipes = pipes.drop(pipes[pipes['geometry'].length==0].index).reset_index(drop=True,level=None)
    
    pos = {}
    m = 1
    for i,coords in enumerate(pipes['coords']):
        for coord in [coords[0],coords[-1]]:
            crd = coord.coords[0]
            crd = (round(crd[0],6),round(crd[1],6))
            if crd not in pos.values():
                po_name = 'A' + str(m)
                m += 1
                pos.update({po_name:crd})
        if len(coords) == 2:
            continue
        else:
            n = 1
            for coord in coords[1:-1]:
                crd = coord.coords[0]
                crd = (round(crd[0],6),round(crd[1],6))
                n += 1
                po_name = 'B'+str(i)+'-'+str(n)
                pos.update({po_name:crd})
    crdss = [[(round(coord.coords[0][0],6),round(coord.coords[0][1],6)) 
              for coord in coords] for coords in pipes['coords']]    
    
    pipes['geometry'] = [LineString([Point(crd) for crd in crds]) for crds in crdss]
    pipes['geometry'] = pipes['geometry'].apply(lambda line:
                                                split(line,MultiPoint(line.coords)))
    pipes = pipes.explode().reset_index(drop=True,level=None)
    pos_rev = {v:k for k,v in pos.items()}
    pipes['coords'] = pipes['geometry'].apply(lambda pipe: 
                                              (pos_rev[pipe.coords[0]],pos_rev[pipe.coords[-1]]))    
    pipes = pipes[pipes['coords'].apply(lambda coords:True 
                                        if len(set(coords))==2 else False)].reset_index(drop=True,level=None)
    pipes['length'] = pipes['geometry'].length
    pipes['us_node'] = pipes['coords'].apply(lambda co:co[0])
    pipes['ds_node'] = pipes['coords'].apply(lambda co:co[-1])
    pipes = pipes.drop(['coords','length'],axis=1)
    nodes = [(k,*v,Point(v)) for k,v in pos.items()]
    nodes = gpd.GeoDataFrame(nodes,columns=['name','x','y','geometry'])
    # X = nx.from_pandas_edgelist(pipes,'us_node','ds_node',['length','geometry'])
    # for node in X.nodes():
    #     X.nodes[node].update(pos = pos[node], geometry = Point(pos[node]))    
    # return X
    return nodes,pipes

def read_pipes(pipes,river):

    pipes = pipes.drop([co for co in pipes.columns if co !='geometry'],axis=1)
    if river is not None:
        riverpolys = unary_union(river['geometry'])
        pipes['geometry'] = pipes['geometry'].apply(lambda pipe:pipe.difference(riverpolys)
                                                    if pipe.intersects(riverpolys) else pipe)
            
    pipes = pipes.drop(pipes[pipes['geometry'].is_empty].index)
    pipes = pipes.explode().reset_index(drop=True,level=None)
    pipes['coords'] = pipes['geometry'].apply(lambda pipe: 
                                              MultiPoint([Point(pipe.coords[i][:2]) for i in range(len(pipe.coords))]))
    pipes = pipes.drop(pipes[pipes['geometry'].length==0].index).reset_index(drop=True,level=None)

    pos = {}
    m = 1
    for i,coords in enumerate(pipes['coords']):
        for coord in [coords[0],coords[-1]]:
            crd = coord.coords[0]
            crd = (round(crd[0],6),round(crd[1],6))
            if crd not in pos.values():
                po_name = 'A' + str(m)
                m += 1
                pos.update({po_name:crd})
        if len(coords) == 2:
            continue
        else:
            n = 1
            for coord in coords[1:-1]:
                crd = coord.coords[0]
                crd = (round(crd[0],6),round(crd[1],6))
                n += 1
                po_name = 'B'+str(i)+'-'+str(n)
                pos.update({po_name:crd})
    crdss = [[(round(coord.coords[0][0],6),round(coord.coords[0][1],6)) 
              for coord in coords] for coords in pipes['coords']]  
    crdss = [sorted(set(crds),key=crds.index) for crds in crdss]
    ind = [i for i,crds in enumerate(crdss) if len(crds)<=1]
    pipes = pipes.drop(ind).reset_index(drop=True,level=None)
    crdss = [crds for i,crds in enumerate(crdss) if i not in ind]
    
    pipes['geometry'] = [LineString([Point(crd) for crd in crds]) for crds in crdss]  
    pos_rev = {v:k for k,v in pos.items()}
    pipes['coords'] = pipes['geometry'].apply(lambda pipe: 
                                              (pos_rev[pipe.coords[0]],pos_rev[pipe.coords[-1]]))              
    pipes['length'] = pipes['geometry'].length
    pipes['us_node'] = pipes['coords'].apply(lambda co:co[0])
    pipes['ds_node'] = pipes['coords'].apply(lambda co:co[-1])
    pipes = pipes.drop(['coords'],axis=1)
    nodes = [(k,*v,Point(v)) for k,v in pos.items()]
    nodes = gpd.GeoDataFrame(nodes,columns=['name','x','y','geometry'])    
    return nodes,pipes


def voronoi_frames(points,clip='extent',radius=10000):
    mp = MultiPoint(points)
    bufferpoints = mp.convex_hull.buffer(radius)
    points.extend(bufferpoints.exterior.coords)
    vor = Voronoi(array(points))
    vertices = vor.vertices
    regions = vor.regions
    point_region = vor.point_region
    Polygons = gpd.GeoDataFrame(geometry = 
                                [Polygon(vertices[regions[i]]) for i in point_region])
    if clip == 'extent':
        clip = MultiPoint(points).envelope
    Polygons['geometry'] = Polygons['geometry'].intersection(clip)
    Polygons = Polygons.drop([i for i in Polygons.index if i>=len(mp)])
    return Polygons

    

def create_subcatch(nodes,pipes,road,river,
                    region = None, has_river=False, dis = None,radius=10000):
    '''
    Delineate subcatchments using voronoi diagram.

    Parameters
    ----------
    nodes : GeoDataFrame
        Drainage nodes.
    pipes : GeoDataFrame
        Drainage pipes.
    road : GeoDataFrame
        Road polygon.
    river : GeoDataFrame
        River polygon.
    region : GeoDataFrame, optional
        Region polygon. The default is None.
    has_river : bool, optional
        If the delineation considers dispersed drainage area nearby the river. The default is False.
    dis : int, optional
        Riverside node spacing. The default is None.
    radius : int, optional
        The buffer radius of the convex hull in the voronoi diagram. The default is 10000.

    Returns
    -------
    node_poly : GeoDataFrame
        Delineated subcatchments.
    river_poly : GeoDataFrame
        Dispersed riverside drainage area.
    riverpoints : GeoDataFrame
        Extracted riverside nodes.

    '''
    # nodes = gpd.read_file(node_file)
    # pipes = gpd.read_file(pipe_file)
    road = unary_union(road['geometry'])    
    river = unary_union(river['geometry'])
    points = {k:v.coords[0] for k,v in zip(nodes['name'],nodes['geometry']) if v.distance(river) > 1}
    points_total = points.copy()    
    if has_river:
        riverline = river.boundary
        dis = sum([length for length in pipes.geometry.length])/len(pipes) if dis is None else dis
        if riverline.geom_type == 'LineString':
            riverpoints = MultiPoint([riverline.interpolate(x) 
                                      for x in range(0,int(riverline.length),int(dis))])
        else:
            riverpoints = MultiPoint([line.interpolate(x) 
                                      for line in riverline 
                                      for x in range(0,int(line.length),int(dis))])
        river_pos = {k:(po.x,po.y) for k,po in enumerate(riverpoints)}
        riverpoints = gpd.GeoDataFrame(geometry = [Point(po) for po in river_pos.values()])
        riverpoints['name'] = list(river_pos.keys())
        points_total.update(river_pos)
    points_value = list(points_total.values())
    
    region = unary_union(region['geometry']) if region is not None \
    else MultiPoint(points_value).convex_hull.buffer(radius)
    
    regions_df = voronoi_frames(points_value,clip=region,radius=radius)
    node_poly = regions_df.drop([i for i in regions_df.index if i>=len(points)])
    node_poly['geometry'] = node_poly['geometry'].intersection(region)
    
    river_no_road = river.difference(road)
    node_poly['geometry'] = node_poly['geometry'].difference(river_no_road)
    node_poly['node'] = points.keys()
    node_poly = node_poly.drop(node_poly[node_poly['geometry'].is_empty].index)
    
    if has_river:
        river_poly = regions_df.drop([i for i in regions_df.index if i<len(points)])
        river_poly['geometry'] = river_poly['geometry'].intersection(region)
        river_poly['node'] = river_pos.keys()
        river_poly = river_poly.drop(river_poly[river_poly['geometry'].is_empty].index)

    # road diff parts comcated to node
        pos_rev = {v:k for k,v in points.items() if Point(v).distance(river) > 1}
        mp = MultiPoint(list(pos_rev.keys()))
        road_diff = gpd.GeoDataFrame(geometry = [ro for ro in river_poly.intersection(road) 
                                                 if ro.is_empty == False])
        road_diff = road_diff.explode().reset_index(drop=True,level=None)
        road_diff['node'] = road_diff['geometry'].apply(lambda geo: 
                                                        pos_rev[nearest_points(geo,mp)[1].coords[0]])    
        node_poly = concat([node_poly,road_diff],axis=0)    

    # repair river polys outside the road
        river_poly['geometry'] = river_poly['geometry'].difference(road)
        river_poly['geometry'] = river_poly['geometry'].difference(river)
        river_poly = river_poly.drop(river_poly[river_poly['geometry'].is_empty].index)
        river_poly = river_poly.explode().reset_index(drop=True,level=None)
        river_na = river_poly[river_poly['geometry'].apply(lambda geo: 
                                                           river.distance(geo)>road.distance(geo))].index
        for j in river_na:
            _,minp = nearest_points(river_poly['geometry'][j],mp)
            minpoint = pos_rev[minp.coords[0]]
            river_poly.loc[j,'node'] = minpoint
            node_poly = node_poly.append(river_poly.loc[j,:],ignore_index=True)
            river_poly = river_poly.drop(j)
    
    node_poly = node_poly.dissolve(by = 'node')
    node_poly['node'] = node_poly.index
    node_poly = node_poly.reset_index(drop=True,level=None)
    node_poly = node_poly.explode().reset_index(drop=True,level=None)
    node_poly = node_poly.drop(node_poly[node_poly.geom_type!='Polygon'].index)         
    node_poly = node_poly.reset_index(drop=True,level=None)
    node_poly['id'] = node_poly.index
    node_poly['id'] = node_poly['id'].apply(lambda x: 'S' + str(x+1))
    node_poly['area'] = node_poly['geometry'].area
    # node_poly['outfall'] = node_poly['node'].apply(lambda x: X.nodes[x]['outfall'] if isinstance(x,str) else '')   
    if has_river:
        river_poly = river_poly.drop(river_poly[river_poly.geom_type!='Polygon'].index)
        river_poly = river_poly.reset_index(drop=True,level=None)
        river_poly['id'] = river_poly.index
        river_poly['id'] = river_poly['id'].apply(lambda x: 'R' + str(x+1))
        river_poly['area'] = river_poly['geometry'].area
        return node_poly,river_poly,riverpoints
    else:
        return node_poly,None,None

def find_node(nodes,subcatch):
    '''
    Find the outlet node of each subcatchment using spatial join.

    Parameters
    ----------
    nodes : GeoDataFrame
        Drainage nodes.
    subcatch : GeoDataFrame
        Sub-catchments.

    Returns
    -------
    subcatch : GeoDataFrame
        Sub-catchments.

    '''
    subcatch = subcatch.drop([co for co in subcatch.columns if co!='geometry'],axis=1)
    subcatch['id'] = subcatch.index
    subcatch['id'] = subcatch['id'].apply(lambda x: 'S' + str(x+1))
    subcatch.crs = nodes.crs
    nodes = nodes[nodes['node_type']=='node'].reset_index(drop=True,level=None)
    subcatch = gpd.sjoin(subcatch,nodes,how='left').drop_duplicates(['id'],keep='first')
    na_polys = subcatch[subcatch['name'].isna()]
    
    # TODO join polys with nodes
    for ind,poly in zip(na_polys.index,na_polys['geometry']):
        dist = 100
        candis = nodes[nodes.intersects(poly.buffer(dist))][nodes['node_type']=='node']
        while len(candis) == 0:
            dist += 100
            if dist >= 1000:
                break
            candis = nodes[nodes.intersects(poly.buffer(dist))]
        if len(candis) == 0:
            continue
        candis['dist'] = candis['geometry'].distance(poly)
        subcatch.loc[ind,'name'] = candis.loc[candis['dist'].idxmin(),'name']
    subcatch = subcatch[~subcatch['name'].isna()].reset_index(drop=True,level=None)
    subcatch = subcatch.rename(columns={'name':'node'})
    subcatch['area'] = subcatch.geometry.area
    subcatch = subcatch[['id','node','area','geometry']]
    return subcatch

    
def update_area(nodes,Polygons):
    '''
    Set the catchment area value to the drainage nodes.

    Parameters
    ----------
    nodes : GeoDataFrame
        Drainage nodes.
    Polygons : GeoDataFrame
        Subcatchments.

    Returns
    -------
    nodes : GeoDataFrame
        Drainage nodes.

    '''
    # nodes = gpd.read_file(node_file)
    nodes['lo_area'] = nodes['name'].apply(lambda node:sum([area 
                                        for poly_node,area in zip(Polygons['node'],Polygons['area']) 
                                        if poly_node==node]))
    return nodes



def create_out(nodes,pipes,river,dis=None):
    '''
    Identify outfalls from the drainage nodes based on the distance to the river.

    Parameters
    ----------
    nodes : GeoDataFrame
        Drainage nodes.
    pipes : GeoDataFrame
        Drainage pipes.
    river : GeoDataFrame
        River polygon.
    dis : float, optional
        The maximum distance from outfalls to the river. The default is None.

    Returns
    -------
    nodes : GeoDataFrame
        Drainage nodes.
    pipes : GeoDataFrame
        Drainage pipes.

    '''
    # nodes = gpd.read_file(node_file)
    # pipes = gpd.read_file(pipe_file)
    rivers = river.unary_union.boundary
    dis = 1 if dis==None else dis
    candi_outs = [node for node,geo in zip(nodes['name'],nodes['geometry']) 
                  if geo.distance(rivers)<dis]
    '''
    check connected outfalls
    '''    
    ud_nodes = list(pipes['us_node'])+list(pipes['ds_node'])
    candi_outs = [out for out in candi_outs if ud_nodes.count(out)==1]
    
    nodes = nodes.replace(candi_outs,['O'+str(i+1) for i in range(len(candi_outs))])
    pipes = pipes.replace(candi_outs,['O'+str(i+1) for i in range(len(candi_outs))])
    nodes['node_type'] = nodes['name'].apply(lambda na:'outfall' if na.startswith('O') else 'node')
    return nodes,pipes
    
    
    # 检查outfall是否连接了多个节点
    # for out in outfalls:
    #     if len(list(X.neighbors(out)))>1:
    #         neighbors = {k:v['length'] for k,v in X[out]}
    #         node = max(neighbors)
    #         neighbors.pop(node)
    #         for k in neighbors:
    #             X.add_edge(k,out,geometry = LineString([Point(pos[k]),Point(pos[node])]),
    #                        length = Point(pos[k]).distance(Point(pos[node])))
    #         X.remove_from_edges([(k,out) for k in neighbors])
    #     else:
    #         pass

def formulate(nodes,pipes,
              nodefield=['name','node_type','lo_area'],
              linefield=['us_node','ds_node','geometry.length'],
              di=False):
    '''
    Formulate the networkx Graph class from the GeoDataFrames.

    Parameters
    ----------
    nodes : GeoDataFrame
        Drainage nodes.
    pipes : GeoDataFrame
        Drainage pipes.
    di : bool, optional
        If a directed graph is returned. The default is False.

    Returns
    -------
    X : Graph/Digraph
        The drainage network.

    '''
    # nodes = gpd.read_file(node_file)
    name,node_type,lo_area = nodefield
    nodes = nodes.set_index(name,drop=True)
    nodes['node_type'] = nodes[node_type]
    nodes['lo_area'] = nodes[lo_area]
    # pipes = gpd.read_file(pipe_file)
    us,ds,length = linefield
    pipes['length'] = pipes[length] if length!= 'geometry.length' else pipes['geometry'].length
    X = nx.from_pandas_edgelist(pipes,us,ds,[col for col in pipes.columns],
                                create_using = nx.DiGraph() if di else None)
    for col in nodes.columns:
        nx.set_node_attributes(X,nodes[col].to_dict(),name=col)
    return X
    
    
def break_pipes(X):
    '''
    Delineate the drainage paths based on the distance to outfalls and drainage areas.

    Parameters
    ----------
    X : Graph/Digraph
        The drainage network.

    Returns
    -------
    X : Graph
        The drainage network.
    outfalls : list
        The outfall list.

    '''
    outfalls = [node for node in X.nodes if X.nodes[node]['node_type'] == 'outfall']
    for out in outfalls:
        if len(list(X.neighbors(out)))>1:
            neighbors = {k:v['geometry'].length for k,v in X[out]}
            node = max(neighbors)
            neighbors.pop(node)
            for k in neighbors:
                X.add_edge(k,out,geometry = LineString([X.nodes[k]['geometry'],X.nodes[node]['geometry']]))
            X.remove_from_edges([(k,out) for k in neighbors])
        else:
            pass    
    nulls = [v for c in nx.connected_components(X)
             if c.issubset(set(outfalls))
             for v in c]
    for o in nulls:
        outfalls.remove(o)
    nulls += [v for c in nx.connected_components(X)
              if c.intersection(set(outfalls))==set()
              for v in c]
    X.remove_nodes_from(nulls)
    for u,v in X.edges():
        X[u][v]['length'] = X[u][v]['geometry'].length    
    
    
    subxs = [nx.subgraph(X,c) for c in nx.connected_components(X)]

    d_outs = []

    for subx in subxs:
        outfall = set(outfalls).intersection(set(subx.nodes))
        length,paths = nx.multi_source_dijkstra(subx,outfall,weight = 'length')
        max_node,max_outdis = max(length.items(),key = lambda x:x[1])        
        
        node_set = {k:v[0] for k,v in paths.items()}
        out_nodes = {out:[n for n in node_set if node_set[n]==out] for out in outfall}
        max_out_nodes = {k:max([n for n in v],key = lambda x:length[x]) for k,v in out_nodes.items()}

        dfs_trees = {k:paths[v] for k,v in max_out_nodes.items()}
        tree_nodes = {n for v in dfs_trees.values() for n in v}
        tree_areas = {k:sum([subx.nodes[i]['lo_area'] for i in v])
                                for k,v in dfs_trees.items()}
        tree_forward_nodes = dfs_trees.copy()
        tree_dist = {k:length[k] for k in tree_nodes}
        
        # tree_stopped_nodes = {out:[] for out in outfall}
        # i = {out:1 for out in outfall}
        while set(tree_nodes)!=set(subx.nodes):
            out = min(tree_areas,key=tree_areas.get)
            sx = nx.subgraph(subx,dfs_trees[out])
            dfs_nodes = set([n for node in tree_forward_nodes[out] 
                                for n in nx.neighbors(subx,node)
                             if nx.dijkstra_path_length(sx,out,node,'length')+\
                                 subx[node][n]['length']<= max_outdis 
                             and n not in outfall]) - set(dfs_trees[out])            
                
            # i[out] += 1
            # dfs_nodes = set(nx.dfs_postorder_nodes(subx,out,i[out]))-set(dfs_trees[out])
            # dfs_nodes = set([n for node in tree_forward_nodes[out] 
            #                  for n in nx.neighbors(subx,node)]) - set(dfs_trees[out])
            # dfs_nodes = {node for node in dfs_nodes 
            #              if set(nx.dijkstra_path(subx,out,node)).isdisjoint(set(tree_stopped_nodes[out]))}
            if dfs_nodes.issubset(tree_nodes):
                tree_areas[out] = 1e10
            tree_forward_nodes[out] = dfs_nodes - set(tree_nodes)
            dfs_trees[out] += [v for v in tree_forward_nodes[out]]
            tree_areas[out] += sum([X.nodes[v]['lo_area'] 
                                    for v in tree_forward_nodes[out]])
            sx = nx.subgraph(subx,dfs_trees[out])
            for no in tree_forward_nodes[out]:
                tree_dist[no] = nx.dijkstra_path_length(sx, out, no,'length')
            # else:
            #     for v in dfs_nodes:
            #         if v not in tree_nodes:
            #             dfs_trees[out] += [v]
            #             tree_areas[out] += X.nodes[v]['lo_area']
                    # else:
                    #     tree_stopped_nodes[out] += [v]
            tree_nodes = tree_nodes.union(dfs_nodes)
            
            if min(tree_areas.values()) == 1e10:
                break

        
        sub_set = {out:
                   set(nx.single_source_dijkstra(subx,
                                                 out,
                                                 cutoff=max_outdis,
                                                 weight='length')[0].keys()) 
                   for out in outfall}
        
        dis_set = {k:set(v).intersection(sub_set[k]) for k,v in dfs_trees.items()}
        node_set = {n:k for k,v in dis_set.items() for n in v}
        single_nodes = set(subx.nodes).difference(set(node_set))
        # snx = nx.subgraph(subx,single_nodes.keys())
        single_nodes_outfall = {}
        for k in single_nodes:
            single_nodes_neighbor = set(nx.neighbors(subx,k))
            single_nodes_neighbors = single_nodes_neighbor.intersection(set(node_set.keys()))
            while single_nodes_neighbors==set():
                single_nodes_neighbor = {no for k in single_nodes_neighbor
                                         for no in nx.neighbors(subx,k)}
                single_nodes_neighbors = single_nodes_neighbor.intersection(set(node_set.keys()))
            single_nodes_outfall[k] = node_set[min({no: nx.dijkstra_path_length(subx,k,no,weight='length')+tree_dist[no]
                                       for no in single_nodes_neighbors})]
            
        # single_nodes_neighbors = {node:
        #                           {no:v for no,v in 
        #                             nx.single_source_dijkstra_path_length(subx,
        #                                                                   node,
        #                                                                   weight='length',
        #                                                                   cutoff=length[node]).items()
        #                             if no in node_set}
        #                           for node in single_nodes}
        # single_nodes_outfall = {k: node_set[min(v, key=v.get)]
        #                           for k,v in single_nodes_neighbors.items()}
        node_set.update(single_nodes_outfall)
        
        for node in subx.nodes():
            X.nodes[node].update(outfall = node_set[node])
        
        
        # 合并小范围排口为双排口
        out_area = {out:sum([subx.nodes[n]['lo_area'] 
                              for n in subx.nodes 
                              if subx.nodes[n]['outfall']==out])
                    for out in outfall}

        s_out = [k for k,v in out_area.items() 
                      if v<sum(out_area.values())/len(out_area)/2]
        
        s_nodes = {out:[node for node in subx.nodes 
                            if subx.nodes[node]['outfall']==out]
                          for out in s_out}
        s_neighbor = {out:{subx.nodes[n]['outfall']
                            for node in nodes 
                            for n in subx.neighbors(node) 
                            if n not in nodes}
                      for out,nodes in s_nodes.items()}
        d_outs.extend([{k,min({o:out_area[o] for o in v})} 
                  for k,v in s_neighbor.items()])
        

            
                
    '''
    adjust outfall positions. delete repeated outfalls
    '''
    out_remove = []
    for out in outfalls:
        if X.nodes[list(X[out])[0]]['outfall']!=out:
            out_remove.append(out)
            for node in [node for node in X.nodes if X.nodes[node]['outfall'] == out]:
                X.nodes[node].update(outfall = X.nodes[list(X[out])[0]]['outfall'])

    
            
    out_remove =list(set(out_remove))
    X.remove_nodes_from(out_remove)
    for out in out_remove:
        outfalls.remove(out)
        
    '''
    remove delineated pipes (with different outfalls)
    '''
    # remove_list = [(u,v) for u,v in X.edges()
    #                if X.nodes[u]['outfall'] != X.nodes[v]['outfall']]
    remove_list = [(u,v) for u,v in X.edges()
                    if X.nodes[u]['outfall'] != X.nodes[v]['outfall'] and 
                    {X.nodes[u]['outfall'],X.nodes[v]['outfall']} not in d_outs]
    X.remove_edges_from(remove_list)
    
    '''
    remove unreachable nodes and update the outfall and distance of nodes
    '''
    length,paths = nx.multi_source_dijkstra(X,set(outfalls),weight = 'length')   
    nodes_remove = [node for node in X.nodes if node not in length]
    X.remove_nodes_from(nodes_remove)
    for node in X.nodes:
        X.nodes[node].update(outfall = paths[node][0], outdis = length[node])
        
        
    '''
    change the outfall name in double-out systems
    '''
    double_outs = {out for outs in d_outs for out in outs}
    for node in X.nodes:
        if X.nodes[node]['outfall'] in double_outs:
            outs = [outs for outs in d_outs if X.nodes[node]['outfall'] in outs][0]
            X.nodes[node].update(outfall = ''.join(outs))
        X.nodes[node].update(region = X.nodes[node]['outfall'].replace('O','SA')) 

    # singles = [node for node in X.nodes() if nx.has_path(X,node,X.nodes[node]['outfall'])==False] # For debugging

    return X,outfalls

def break_cycle(X,outfalls):
    '''
    Break the cyclic paths in the farthest node.

    Parameters
    ----------
    X : Graph
        The drainage network.
    outfalls : list
        The outfall list.

    Returns
    -------
    X : Graph
        The drainage network.

    '''
    sub_set = {out:nx.node_connected_component(X,out) for out in outfalls}
    path_length = nx.multi_source_dijkstra_path_length(X, set(sub_set.keys()), weight = 'length')
    for out,c in sub_set.items():
        subx = nx.subgraph(X,c)        
        while nx.cycle_basis(subx) != []:
            cycles = nx.cycle_basis(subx)
            for cycle in cycles:
                if nx.cycle_basis(nx.subgraph(subx,cycle)) == []:
                    continue
                cycle_path = {cnode: path_length[cnode] for cnode in cycle}
                far_node_name = max(cycle_path, key = cycle_path.get)
                split_edges = [(u, v) for u,v in subx.edges(far_node_name) if u in cycle and v in cycle]
                for spl_edge in split_edges:
                    if set(spl_edge).issubset(set(nx.dijkstra_path(subx,far_node_name,out))) == False:
                        split_edge = spl_edge
                end_name = [split_edge[i] for i in range(2) if split_edge[i] != far_node_name][0]
                # line = X[end_name][far_node_name]['geometry']
                X.remove_edge(end_name,far_node_name)
        for node in c:
            X.nodes[node].update(outdis = path_length[node])
    return X

def export_net(X):
    '''
    Export the GeoDataFrames from the networkx class Graph.

    Parameters
    ----------
    X : Graph
        The drainage network.

    Returns
    -------
    nodes : GeoDataFrame
        Drainage nodes.
    pipes : GeoDataFrame
        Drainage pipes.

    '''
    edges = gpd.GeoDataFrame(nx.to_pandas_edgelist(X))
    nodes = gpd.GeoDataFrame.from_dict(X.nodes,orient='index')
    nodes['name'] = nodes.index
    nodes = nodes.reset_index(drop=True,level=None)
    if 'pos' in nodes.columns:
        nodes = nodes.drop('pos',axis=1)
    # nodes.to_file(node_file)
    # edges.to_file(pipe_file)
    return nodes,edges
    
# def configurate(file):
#     config = {}
#     with open(file,'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.strip('\n').split(',')
#             if len(line)==2:
#                 config[line[0]] = eval(line[-1])
#             elif len(line)>2:
#                 config[line[0]] = [eval(i) for i in line[1:]]
#             else:
#                 print('Configuration %s Incomplete!'%line[0])
#     return config

def configurate(text):
    txt = [eval(te).split('=') for te in text.split(';')]
    config = {k:eval(v) for k,v in txt}
    return config

def round_flow(x):
    return float(piecewise(x,[x<0.17,0.17<=x<0.246,0.246<=x<0.37,x>=0.37],
                           [0.203,math.ceil(100*x)/100,0.37,math.ceil(100*x)/100]))

def get_diameter(q,config):
    flow,diam = config['flow'],config['diam']
    flow = sorted(flow)
    flow_tuple = [(flow[i],flow[i+1]) for i in range(len(flow)-1)]
    condlist = [a <= q < b for a,b in flow_tuple]
    condlist.append(q >= flow[-1])
    diam = sorted(diam)
    return int(piecewise(q,condlist,diam))

def calc_subx(dx,nodestree,config):
 
    edgestree = {k: set(edge for v0 in v for edge in list(dx.in_edges(v0))) 
                 for k,v in sorted(list(nodestree.items())[:-1],reverse=True)}
    
    for u,v in dx.edges():
        dx[u][v].update(velocity=1)
    err = 1
    while err > 0.1:
        for u,v in list(dx.edges()):
            dx[u][v].update(t0 = dx[u][v]['length']/dx[u][v]['velocity']/60)
        for values in edgestree.values():
            for u,v in values:
                if dx.in_degree(u) == 0:
                    dx[u][v].update(t = config['t0'])
                else:
                    dx[u][v].update(t =  max([dx[a][b]['t'] + dx[a][b]['t0'] 
                                                  for a,b in list(dx.in_edges(u))]))
                
                dx[u][v].update(q = config['phi'] * config['A'] *\
                                (1 + config['c'] * math.log10(config['P']))/\
                                    math.pow(dx[u][v]['t']+config['b'],config['n']))
                dx[u][v].update(flow = dx[u][v]['q'] * dx[u][v]['area']/1000)    
                
                '''
                avoid zero flow rate error.
                '''
                if dx[u][v]['flow'] == 0:
                    dx[u][v]['flow'] = 0.01
                # dx[u][v].update(flow = max([dx[a][b]['flow'] 
                #                                for a,b in list(dx.in_edges(u))]\
                #                               +[dx[u][v]['flow']]))
                # dx[u][v].update(flow = round_flow(dx[u][v]['flow']))
                dx[u][v].update(diameter = get_diameter(dx[u][v]['flow'],config))
                dx[u][v].update(err = 
                                    dx[u][v]['velocity'] -\
                                    4*dx[u][v]['flow']/math.pi/math.pow(dx[u][v]['diameter']/1000,2))
                dx[u][v].update(velocity = 
                                    4*dx[u][v]['flow']/math.pi/math.pow(dx[u][v]['diameter']/1000,2))
                dx[u][v].update(I = math.pow(config['n0']*dx[u][v]['velocity'],2)/math.pow(dx[u][v]['diameter']/4000,4/3))                
        err = max(abs(array([dx[u][v]['err'] for u,v in dx.edges()])))
        
    # for values in edgestree.values():
    #     for u,v in values:
    #         dx[u][v].update(q = config['A'] * (1 + config['c'] * math.log10(config['P']))/math.pow(config['t']+config['b'],config['n']))
    #         dx[u][v].update(flow = (config['phi'] * dx[u][v]['q'] * dx[u][v]['area'])/1000)     
    #         dx[u][v].update(flow = round_flow(dx[u][v]['flow']))
    #         dx[u][v].update(diameter = get_diameter(dx[u][v]['flow'],config))
    #         dx[u][v].update(velocity = 
    #                             4*dx[u][v]['flow']/math.pi/math.pow(dx[u][v]['diameter']/1000,2))
    #         dx[u][v].update(I = math.pow(config['n0']*dx[u][v]['velocity'],2)/math.pow(dx[u][v]['diameter']/4000,4/3))       
    return dx

def vert_subx(subDX,nodestree,futu):
    edgestree = {k: set(edge for v0 in v for edge in list(subDX.in_edges(v0))) 
                 for k,v in sorted(list(nodestree.items())[:-1],reverse=True)}
    for values in edgestree.values():
        for u,v in values:
            if subDX.in_degree(u) == 0:
                subDX[u][v].update(us_futu = futu, 
                                   us_depth = subDX[u][v]['us_grdele'] -\
                                       futu - subDX[u][v]['diameter']/1000)                
            else:
                subDX[u][v].update(us_depth = min([subDX[a][b]['ds_depth'] +\
                                                   (subDX[a][b]['diameter'] - subDX[u][v]['diameter'])/1000
                                                   for a,b in list(subDX.in_edges(u))]))
                subDX[u][v].update(us_futu = subDX[u][v]['us_grdele'] -\
                                   subDX[u][v]['us_depth'] - subDX[u][v]['diameter']/1000)
            subDX[u][v].update(ds_depth = subDX[u][v]['us_depth'] -\
                               subDX[u][v]['length'] * subDX[u][v]['I'])
            subDX[u][v].update(ds_futu = subDX[u][v]['ds_grdele'] -\
                               subDX[u][v]['ds_depth'] - subDX[u][v]['diameter']/1000)
            if subDX[u][v]['ds_futu'] < futu: #设置跌水井
                delta = futu - subDX[u][v]['ds_futu']
                subDX[u][v].update(us_futu = subDX[u][v]['us_futu'] + delta,
                                    us_depth = subDX[u][v]['us_depth'] - delta,
                                    ds_depth = subDX[u][v]['ds_depth'] - delta,
                                    ds_futu = futu)
            subDX.nodes[u].update(invelev = subDX[u][v]['us_depth'])
    for out in nodestree[0]:
        subDX.nodes[out].update(invelev = subDX[list(subDX.in_edges(out))[0][0]][out]['ds_depth'])
    # for values in edgestree.values():
    #     for u,v in values:
    #         subDX[u][v].update(outoffset = subDX[u][v]['ds_depth'] - subDX.nodes[v]['invelev'])
    return subDX

def update_ctrlelev(subDX,nodestree,futu,outlevel):
    # edgestree = {k: set(edge for v0 in v for edge in list(subDX.in_edges(v0))) 
    #              for k,v in sorted(list(nodestree.items())[:-1],reverse=True)}    
    for out in nodestree[0]:
        subDX.nodes[out].update(elev = outlevel,
                                invelev = max(subDX.nodes[out]['invelev'],outlevel),
                                ctrlelev = max(subDX.nodes[out]['invelev'],outlevel))
        
        for u,v in subDX.in_edges(out):
            delta = subDX.nodes[v]['invelev']-subDX[u][v]['ds_depth']
            if delta>0:
                subDX[u][v].update(us_depth = subDX.nodes[u]['invelev'] + delta,
                                   ds_depth = subDX[u][v]['ds_depth'] + delta,
                                   ds_futu = 0)
            # subDX[u][v].update(ds_grdele = outlevel)
            # subDX[u][v].update(outoffset = subDX[u][v]['ds_depth'] - subDX.nodes[v]['invelev'])                
        
    for k,nodes in list(nodestree.items())[1:]:
        for node in nodes:
            subDX.nodes[node].update(invelev = min([subDX[a][b]['us_depth']
                                                    for a,b in subDX.out_edges(node)]),
                                     ctrlelev = max([subDX[a][b]['us_depth']+subDX[a][b]['diameter']/1000+futu
                                                     for a,b in subDX.out_edges(node)]))                
            for u,v in subDX.in_edges(node):
                delta = min([subDX[a][b]['us_depth']+subDX[a][b]['diameter']/1000 
                             for a,b in subDX.out_edges(node)]) - \
                    (subDX[u][v]['ds_depth']+subDX[u][v]['diameter']/1000)
                if delta>0:
                    subDX[u][v].update(us_depth = subDX[u][v]['us_depth'] + delta,
                                       ds_depth = subDX[u][v]['ds_depth'] + delta)
        
            for u,v in subDX.in_edges(node):
                subDX.nodes[v].update(invelev = min(subDX[u][v]['ds_depth'],subDX.nodes[v]['invelev']),
                                      ctrlelev = max(subDX[u][v]['ds_depth']+subDX[u][v]['diameter']/1000 + futu,
                                                     subDX.nodes[v]['ctrlelev']))
                # subDX[u][v].update(outoffset = subDX[u][v]['ds_depth'] - subDX.nodes[v]['invelev'])
                subDX[u][v].update(ds_grdele = max(subDX[u][v]['ds_grdele'],subDX.nodes[v]['ctrlelev']))
                subDX[u][v].update(ds_futu = subDX[u][v]['ds_grdele']-\
                                   subDX[u][v]['ds_depth'] - \
                                       subDX[u][v]['diameter']/1000)   
            for u,v in subDX.out_edges(node):
                subDX[u][v].update(us_grdele = max(subDX[u][v]['us_grdele'],subDX.nodes[u]['ctrlelev']))
                subDX[u][v].update(us_futu = subDX[u][v]['us_grdele']- \
                                   subDX[u][v]['us_depth'] - \
                                       subDX[u][v]['diameter']/1000)        
    # for k,edges in  sorted(edgestree.items()):
    #     for u,v in edges:
    #         delta = subDX.nodes[v]['invelev']-subDX[u][v]['ds_depth']
    #         if subDX.nodes[v]['node_type'] == 'node':
    #             delta += (max([subDX[v][no]['diameter'] for no in list(subDX[v])]) - subDX[u][v]['diameter'])/1000     
    #         if delta>0:
    #             # subDX.nodes[u].update(invelev=subDX.nodes[u]['invelev'] + delta)
    #             subDX[u][v].update(us_depth = subDX.nodes[u]['invelev'] + delta,
    #                                ds_depth = subDX[u][v]['ds_depth'] + delta,
    #                                us_futu = max(subDX[u][v]['us_futu'] - delta,futu),
    #                                ds_futu = max(subDX[u][v]['ds_futu']-delta,futu))
                
    #         # subDX.nodes[u].update(ctrlelev = subDX[u][v]['us_depth']+subDX[u][v]['diameter']/1000 + futu)
    #         if subDX.nodes[v]['node_type'] == 'node':
    #             subDX.nodes[v].update(ctrlelev = max(subDX[u][v]['ds_depth']+subDX[u][v]['diameter']/1000 + futu,
    #                                                  subDX.nodes[v]['ctrlelev']))
    #         subDX[u][v].update(us_grdele = max(subDX[u][v]['us_grdele'],subDX.nodes[u]['ctrlelev']),
    #                            ds_grdele = max(subDX[u][v]['ds_grdele'],subDX.nodes[v]['ctrlelev']))
    #         subDX[u][v].update(outoffset = subDX[u][v]['ds_depth'] - subDX.nodes[v]['invelev'])
    return subDX


def DEMGenerate(point_file,field,tif_file=None,pixel=30, alg=None):
    filedir,file = osplit(point_file)
    x,y,z = field
    name = file.split('.')[0]
    if point_file.endswith('.csv') or point_file.endswith('.txt'):
        points = gpd.pd.read_csv(point_file)
        points['geometry'] = points.apply(lambda row:Point(row[x],row[y],row[z]),axis=1)
        points = gpd.GeoDataFrame(points)
        points.to_file(join(filedir,name+'.shp'))
        
    elif point_file.endswith('.shp'):
        points = gpd.read_file(point_file)
        if set(points.has_z) != {True}:
            points['geometry'] = points.apply(lambda row:Point(*row['geometry'].coords[0],row[z]),axis=1)
            points.to_file(join(filedir,name+'_z.shp'))
            name += '_z'
    else:
        raise Exception('Wrong format! Please use csv or shp')
            
    minx,miny,maxx,maxy = points.unary_union.bounds
    lonext = maxx - minx
    latext = maxy - miny    
    nlng,nlat = int(lonext/pixel),int(latext/pixel)            
        
    
    line = 'gdal_grid -l {0} -ot Float32 -of GTiff -outsize {1} {2} '.format(name,nlng,nlat)
    line += '-a invdist:power=2.0:smothing=0.0:max_points=0:min_points=0:nodata=0.0' if alg == None else alg
    line += ' '+join(filedir, name + '.shp') +' '
    line += join(filedir,name+'.tif') if tif_file is None else tif_file
    
    system(line)
    
    return join(filedir,name+'.tif') if tif_file is None else tif_file

    
    
    
def get_elevation(X,demfile):
    ds = gdal.Open(demfile)
    ini_x = ds.GetGeoTransform()[0]
    ini_y = ds.GetGeoTransform()[3]
    dx = ds.GetGeoTransform()[1]
    dy = ds.GetGeoTransform()[5]
    
    im_width = ds.RasterXSize
    im_height = ds.RasterYSize
    im_data = ds.GetRasterBand(1).ReadAsArray(0,0,im_width,im_height)
    
    for node in X.nodes:
        point = X.nodes[node]['geometry'].coords[0]
        numx = int((point[0] - ini_x) // dx)
        numy = int((point[1] - ini_y) // dy)
        X.nodes[node].update(elev = im_data[numy, numx])
    del ds
    return X

def check_field(X):
    nodes = gpd.GeoDataFrame.from_dict(X.nodes,orient='index')
    if 'outfall' or 'outdis' not in nodes.columns:
        outfalls = set(nodes[nodes['node_type'] == 'outfall'].index)
        for c in nx.connected_components(X.to_undirected()):
            outs = list(outfalls.intersection(c))
            subx = nx.subgraph(X.to_undirected(),c)
            length = nx.multi_source_dijkstra_path_length(subx,outs,weight = 'length')
            for node in c:
                X.nodes[node].update(outfall = ''.join(outs),outdis = length[node])
    return X

# todo 确定双排扣的转输面积和流向怎么算
def update_graph(X,config):
    # for node in X.nodes:
    #     X.nodes[node].update(elev = get_elevation(X.nodes[node]['geometry'].coords[0],demfile))
    node_forest = []
    DX = nx.DiGraph()
    for c in nx.connected_components(X):
        subx = nx.subgraph(X,c)
        outs = {no for no in c if subx.nodes[no]['node_type'] == 'outfall'}
        i = 1
        nodestree = {i-1:outs}
        dfsnodes = outs.copy()
        while dfsnodes!=set(subx.nodes):
            i += 1
            nodes = set()
            for out in outs:
                nodes = nodes.union(set(nx.dfs_postorder_nodes(subx,out,depth_limit=i)))
            nodes = nodes - dfsnodes
            nodestree[i-1] = nodes
            dfsnodes = dfsnodes.union(nodes)
        node_forest.append(nodestree)
        
        if len(outs) == 2:
            line = nx.shortest_path(subx,*outs)
            for key in sorted(nodestree,reverse=True):
                for node in nodestree[key]:
                    tr_area = 0
                    for no in list(X[node]):
                        if key == max(nodestree):
                            break
                        elif no in nodestree[key+1] and no not in line:
                            tr_area += X.nodes[no]['lo_area'] + X.nodes[no]['tr_area']
                            DX.add_edge(no,node,area=(X.nodes[no]['lo_area'] + X.nodes[no]['tr_area'])/10000,
                                        us_node = no, ds_node = node, length = X[no][node]['length'],
                                        velocity=1,
                                        us_grdele = X.nodes[no]['elev'],
                                        ds_grdele = X.nodes[node]['elev'],
                                        geometry = LineString([X.nodes[no]['geometry'],X.nodes[node]['geometry']]))                            
                    X.nodes[node].update(tr_area = tr_area) 
                    DX.add_node(node,name = node,
                                elev = X.nodes[node]['elev'], 
                                node_type = X.nodes[node]['node_type'],
                                outfall = X.nodes[node]['outfall'], 
                                outdis = X.nodes[node]['outdis'], 
                                lo_area = X.nodes[node]['lo_area'], 
                                tr_area = tr_area,
                                geometry = X.nodes[node]['geometry'])
                                            
            area_weight = {node:DX.nodes[node]['tr_area']+DX.nodes[node]['lo_area']
                           for node in line}
            inter = [n for n in line if subx.degree(n)>2]
            acc_area = [sum(list(area_weight.values())[:idx+1]) for idx in range(len(area_weight))]
            acc_med_area = ((array(acc_area)-sum(area_weight.values())/2)>0).tolist()
            if True in acc_med_area:
                med_node_ind = acc_med_area.index(True)
            else:
                med_node_ind = len(acc_med_area)//2
            med_node = line[med_node_ind]         
            # if med_node in inter:
                
            errs = []
            for alpha in [0.1*i for i in range(1,10)]:
                area1,area2 = area_weight[med_node]*alpha,area_weight[med_node]*(1-alpha)
                DX.add_edge(med_node,line[med_node_ind-1],area = area1/10000,
                            us_node = med_node, ds_node = line[med_node_ind-1], 
                            length = X[med_node][line[med_node_ind-1]]['length'],
                            velocity=1,
                            us_grdele = X.nodes[med_node]['elev'],
                            ds_grdele = X.nodes[line[med_node_ind-1]]['elev'],
                            geometry = LineString([X.nodes[med_node]['geometry'],
                                                   X.nodes[line[med_node_ind-1]]['geometry']]))
                DX.add_edge(med_node,line[med_node_ind+1],area = area2/10000,
                            us_node = med_node, ds_node = line[med_node_ind+1], 
                            length = X[med_node][line[med_node_ind+1]]['length'],
                            velocity=1,
                            us_grdele = X.nodes[med_node]['elev'],
                            ds_grdele = X.nodes[line[med_node_ind+1]]['elev'],
                            geometry = LineString([X.nodes[med_node]['geometry'],
                                                   X.nodes[line[med_node_ind+1]]['geometry']]))            
                
                path1 = nx.shortest_path(X,med_node,line[0])
                path2 = nx.shortest_path(X,med_node,line[-1])                                                   
                for i,node in enumerate(path1[1:]):        
                    # DX.nodes[node].update(tr_area = DX[path1[i]][node]['area']*1000 + \
                    #                      DX.nodes[node]['tr_area'])
                    if node is not line[0]:
                        DX.add_edge(node,path1[i+2],area = (DX.nodes[node]['lo_area'] +\
                                                   DX.nodes[node]['tr_area'])/10000 + DX[path1[i]][node]['area'],
                                    us_node = node, ds_node = path1[i+2],
                                    length = X[node][path1[i+2]]['length'],
                                    velocity=1,
                                    us_grdele = X.nodes[node]['elev'],
                                    ds_grdele = X.nodes[path1[i+2]]['elev'],
                                    geometry = LineString([X.nodes[node]['geometry'],
                                                           X.nodes[path1[i+2]]['geometry']]))                            
                for i,node in enumerate(path2[1:]):
                    # DX.nodes[node].update(tr_area = DX[path2[i]][node]['area'] + \
                    #                      DX.nodes[node]['tr_area'])         
                    if node is not line[-1]:
                        DX.add_edge(node,path2[i+2],area = (DX.nodes[node]['lo_area'] +\
                                                   DX.nodes[node]['tr_area'])/10000+ DX[path2[i]][node]['area'],
                                    us_node = node, ds_node = path2[i+2],
                                    length = X[node][path2[i+2]]['length'],
                                    velocity=1,
                                    us_grdele = X.nodes[node]['elev'],
                                    ds_grdele = X.nodes[path2[i+2]]['elev'],
                                    geometry = LineString([X.nodes[node]['geometry'],
                                                           X.nodes[path2[i+2]]['geometry']]))                        
                
                subdx = nx.subgraph(DX,c)
                (long_path,short_path) = (path1,path2) if med_node_ind*2>=len(line) else (path2, path1)
                nodestree = {i:[no] for i,no in enumerate(reversed(long_path))}
                for i,no in enumerate(reversed(short_path[1:])):
                    nodestree[i] += [no]
                dfsnodes = set(path1) | set(path2)
                i = max(nodestree)
                nodes = inter
                while dfsnodes!=set(subx.nodes):
                    i += 1
                    nodes = [edge[0] for node in nodes 
                             for edge in set(subdx.in_edges(node)) 
                             if edge[0] not in line]
                    dfsnodes = dfsnodes.union(set(nodes))
                    nodestree[i] = nodes
                subdx = calc_subx(subdx,nodestree,config)
                subdx = vert_subx(subdx,nodestree,config['futu'])
                err = subdx.nodes[nodestree[0][0]]['invelev'] - subdx.nodes[nodestree[0][1]]['invelev']
                errs.append(abs(err))
                
            alpha = errs.index(min(errs))*0.1
            

            area1,area2 = area_weight[med_node]*alpha,area_weight[med_node]*(1-alpha)
            DX.add_edge(med_node,line[med_node_ind-1],area = area1/10000,
                        us_node = med_node, ds_node = line[med_node_ind-1], 
                        length = X[med_node][line[med_node_ind-1]]['length'],
                        velocity=1,
                        us_grdele = X.nodes[med_node]['elev'],
                        ds_grdele = X.nodes[line[med_node_ind-1]]['elev'],
                        geometry = LineString([X.nodes[med_node]['geometry'],
                                               X.nodes[line[med_node_ind-1]]['geometry']]))
            DX.add_edge(med_node,line[med_node_ind+1],area = area2/10000,
                        us_node = med_node, ds_node = line[med_node_ind+1], 
                        length = X[med_node][line[med_node_ind+1]]['length'],
                        velocity=1,
                        us_grdele = X.nodes[med_node]['elev'],
                        ds_grdele = X.nodes[line[med_node_ind+1]]['elev'],
                        geometry = LineString([X.nodes[med_node]['geometry'],
                                               X.nodes[line[med_node_ind+1]]['geometry']]))            
            
            path1 = nx.shortest_path(X,med_node,line[0])
            path2 = nx.shortest_path(X,med_node,line[-1])                                                   
            for i,node in enumerate(path1[1:]):        
                # DX.nodes[node].update(tr_area = DX[path1[i]][node]['area']*1000 + \
                #                      DX.nodes[node]['tr_area'])
                if node is not line[0]:
                    DX.add_edge(node,path1[i+2],area = (DX.nodes[node]['lo_area'] +\
                                               DX.nodes[node]['tr_area'])/10000 + DX[path1[i]][node]['area'],
                                us_node = node, ds_node = path1[i+2],
                                length = X[node][path1[i+2]]['length'],
                                velocity=1,
                                us_grdele = X.nodes[node]['elev'],
                                ds_grdele = X.nodes[path1[i+2]]['elev'],
                                geometry = LineString([X.nodes[node]['geometry'],
                                                       X.nodes[path1[i+2]]['geometry']]))                            
            for i,node in enumerate(path2[1:]):
                # DX.nodes[node].update(tr_area = DX[path2[i]][node]['area'] + \
                #                      DX.nodes[node]['tr_area'])         
                if node is not line[-1]:
                    DX.add_edge(node,path2[i+2],area = (DX.nodes[node]['lo_area'] +\
                                               DX.nodes[node]['tr_area'])/10000+ DX[path2[i]][node]['area'],
                                us_node = node, ds_node = path2[i+2],
                                length = X[node][path2[i+2]]['length'],
                                velocity=1,
                                us_grdele = X.nodes[node]['elev'],
                                ds_grdele = X.nodes[path2[i+2]]['elev'],
                                geometry = LineString([X.nodes[node]['geometry'],
                                                       X.nodes[path2[i+2]]['geometry']]))                        
            
            subdx = nx.subgraph(DX,c)
            (long_path,short_path) = (path1,path2) if med_node_ind*2>=len(line) else (path2, path1)
            nodestree = {i:[no] for i,no in enumerate(reversed(long_path))}
            for i,no in enumerate(reversed(short_path[1:])):
                nodestree[i] += [no]
            dfsnodes = set(path1) | set(path2)
            i = max(nodestree)
            nodes = inter
            while dfsnodes!=set(subx.nodes):
                i += 1
                nodes = [edge[0] for node in nodes 
                         for edge in set(subdx.in_edges(node)) 
                         if edge[0] not in line]
                dfsnodes = dfsnodes.union(set(nodes))
                nodestree[i] = nodes
            subdx = calc_subx(subdx,nodestree,config)
            subdx = vert_subx(subdx,nodestree,config['futu'])            
            subdx = update_ctrlelev(subdx,nodestree,config['futu'],config['outlevel'])
            # a_l = {k:(v*nx.shortest_path_length(subx,k,line[0]),v*nx.shortest_path_length(subx,k,line[-1]))
            #        for k,v in area_weight.items()}
            # a_l_cum = {k:(sum([a_l[n][0] for n in nx.shortest_path(subx,line[0],k)]),
            #               sum([a_l[n][-1] for n in nx.shortest_path(subx,line[-1],k)]))
            #            for k,v in a_l.items()}
            # a_l_cust = {line[0]:[k for k,v in a_l_cum.items() if v[0]<=v[-1]],
            #             line[-1]:[k for k,v in a_l_cum.items() if v[-1]<v[0]]}                
            # a_l_cust = {k:sorted(v,key=lambda a:nx.shortest_path_length(subx,k,weight='length')[a],reverse=True)
            #                      for k,v in a_l_cust.items()}
            # for k,v in a_l_cust.items():
            #     for i,node in enumerate(v[1:]):
            #         X.nodes[node].update(tr_area = X.nodes[v[i]]['lo_area'] + \
            #                                   X.nodes[v[i]]['tr_area'])    
            #     nodetree = {a:b.intersection(v) for a,b in nodestree.items() if b.intersection(v) !=set()}
            #     node_forest[k] = nodetree
        else:
            for key in sorted(nodestree,reverse=True):
                for node in nodestree[key]:
                    tr_area = 0
                    for no in list(X[node]):
                        if key == max(nodestree):
                            break                        
                        elif no in nodestree[key+1]:
                            tr_area += X.nodes[no]['lo_area'] + X.nodes[no]['tr_area']
                            DX.add_edge(no,node,area=(X.nodes[no]['lo_area'] + X.nodes[no]['tr_area'])/10000,
                                        us_node = no, ds_node = node, length = X[no][node]['length'],
                                        velocity=1,
                                        us_grdele = X.nodes[no]['elev'],
                                        ds_grdele = X.nodes[node]['elev'],
                                        geometry = LineString([X.nodes[no]['geometry'],X.nodes[node]['geometry']]))                            
                    X.nodes[node].update(tr_area = tr_area) 
                    DX.add_node(node,name = node,
                                elev = X.nodes[node]['elev'], 
                                node_type = X.nodes[node]['node_type'],
                                outfall = X.nodes[node]['outfall'], 
                                outdis = X.nodes[node]['outdis'], 
                                lo_area = X.nodes[node]['lo_area'], 
                                tr_area = tr_area,
                                region = X.nodes[node]['outfall'].replace('O','SA'), 
                                geometry = X.nodes[node]['geometry'])
            
            subdx = nx.subgraph(DX,c)
            subdx = calc_subx(subdx,nodestree,config)
            subdx = vert_subx(subdx,nodestree,config['futu'])   
            subdx = update_ctrlelev(subdx,nodestree,config['futu'],config['outlevel'])
    for node in DX.nodes:
        if DX.nodes[node]['node_type'] == 'outfall':
            DX.nodes[node].update(outdis = max([DX.nodes[po]['outdis']
                                                for po in nx.node_connected_component(X,node)]))
    for u,v in DX.edges:
        geom = DX[u][v]['geometry']
        geom = LineString([Point(*geom.coords[0],DX[u][v]['us_depth']+DX[u][v]['diameter']/2000),
                           Point(*geom.coords[-1],DX[u][v]['ds_depth']+DX[u][v]['diameter']/2000)])
        DX[u][v].update(geometry=geom)
    return DX

def export_hydraulic_table(DX):
    columns = {'region':'服务系统','source':'起点编号','target':'终点编号',
               'length':'管长L(m)','area':'汇水面积A(ha)',
               't0':'管内雨水流行时间t0(min)','t':'总雨水流行时间t(min)',
               'q':'单位面积径流量qs(L/(s·ha))','flow':'设计流量Q(L/s)',
               'diameter':'管径(mm)','I':'水力坡度S(‰)',
               'velocity':'流速v(m/s)','decent':'水力坡降S·L(m)',
               'us_grdele':'起点地面标高(m)','ds_grdele':'终点地面标高(m)',
               'us_depth':'起点管内底标高(m)','ds_depth':'终点管内底标高(m)',
               'us_futu':'起点埋深(m)','ds_futu':'终点埋深(m)'}
    for u,v in DX.edges:
        if 'source' in DX[u][v]:
            DX[u][v].pop('source')
        if 'target' in DX[u][v]:
            DX[u][v].pop('target')
    edges = nx.to_pandas_edgelist(DX)
    edges['flow'] = edges['flow'] * 1000
    edges['decent'] = edges['length'] * edges['I']
    edges['I'] = edges['I'] * 1000
    
    nodes = gpd.GeoDataFrame.from_dict(DX.nodes,orient='index')
    edges['region'] = edges['source'].apply(lambda so:nodes.loc[so,'region'])
    edges['outdis'] = edges['target'].apply(lambda tar:nodes.loc[tar,'outdis']
                                            if nodes.loc[tar,'node_type'] != 'outfall' else 0)
    
    edges = edges.rename(columns=columns)
    gp = edges.groupby(['服务系统','起点编号','终点编号','outdis']).agg('mean')
    gp = gp.sort_index(level=['服务系统','outdis'],ascending=False)
    gp = gp.droplevel('outdis')
    gp = gp.loc[sorted(gp.index,key=lambda x:min([eval(a) for a in x[0].split('SA')[1:]]))]

    
    table = gp.drop(columns=[col for col in gp.columns
                                if col not in columns.values()])
    table = table[[col for col in columns.values() if col in table.columns]]
    table = table.round({'管长L(m)':1,'汇水面积A(ha)':3,
                         '管内雨水流行时间t0(min)':2,'总雨水流行时间t(min)':2,
                         '单位面积径流量qs(L/(s·ha))':2,'设计流量Q(L/s)':2,
                         '管径(mm)':0,'水力坡度S(‰)':2,'流速v(m/s)':2,'水力坡降S·L(m)':3,
                         '起点地面标高(m)':2,'终点地面标高(m)':2,
                         '起点管内底标高(m)':2,'终点管内底标高(m)':2,
                         '起点埋深(m)':2,'终点埋深(m)':2})
    return table

def merge_outfall_area(X,Polygons):
    '''
    Merge the subcatchments of the same outfall into the service area.

    Parameters
    ----------
    X : Graph
        The drainage network.
    Polygons : GeoDataFrame
        Subcatchments.

    Returns
    -------
    Polygons : GeoDataFrame
        Subcatchments.
    area : GeoDataFrame
        The service region of outfalls (merged subcatchments).

    '''
    # Polygons = gpd.read_file(subcatch)
    Polygons = Polygons.drop(Polygons[Polygons['node'].apply(lambda x:
                                                             x not in X.nodes)].index)
    Polygons['outfall'] = Polygons['node'].apply(lambda x: 
                                                 X.nodes[x]['outfall'] 
                                                 if isinstance(x,str) else '')
    Polygons['region'] = Polygons['outfall'].apply(lambda out:out.replace('O','SA'))
    area = gpd.GeoDataFrame(Polygons.groupby(by = 'outfall').sum())
    area['geometry'] = Polygons.groupby(by = 'outfall')['geometry'].aggregate(lambda x:unary_union(x))
    area['nodes'] = Polygons.groupby(by = 'outfall')['node'].aggregate(lambda x:list(x))
    # area = Polygons.dissolve(by = 'outfall',aggfunc='sum')
    area['max_outdis'] = area['nodes'].apply(lambda nodes: max([X.nodes[node]['outdis'] 
                                                             for node in nodes])) 
    area['outfall'] = area.index
    area = area.reset_index(drop=True,level=None)
    area = area.drop([co for co in area.columns 
                      if co not in ['max_outdis','outfall','area','geometry']],axis=1)

    area['name'] = area['outfall'].apply(lambda out:out.replace('O','SA'))
    return Polygons,area
    
    

    
    
    
    
    
    
    
    
    