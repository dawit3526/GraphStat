# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 13:58:34 2017

@author: ailab
"""


import os 
import pandas as pd 
import matplotlib.pyplot as plt 
import time 
import sys 
import networkx as nx 
import datetime 
start_time = time.time() 
print("Time Strarted :",start_time)
def DFS(G,v,seen=None,path=None): 
    if seen is None: seen = [] 
    if path is None: path = [v] 

    seen.append(v) 

    paths = [] 
    try: 
        for t in G[v]: 
            if t not in seen: 
                t_path = path + [t] 
                paths.append(tuple(t_path)) 
                paths.extend(DFS(G, t, seen[:], t_path)) 
    except KeyError: 
        return paths 
    return paths 

def dfs_iterative(graph, start,seen=None,): 
    stack, path = [start], [] 
    if seen is None: seen = [] 
    paths = [] 
    seen.append(start) 
    try: 
        while stack: 
            vertex = stack.pop() 
            if vertex in path: 
                continue 
            path.append(vertex) 
            for neighbor in graph[vertex]: 
                if neighbor not in seen : 
                   t_path = path + [neighbor] 
                   paths.append(tuple(t_path)) 
                stack.append(neighbor) 
    except KeyError: 
        return paths 
    return paths 

base_csv = '/media/dawit/Transcend/GraphStata/Allchains.csv'

df = pd.read_csv(base_csv)
df.drop(df.columns[[0,1,2,4]],axis=1,inplace=True) 
df = df.drop_duplicates(subset=['Source','Destination'],keep=False) 
df= df.reset_index() 
df =df.drop('index',1) 
df['Date'] = df['Date'].str.strip('[]')
df['Date'] = df['Date'].str.strip("''")
df['Date'] = df['Date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d')) 
df['Month'] = pd.DatetimeIndex(df['Date']).month
mask1 = (df['Month'] == 3) 
df1 = df.loc[mask1] 

mask2 = (df['Month'] == 4) 
df2 = df.loc[mask2] 

mask3 = (df['Month'] == 5)
df3 = df.loc[mask3] 

mask4 = (df['Month'] == 6)  
df4 = df.loc[mask4] 

mask5 = (df['Month'] == 7) 
df5 = df.loc[mask5] 

mask6 = (df['Month'] == 8) 
df6 = df.loc[mask6] 

df1.drop(df.columns[[3]],axis=1,inplace=True)
df1= df1.reset_index() 
df1 =df1.drop('index',1) 
df2.drop(df.columns[[3]],axis=1,inplace=True)
df2= df2.reset_index() 
df2=df2.drop('index',1)
df3.drop(df.columns[[3]],axis=1,inplace=True)
df3= df3.reset_index() 
df3 =df3.drop('index',1) 
df4.drop(df.columns[[3]],axis=1,inplace=True)
df4= df4.reset_index() 
df4 =df4.drop('index',1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
df5.drop(df.columns[[3]],axis=1,inplace=True)
df5= df5.reset_index() 
df5 =df5.drop('index',1) 
df6.drop(df.columns[[3]],axis=1,inplace=True)  
df6= df6.reset_index() 
df6 =df6.drop('index',1)
paths = [] 
longes_paths_everyDay = [] 
t_number_of_nodes_everyday = [] 
number_of_nodes_everyday = [] 
d_in_everyDay = [] 
d_out_everyday = [] 
DSUM = 0 
RMHallPaths=[] 
RMHLongestPaths=[] 
sizes = [] 
nodes_numbers = [] 
paths = [] 
xlabels = [] 
d_out = [] 
d_in = [] 
Dates = [] 
Dates_of_allpaths=[] 
Seen_Src=[]
total_paths = [] 

def run_me(dff):
    for j in range(len(dff)):
       
        date = datetime.datetime(2014,02,28) 
        paths = [] 
        d_out = [] 
        d_in = [] 
        #Dates=[] 
        t_number_of_nodes_everyday = [] 
        nodes_numbers = []
        if (dff.Source.ix[j] not in Seen_Src):
            for i in range(151): 
                
                date += datetime.timedelta(days=1) 
                #print(date)
                mask = (dff['Date'] ==date) 
                #print(date) 
                dfg = dff.loc[mask] 
                if len(dfg)>0: 
                    #print("I am here")
                    dfg.drop(dfg.columns[[2]],axis=1,inplace=True) 
                    G=nx.DiGraph() 
                    dfg = dfg.drop_duplicates(keep=False) 
                    dfg= dfg.reset_index() 
                    dfg =dfg.drop('index',1) 
                    dfgg = dfg.values.tolist() 
                    G.add_edges_from(dfgg) 
                     
                    if dff.Source.ix[j] in G: 
                        
                        #print(nx.info(G)) 
                        #    nx.draw(G) 
                        #    plt.savefig("path_graph1.png") 
                        #    plt.show() 
                             
                            #path = createPath(df2,df.Source.ix[0],1) 
                                #for i in range(len(df3)): 
                             
                        #for j in range(len(df)):
                      
                        total_number_of_nodes = len(G) 
                        t_number_of_nodes_everyday.append(total_number_of_nodes) 
                        len_path =0
                        all_paths = dfs_iterative(G,dff.Source.ix[j]) 
                        if(len(all_paths)>=1): 
                           len_path = (max(len(p) for p in all_paths)-1) 
                        elif(len(all_paths)==0): 
                           len_path = 0 
                           degrees_out = 0 
                           degrees_in = 0 
                        #print (path) 
                            # print(path) 
                            # print("--- %s seconds ---" % (time.time() - start_time)) 
                            # print() 
                        #RMHallPaths.append(all_paths) 
                        #RMHLongestPaths.append(len_path) 
                         
                        paths.append(len_path) 
                        degrees_out = G.out_degree(dff.Source.ix[j]) 
                        degrees_in = G.in_degree(dff.Source.ix[j]) 
                         
                        if degrees_out =={} and degrees_in =={}: 
                            degrees_out =0 
                            degrees_in = 0 
                        #D =str(date.month) +'/'+ str(date.day) 
                        #Dates.append(D) 
                        d_out.append(G.number_of_edges()) 
                        d_in.append(G.number_of_edges()) 
                        size_of_graph =  G.size() 
                        sizes.append(size_of_graph) 
                        number_of_nodes = len(G.neighbors(dff.Source.ix[j])) 
                        nodes_numbers.append(G.number_of_nodes()) 
                        Seen_Src.append(dff.Source.ix[j])
            
            longes_paths_everyDay.append(paths) 
            number_of_nodes_everyday.append(nodes_numbers) 
            d_in_everyDay.append(d_in) 
            d_out_everyday.append(d_out)
            
            
            
from multiprocessing import Process
import time

Data = [df1,df2,df3,df4,df5,df6]
print("Thread started")
if __name__ == '__main__':
    threads = [Process(target = run_me, args= (d,)) for d in Data]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
        #print ( t, time.ctime(time.time()) )
res = pd.DataFrame(longes_paths_everyDay)
res.to_csv('longest_path_everyDay.csv')
for path in longes_paths_everyDay:     
    s_paths = pd.Series(path) 
    s_paths = s_paths.value_counts() 
    s_paths.sort_index(inplace=True) 
    #s_paths_values_norm = (s_paths.values - s_paths.values.min()) / (s_paths.values.max() - s_paths.values.min()) 
    total_paths.append(s_paths) 
plt.figure()
for path in total_paths:
    plt.plot(list(path.index),list(path)) 

#plt.xlim([0,len(paths)]) 
#plt.ylim([0,3]) 
#ax =plt.gca() 
#ax.get_xaxis().get_major_formatter().set_useOffset(False) 
#plt.ticklabel_format(useOffset=False) 
plt.xlabel("Longest Path Length") 
plt.ylabel("Number of Paths")
plt.xlim(xmin=1)
#plt.xticks(range(len(paths)), xlabels, rotation='vertical') 
plt.title("Longest paths for all chains") 
plt.legend(loc='upper right', shadow=True) 


plt.savefig('res.png')
End_time = time.time() 
print("Time Finished :",End_time)
