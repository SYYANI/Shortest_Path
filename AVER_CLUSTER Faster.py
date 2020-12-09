#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 21:07:49 2020

@author: sunyiyan
"""

# ____________________________________________________________________
# SECTION 1 - IMPORTS

import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import copy

# ____________________________________________________________________
# SECTION 2 - VISUALISATION FUNCTION

def k_distrib(graph, scale='lin', colour='#40a6d1', alpha=.8, expct_lo=1, expct_hi=10, expct_const=1,whic=0):
    plt.close()
    num_nodes = graph.number_of_nodes()
    max_degree = 0
    # Calculate the maximum degree to know the range of x-axis
    for n in graph.nodes():
        if graph.degree(n) > max_degree:
            max_degree = graph.degree(n)
    # X-axis and y-axis values
    x = []
    y_tmp = []
    # loop for all degrees until the maximum to compute the portion of nodes for that degree
    for i in range(max_degree+1):
        x.append(i)
        y_tmp.append(0)
        for n in graph.nodes():
            if graph.degree(n) == i:
                y_tmp[i] += 1
        y = [i/num_nodes for i in y_tmp]
    # Plot the graph
    #plt.figure(figsize=(15, 15))
    deg, = plt.plot(x, y,label='Degree distribution',linewidth=0, marker='o',markersize=8, color=colour, alpha=alpha)
    # Check for the lin / log parameter and set axes scale
    if scale == 'log':
        #plt.figure(figsize=(15, 15))
        plt.xscale('log')
        plt.yscale('log')
        if whic==1:
            plt.title('Scale-Free model Degree distribution (log-log scale)')
        elif whic==2:
            plt.title('Erdos-Renyi model Degree distribution (log-log scale)')
        #plt.title('Degree distribution (log-log scale)')
        # add theoretical distribution line k^-3
        w = [a for a in range(expct_lo,expct_hi)]
        z = []
        for i in w:
            x = (i**-3) * expct_const # set line's length and fit intercept
            z.append(x)

        plt.plot(w,z, 'k-', color='#7f7f7f')
    else:
        if whic==1:
            plt.title('Scale-Free model Degree distribution (linear scale)')
        elif whic==2:
            plt.title('Erdos-Renyi model Degree distribution (linear scale)')
    #plt.figure(figsize=(15, 15))
    plt.ylabel('P(k)')
    plt.xlabel('k')
    plt.show()
    
def find_shortest_path(Gr,PN1=None,PN2=None):
    node_con=[]
    for i in range(10000):
        node_con.append([])
    temp=[]
    for i in range(10000):
        temp.append([])
    temp1=[]    
    for i in range(10000):
        temp1.append(temp)
    temp2=[]
    for i in range(10000):
        temp2.append(temp1)
    Connet_Nodes=[]
    for i in range(10000):
        Connet_Nodes.append(temp2)
     
    for jsd, nbrs in Gr.adjacency():
        node_con[jsd]=list(nbrs.keys()) 
        
    node_conn=copy.deepcopy(node_con) 
    node_connn=copy.deepcopy(node_con)
    node_connnn=copy.deepcopy(node_con)
    
    for j1 in range(len(Gr.nodes())):   
        Connet_Nodes[j1]=node_con[j1]
    
    for jk in range(len(Gr.nodes())):
        j3=0
        for j2 in Connet_Nodes[jk]:
            Connet_Nodes[jk][j3]=node_conn[j2]
            j3+=1
        
    Connet_Nodes_Copy=copy.deepcopy(Connet_Nodes)
    
    for coun1 in range(len(Gr.nodes())):
        coun2=0
        for elem1 in Connet_Nodes[coun1]:
            coun3=0
            for elem2 in Connet_Nodes[coun1][coun2]:
                Connet_Nodes_Copy[coun1][coun2][coun3]=node_connn[elem2]
                coun3+=1
            coun2+=1
    
    path=[]
    
    flag=6
    if PN1==PN2:
        flag=0
        path.append([PN1,PN2])
    else:
        if flag>0:
            for node_depth_1 in node_connnn[PN1]:
                if node_depth_1 == PN2:
                    flag=1
                    path.append([(PN1,node_depth_1)])
                else:
                    if flag>1:
                        count_1=0
                        for node_depth_2 in Connet_Nodes[PN1][count_1]:
                            if node_depth_2 == PN2:
                                flag=2
                                path.append([(PN1,node_depth_1),(node_depth_1,node_depth_2)])
                            else:
                                if flag >2:
                                    count_2=0
                                    for node_depth_3 in Connet_Nodes_Copy[PN1][count_1][count_2]:
                                        if node_depth_3 == PN2:
                                            flag =3
                                            path.append([(PN1,node_depth_1),(node_depth_1,node_depth_2),(node_depth_2,node_depth_3)])
                                        else:
                                            if flag>3:
                                                for node2_depth_1 in node_connnn[PN2]:
                                                    if node2_depth_1 == node_depth_3:
                                                        flag=4
                                                        path.append([(PN1,node_depth_1),(node_depth_1,node_depth_2),(node_depth_2,node_depth_3),(node2_depth_1,PN2)])
                                                    else:
                                                        if flag>4:
                                                            count2_1=0
                                                            for node2_depth_2 in Connet_Nodes[PN2][count2_1]:
                                                                if node2_depth_2 == node_depth_3:
                                                                    flag=5
                                                                    path.append([(PN1,node_depth_1),(node_depth_1,node_depth_2),(node_depth_2,node_depth_3),(node2_depth_2,node2_depth_1),(node2_depth_1,PN2)])
                                                                else:
                                                                    if flag>5:
                                                                        count2_2=0
                                                                        for node2_depth_3 in Connet_Nodes_Copy[PN2][count2_1][count2_2]:
                                                                            if node2_depth_3 == node_depth_3:
                                                                                path.append([(PN1,node_depth_1),(node_depth_1,node_depth_2),(node_depth_2,node_depth_3),(node2_depth_3,node2_depth_2),(node2_depth_2,node2_depth_1),(node2_depth_1,PN2)]) 
                                                                            if count2_2 < len(Connet_Nodes_Copy[PN2][count2_1])-1:
                                                                                count2_2+=1
                                                                if count2_1 < len(Connet_Nodes[PN2])-1:
                                                                    count2_1+=1
                                        if count_2 < len(Connet_Nodes_Copy[PN1][count_1])-1:
                                            count_2+=1
                            if count_1 < len(Connet_Nodes[PN1])-1:
                                count_1+=1
    edge_lists=[]
    if path:
        edge_lists=min(path, key=len)  
        return len(edge_lists)   
 
def getPI(start,Gr):
    node_con=[]
    for i in range(10000):
        node_con.append([])
    for jsd, nbrs in Gr.adjacency():
        node_con[jsd]=list(nbrs.keys()) 
    visiting = list()
    visited = set()
    steps = 0
    num = 0
    visiting.append(start)
    visited.add(start)
    while visiting:
        next_node_list = list()
        num += 1
        for i in range(len(visiting)):
            visiting_node = visiting.pop()
            for next_node in list(node_con[visiting_node]):
                if next_node not in visited:
                    next_node_list.append(next_node)
                    visited.add(next_node)
                    steps += num
        visiting = next_node_list
    return steps
# ____________________________________________________________________
# SECTION 3 - ALGORITHM

#Choose Model
chooseNumber =int(input("\nchoose Scale-Free model by 1 or Erdos-Renyi model by 2: "))

if chooseNumber==1:
    # Get parameters
    
    init_nodes =5 #int(input("Please type in the initial number of nodes (m_0): "))
    final_nodes =50 #int(input("\nPlease type in the final number of nodes: "))
    m_parameter =4 #int(input("\nPlease type in the least number of each nood connected: "))
    while m_parameter>init_nodes:
        m_parameter = int(input("\nWring Input!Please type in the least number of each nood connected again: "))
    #确保(m)<(m0)
    print("\n")
    print("Creating initial graph...")
    
    G = nx.complete_graph(init_nodes)
    
    print("Graph created. Number of nodes: {}".format(len(G.nodes())))
    print("Adding nodes...")
    
    def rand_prob_node():
        nodes_probs = []
        for node in G.nodes():
        #对网络中的节点进行遍历
            node_degr = G.degree(node)
            #获取节点的度
            node_proba = node_degr / (2 * len(G.edges()))
            #每一条边连接两个结点 根据公式计算出网络中每一个节点被连接的概率
            nodes_probs.append(node_proba)
            #将节点被连接的概率记录到列表中
        #从网络中的节点列表(G.nodes())中中随机抽取数字
        #数组p:与传入的第一个参数相对应，表示取(G.nodes())中每个元素的概率
        random_proba_node = np.random.choice(G.nodes(),p=nodes_probs)
        #完成具有优先性的随机节点抽取 并将其作为函数的返回值
        return random_proba_node
    
    def add_edge():
            if len(G.edges()) == 0:
            #判断是否是第一次在节点中建立连接
            #若是初次连接则将目标节点连接到序号为0的节点
                random_proba_node = 0
            else:
            #调用节点获取函数
                random_proba_node = rand_prob_node()
            #在目标节点和随机节点间建立临时连接
            new_edge = (random_proba_node, new_node)
            #判断临时连接是否已经存在于网络中
            if new_edge in G.edges():
                #如果已经存在则打印提示“节点间已存在连接”
                #print("\tThe edge connecting node {} and node {} already exists！".format(new_node + 1, random_proba_node))
                #递归调用节点连接函数
                add_edge()
            else:
                #如果（临时连接变量）是一条新连接则执行节点连接操作
                G.add_edge(new_node, random_proba_node)  
    count = 0
    new_node = init_nodes
    for f in range(final_nodes - init_nodes):
    #网络已经完成了对最初(m)节点的初始化，故经概率判断加入网络的节点为总节点数减去初始化节点数
        G.add_node(init_nodes + count)
        #用初始化节点数加上设置的计数参数作为新节点的序号
        count += 1
        for e in range(0, m_parameter):
        #对每个新加入的节点进行(m0)次与现有节点的连接操作
            add_edge()
        new_node += 1
        #为下一个节点的加入做准备
# ____________________________________________________________________
# SECTION 4 - AVERAGE PATH LENGTH
    link_num = 0
    link_sum = 0
    for PN1k in range(len(G.nodes())):
        link_num=0
        for PN2k in range(len(G.nodes())):
            link_num+=find_shortest_path(G,PN1k,PN2k)
        link_sum += link_num
    
    link_sum -= 2 * len(G.nodes())
    average_path_length=link_sum / ((len(G.nodes))*(len(G.nodes)-1))
    print("\nFinal number of nodes ({}) reached".format(len(G.nodes())))
    print("\nAverage Path Length is: {}".format(average_path_length))
    #print(nx.average_shortest_path_length(G))
    #link_num=nx.shortest_path_length()
    
    '''
    print("\nFinal number of nodes ({}) reached".format(len(G.nodes())))
    PhotoNumber = int(input("\nchoose Network Diagram by 1 or Scatter Diagram by 2 or Double logarithm processed scatter plot by 3 : "))
    if PhotoNumber==1:
        plt.figure(figsize=(15, 15))
        nx.draw(G, node_size=50, with_labels=0, alpha=0.6, node_color="#40a6d1", edge_color="#52bced")
        plt.title("Visulation Of The Scale Free Network(Number: {})".format(len(G.nodes())))
    elif PhotoNumber==2:
        k_distrib(graph=G,colour='#40a6d1',alpha=.8,whic=1)
    elif PhotoNumber==3:
        k_distrib(graph=G,colour='#40a6d1', scale='log',alpha=.8, expct_lo=3, expct_hi=14, expct_const=8,whic=1)
    else:
        print("Wrong Input!")
    '''  

elif chooseNumber==2:
    final_nodes2 = int(input("Please type in the final number of nodes: "))
    m_parameter2 = int(input("\nPlease type in the least number of each nood connected: "))
    
    link_possible = float(input("\nPlease type in the probability that the two nodes can be connected(0～1): "))
    #节点连接的可能性阈值 即每对节点连接上的可能性均为用户所指定的值
    while (link_possible<=0 or link_possible>=1):
        link_possible = float(input("\nWring Input!Please type in the probability again: "))
    #确保概率为0～1范围内的值
   
    G2 = nx.Graph()
    
    print("Graph created. Number of nodes: {}".format(len(G2.nodes())))
    print("Adding nodes...")
    
    def rand_prob_node2():
        random_proba_node2 = np.random.choice(G2.nodes())
        #从网络中的节点列表(G.nodes())中中随机抽取节点 并作为函数的返回值
        return random_proba_node2
    
    def add_edge2():
            if len(G2.edges()) == 0:
                random_proba_node2 = 0
            else:
                random_proba_node2 = rand_prob_node2()
            new_edge2 = (random_proba_node2, new_node2)
            if new_edge2 in G2.edges():
                #print("\tThe edge connecting node {} and node {} already exists！".format(new_node2 + 1, random_proba_node2))
                add_edge2()
            else:
                if (random.random()<=link_possible):
                    G2.add_edge(new_node2, random_proba_node2)
    
    count = 0
    new_node2 = 1
    
    for h in range(final_nodes2):
    #进行指定次数的循环来创造用户指定数目的节点的建立
        G2.add_node(h)
        #用进行的循环次数作为新节点的序号
        count += 1
        
    print("Connect nodes...")
    for k in range(final_nodes2-1):
        for e in range(0, m_parameter2):
            add_edge2()
        new_node2 += 1
# ____________________________________________________________________
# SECTION 4 - AVERAGE PATH LENGTH
    #print(find_shortest_path(G2,4,6))    
    #print(find_shortest_path(G2,4,7))  
    #print(find_shortest_path(G2,4,9)) 
    #print(len(G2.nodes()))
    
    '''
    node_pair=[]
    big=0
    small=0
    for x_axis in range(len(G2.nodes())):
        for y_axis in range(len(G2.nodes())):
            if x_axis > y_axis:
                big = x_axis
                small = y_axis
                if [big,small] not in node_pair:
                     node_pair.append([big,small])
            elif x_axis < y_axis:
                big = y_axis
                small=x_axis
                if [big,small] not in node_pair:
                    node_pair.append([big,small])
    '''
                
    #link_num = 0
    link_sum = 0
    #find_num=0
    #steps=getPI(1,G2)
    #print("steps: {}".format(steps))
    for elem in range(len(G2.nodes())):
        link_sum+=getPI(elem,G2)
    '''
    for elems in node_pair:
        PN1k =elems[0]
        PN2k =elems[1]
        link_sum+=(nx.shortest_path_length(G2,PN1k,PN2k))
        #link_sum+=find_shortest_path(G2,PN1k,PN2k)
        find_num+=1
    '''
    '''    
    for PN1k in range(len(G2.nodes())):
        link_num=0
        for PN2k in range(len(G2.nodes())):
            link_num+=find_shortest_path(G2,PN1k,PN2k)
        link_sum += link_num
    '''
    #print("{} done".format(find_num))
    #print(nx.average_shortest_path_length(G2))
    #print(len(G2.nodes()))
    
    average_path_length= (link_sum) / (len(G2.nodes)*(len(G2.nodes)-1))
    print("\nFinal number of nodes ({}) reached".format(len(G2.nodes())))
    print("\nAverage Path Length is: {}".format(average_path_length))
    #print("\nFinal number of nodes ({}) reached".format(len(G2.nodes())))
    '''
    PhotoNumber2 = int(input("\nchoose Network Diagram by 1 or Scatter Diagram by 2: "))
    if PhotoNumber2==1:
        if len(G2.nodes())< 60:
            nx.draw(G2, node_size=40,width=2,with_labels=0, alpha=0.6, node_color="#40a6d1", edge_color="#52bced",whic=2)
            plt.title("Visulation Of Erdos-Renyi Network(Number: {})".format(len(G2.nodes())))
        else:
            plt.figure(figsize=(15, 15))
            nx.draw(G2, node_size=40, with_labels=0, alpha=0.6, node_color="#40a6d1", edge_color="#52bced",whic=2)
            plt.title("Visulation Of Erdos-Renyi Network(Number: {})".format(len(G2.nodes())))
    elif PhotoNumber2==2:
        k_distrib(graph=G2,colour='#40a6d1',alpha=.8,whic=2)
    else:
        print("Wrong Input!")
    '''  
else:
    print("Wrong Input!")
     
