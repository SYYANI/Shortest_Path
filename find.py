#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:19:12 2020

@author: sunyiyan
"""
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import copy

final_nodes2 = int(input("Please type in the final number of nodes: "))
m_parameter2 = 7#int(input("\nPlease type in the least number of each nood connected: "))

link_possible = 0.7#float(input("\nPlease type in the probability that the two nodes can be connected(0～1): "))
    
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

node_lists_community1 = []
node_lists_community2 = []
for k in range(2):
    node_lists_community1.append(k)
for k in range(final_nodes2-2):
    l=k+2
    node_lists_community2.append(l)
all_nodes = node_lists_community1+ node_lists_community2

for h in all_nodes:
#进行指定次数的循环来创造用户指定数目的节点的建立
    G2.add_node(h)
    #用进行的循环次数作为新节点的序号
    count += 1
print("Connect nodes...")
for k in range(final_nodes2-1+5):
    for e in range(0, m_parameter2):
        add_edge2()
    new_node2 += 1
    
print("\nFinal number of nodes ({}) reached".format(len(G2.nodes())))

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
 
for jsd, nbrs in G2.adjacency():
    node_con[jsd]=list(nbrs.keys()) 
    
node_conn=copy.deepcopy(node_con) 
node_connn=copy.deepcopy(node_con)
node_connnn=copy.deepcopy(node_con)

def text_save(content,filename,mode='a'):
    # Try to save a list variable in txt file.
    file = open(filename,mode)
    for i in range(len(content)):
        file.write(str(content[i])+'\n')
    file.close()
#text_save(node_conn,'1.txt')

for j1 in range(len(G2.nodes())):   
    Connet_Nodes[j1]=node_con[j1]

for jk in range(len(G2.nodes())):
    j3=0
    for j2 in Connet_Nodes[jk]:
        Connet_Nodes[jk][j3]=node_conn[j2]
        j3+=1
    
Connet_Nodes_Copy=copy.deepcopy(Connet_Nodes)

for coun1 in range(len(G2.nodes())):
    coun2=0
    for elem1 in Connet_Nodes[coun1]:
        coun3=0
        for elem2 in Connet_Nodes[coun1][coun2]:
            Connet_Nodes_Copy[coun1][coun2][coun3]=node_connn[elem2]
            coun3+=1
        coun2+=1

path=[]

PN1 =random.randint(0,final_nodes2)#int(input("\nThe first person's number(small than {}): ".format(final_nodes2)))
PN2 =random.randint(0,final_nodes2)#int(input("\nThe second person's number(small than {}): ".format(final_nodes2)))   
'''
if PN1==PN2:
    path.append([(PN1,PN2)])
else:
    js=0
    for bl in Connet_Nodes[PN1]:
        js+=1
        for blk in bl:
            if blk==PN2:
                path.append([(PN1,node_conn[PN1][js-1]),(node_conn[PN1][js-1],blk)])
            else:
                if blk in node_conn[PN2]:
                    path.append([(PN1,node_conn[PN1][js-1]),(node_conn[PN1][js-1],blk),(PN2,blk)])
                else:
                    js2=0
                    for bl2 in Connet_Nodes[PN2]:
                        js2+=1
                        for bl2k in bl2:
                            if blk==bl2k:
                                path.append([(PN1,node_conn[PN1][js-1]),(node_conn[PN1][js-1],blk),(node_conn[PN2][js2-1],bl2k),(node_conn[PN2][js2-1],PN2)])
'''
if PN1==PN2:
    path.append([PN1,PN2])
else:
    for node_depth_1 in node_connnn[PN1]:
        if node_depth_1 == PN2:
            path.append([(PN1,node_depth_1),(node_depth_1,PN2)])
        else:
            count_1=0
            for node_depth_2 in Connet_Nodes[PN1][count_1]:
                if node_depth_2 == PN2:
                    path.append([(PN1,node_depth_1),(node_depth_1,node_depth_2),(node_depth_2,PN2)])
                else:
                    count_2=0
                    for node_depth_3 in Connet_Nodes_Copy[PN1][count_1][count_2]:
                        if node_depth_3 == PN2:
                            path.append([(PN1,node_depth_1),(node_depth_1,node_depth_2),(node_depth_2,node_depth_3),(node_depth_3,PN2)])
                        else:
                            for node2_depth_1 in node_connnn[PN2]:
                                if node2_depth_1 == node_depth_3:
                                    path.append([(PN1,node_depth_1),(node_depth_1,node_depth_2),(node_depth_2,node_depth_3),(node_depth_3,node2_depth_1),(node2_depth_1,PN2)])
                                else:
                                    count2_1=0
                                    for node2_depth_2 in Connet_Nodes[PN2][count2_1]:
                                        if node2_depth_2 == node_depth_3:
                                            path.append([(PN1,node_depth_1),(node_depth_1,node_depth_2),(node_depth_2,node_depth_3),(node_depth_3,node2_depth_2),(node2_depth_2,node2_depth_1),(node2_depth_1,PN2)])
                                        else:
                                            count2_2=0
                                            for node2_depth_3 in Connet_Nodes_Copy[PN2][count2_1][count2_2]:
                                                if node2_depth_3 == node_depth_3:
                                                    path.append([(PN1,node_depth_1),(node_depth_1,node_depth_2),(node_depth_2,node_depth_3),(node_depth_3,node2_depth_3),(node2_depth_3,node2_depth_2),(node2_depth_2,node2_depth_1),(node2_depth_1,PN2)])    
                                        count2_1+=1
                        count_2+=1
                count_1+=1



node_lists_community1 = [PN1,PN2]
if len(path)==0:
    print ("Cant Find")
else:
    edge_lists=[]
    edge_lists=min(path, key=len)
    
    print("\n")
    print(min(path, key=len))
    print("You can get to know him only by knowing {} people".format(len(edge_lists)-1))
    pos = nx.spring_layout(G2)
        
    
    plt.figure(figsize=(25, 25))
    nx.draw(G2,pos,node_size=40, with_labels=0, alpha=0.6, node_color="#40a6d1", edge_color="#52bced")
    plt.title("Visulation Of Erdos-Renyi Network(Number: {})".format(len(G2.nodes())))
    nx.draw_networkx_nodes(G2,pos,nodelist=node_lists_community1, node_color='r',node_size=40)
    nx.draw_networkx_edges(G2, pos,edgelist=edge_lists,edge_color='r',width=1)

 

