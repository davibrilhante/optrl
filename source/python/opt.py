#! /usr/bin/env python3
# -*- coding : utf8 -*-

import numpy as np
from json import load
import gurobipy as grb
from gurobipy import GRB

global mvars
mvars = {}

def load_instance(input_file):
    nodes = {}
    aps= {}
    scenario = {}
    with open(input_file) as instance:
        data = load(instance)

        nodes = data['nodes']
        ap = data['aps']
        scenario = data['scenario']

    return nodes, ap, scenario

def channel_label(nodes, ap):
    channels = {}

    for n,user in enumerate(nodes.values()):
        channels[n] = []
        for pos in user['position']:
            dist = np.linalg.norm(np.subtract(ap['position'],pos))
            if dist < 2:
                channels[n].append(3)
            elif dist>= 2 and dist < 3:
                channels[n].append(2)
            elif dist >= 3:
                channels[n].append(1)

    return channels


if __name__ == "__main__":
    optEnv = grb.Env('myEnv.log')                                                
    #optEnv.setParam('OutputFlag', 0)                                           
    model = grb.Model('newModel', optEnv)                                        

    nodes, ap, scenario = load_instance('./test.json')
    channels = channel_label(nodes, ap)
    episode_duration = scenario['duration']
    n_nodes = len(nodes)

    '''
    ADDING MODEL VARIABLES
    '''

    ### Quadratic constraints control                                           
    model.presolve().setParam(GRB.Param.PreQLinearize,1)                        

    # User Scheduling decision variable
    s = {}                                                                      

    # User Packet Loss decision variable
    l = {}                                                                      


    # User Buffer Occupation decision variable
    q = {}                                                                      

    
    # Auxiliar Constraint Indicator decision variable
    ind = {}                                                                      

    # Auxiliar variable decision variable
    aux1 = {}                                                                      
    aux2 = {}                                                                      

    for n, user in enumerate(nodes.values()):                                          
        for t in range(episode_duration):                                
            s[n,t] = model.addVar(vtype=GRB.BINARY,                       
                name='s[{user},{timestep}]'.format(user=n,timestep=t))
            
            l[n,t] = model.addVar(vtype=GRB.INTEGER, lb=0.0, ub=user['demand']*episode_duration,
                name='l[{user},{timestep}]'.format(user=n,timestep=t))

            q[n,t] = model.addVar(vtype=GRB.INTEGER, lb=0.0, ub=user['demand']*episode_duration,
                name='q[{user},{timestep}]'.format(user=n,timestep=t))

            ind[n,t] = model.addVar(vtype=GRB.BINARY, 
                name='ind[{user},{timestep}]'.format(user=n,timestep=t))

            aux1[n,t] = model.addVar(vtype=GRB.INTEGER,
                lb=-user['demand']*episode_duration, ub=user['demand']*episode_duration,
                name='aux[{user},{timestep}]'.format(user=n,timestep=t))

            aux2[n,t] = model.addVar(vtype=GRB.INTEGER,lb=0.0, ub=user['demand']*episode_duration,
                name='aux[{user},{timestep}]'.format(user=n,timestep=t))

    '''
    ADDING MODEL CONSTRAINTS

    '''

    for n, user in enumerate(nodes.values()): 
        for t in range(episode_duration):                                
            model.addConstr(grb.quicksum(s[u,t] for u in range(n_nodes)),GRB.EQUAL,1)

            model.addConstr(aux1[n,t],GRB.EQUAL, q[n,t] - user['buffer'])

            model.addGenConstrMax(ind[n,t],[0, aux1[n,t]])
            
            if t == episode_duration-1:
                break

            model.addGenConstrIndicator(ind[n,t],True,l[n,t+1]-l[n,t],GRB.EQUAL,1)
            
            model.addGenConstrIndicator(s[n,t],True,q[n,t+1]-q[n,t],GRB.EQUAL,user['demand']-channels[n][t])
            
            model.addGenConstrIndicator(s[n,t],False,q[n,t+1]-q[n,t],GRB.EQUAL,1)


    '''
    SETTING MODEL OBJECTIVE FUNCTION
    model.setObjective(grb.quicksum(
        grb.quicksum(user['demand']-l[n,t] for n, user in enumerate(nodes.values()))**2/(
            n_nodes*grb.quicksum((user['demand']-l[n,t])**2 for n, user in enumerate(nodes.values())))
        for t in range(episode_duration)), GRB.MAXIMIZE)
    '''
    model.setObjective(grb.quicksum(grb.quicksum(l[n,t] for t in range(episode_duration)) for n in range(n_nodes)), 
                        GRB.MINIMIZE)

    model.write('myModel.lp')

    '''
    START OPTIMIZATION
    '''
    try:
        model.optimize()
        mvars.update({                                                          
            'nvars': model.getAttr('numVars'),                                  
            'nconst': model.getAttr('numConstrs'),                              
            'nqconst': model.getAttr('numQConstrs'),                            
            'ngconst': model.getAttr('numGenConstrs'),                          
            'status': model.Status == GRB.OPTIMAL,                              
            'runtime': model.getAttr('Runtime'),                                
            'node': model.getAttr('nodecount'),                                 
            'obj': model.objVal,                                                
        })

    except grb.GurobiError as error:
        print(error)
        exit()

    print(mvars)
    print([(i.varName,i.x) for i in s.values()])
    print(channels)
