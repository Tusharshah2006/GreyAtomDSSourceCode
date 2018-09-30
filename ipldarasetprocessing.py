# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 09:59:40 2018

@author: Tushar Shah
"""

import os
import csv

os.chdir("D:\GreyAtom\Datasets")
#print(os.getcwd())

def match(match_code):
    list_1 = []
    for x in ipl[1:]:
        if x[0] == match_code:
            list_1.append(x)
    return list_1

def batsman_stats(batsman_name):
    runs = []
    match_played = []
    for x in ipl[1:]:
        if x[13] == batsman_name:
            runs.append(int(x[16]))
            match_played.append(x[0])
    strick_rate = (sum(runs) / len(runs) * 100)
    match_played = list(set(match_played))

    res = []
    for i,x in enumerate(match_played):
        mp = {}
        mp["Match " + str(i)] = x
        res.append(mp)

    result = {}
    for d in res:
        result.update(d)
        
    mp1 = []
    for x in list(result.items()):
        for y in match_played:
            if y in x:
                mp1.append(x[0])
                
    final = {}
    final["Matache_Played"] = mp1
    final["Strike_Rate"] = strick_rate
    return (final)

def batsman_bowled():
    batsman = []
    bowled_type = []
    for x in ipl[1:]:
        if x[21].find("bowled") != -1:
            bowled_type.append(x[21])
    bowled_type = list(set(bowled_type))
    
    for x in ipl[1:]:
        if x[10] == '2' and x[21] in bowled_type:
            batsman.append(x[13])    
    return batsman

#Project 2.1
ipl = []

with open("ipl_matches_small.csv") as csvfile:
    reader  = csv.reader(csvfile)
    for x in reader:
        ipl.append(x)
        
#set header in the tuple
header = tuple(ipl[0])

match_codes = list(set([x[0] for x in ipl[1:]]))

#project2.2
#set the team1
team_1 = [x[3] for x in ipl[1:]]

#set the team2
team_2 = [x[4] for x in ipl[1:]]

#change the list to set for team_1 and team_2
team_1 = set(team_1)
team_2 = set(team_2)

#retrieve the unique v alues from the set
teams = list(team_1.union(team_2))


#project 2.3
match_code = [x[0] for x in ipl[1:]]

first_match = []
for x in ipl[1:]:
    if x[0] == "392203":
        first_match.append(x)
first_batsman = first_match[0][13]

first_match = match("392203")
first_batsman = first_match[0][13]

second_matach = match("392197")
second_batsman = second_matach[0][13]

third_matach = match("392203")
third_batsman = third_matach[0][13]

fourth_matach = match("392212")
fourth_batsman = fourth_matach[0][13]

fifth_matach = match("501226")
fifth_batsman = fifth_matach[0][13]

sixth_matach = match("729297")
sixth_batsman = sixth_matach[0][13]

#project 2.4
#runs_SRT_first = strike_rate('SR Tendulkar', first_match)
#runs_SRT_second = strike_rate('SR Tendulkar', second_matach)
#runs_SRT_third = strike_rate('SR Tendulkar', third_matach)
#runs_SRT_fourth = strike_rate('SR Tendulkar', fourth_matach)
#runs_SRT_fifth = strike_rate('SR Tendulkar', fifth_matach)   
#runs_SRT_sixth = strike_rate('SR Tendulkar', sixth_matach)

#project 2.5

#How many match batsman played - needs to be implement
srt = batsman_stats('SR Tendulkar')

#set the string match number against each match code
#res = []
#for i,x in enumerate(match_codes):
#    mp={}
#    mp["Match " + str(i)] = x
#    res.append(mp)
#
#
#result = {}
#for d in res:
#    result.update(d)
#
#
#mp1 = []
#for x in list(result.items()):
#    for y in srt:
#        if y in x:
#            mp1.append(x[0])

batsman_name = batsman_bowled()


innings = [x[10] for x in ipl[1:]]
wicket_kind = [x[21] for x in ipl[1:]]
batsman = [x[13] for x in ipl[1:]]
extra_type = [x[19] for x in ipl[1:]]
extra_runs = [x[17] for x in ipl[1:]]

ext1=[]
ext2=[]
wickets=[]
players=[]

for x, y, z in zip(innings, wicket_kind, batsman):
    if x=='2' and len(y) != 0 and "bowled" in y:
        wickets.append(y)
        players.append(z)

for x,y in zip(innings, extra_runs):
    if x == '1' and len(y) != 0:
        ext1.append(int(y))
    elif x == '2' and len(y) != 0:
        ext2.append(int(y))

print(sum(ext1))
print(sum(ext2))

#print (len(ext1)/len(ext2))