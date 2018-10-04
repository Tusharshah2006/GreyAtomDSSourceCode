# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Problem Statement - find the house price of the property

import os
import pandas as pd
import numpy as np
os.chdir("D:\GreyAtom\Datasets")

dc_listings = pd.read_csv("dc_airbnb.csv")
dc_listings_bkup = dc_listings.copy()

#Define all functions starts here**************************************
def getHousePrice(our_accomodation, no_of_rows, df, feature):
    df = cleanData(df)
    df = getAccommodationDistance(our_accomodation, df, feature)
    df = randomize(df)
    mean = findMdeanofDistanceZero(df, no_of_rows)
    findError(df, mean)
    rmse = np.sqrt(df["error_squered"].mean())
    return mean, rmse

def cleanData(df):
    df["price"] = df["price"].str.replace(',', '')
    df["price"] = df["price"].str.replace('$', '')
    df["price"] = df["price"].astype(float)
    
    df = df.drop(["security_deposit", "latitude", "longitude", "cleaning_fee", "host_response_rate", "host_acceptance_rate", "zipcode", "host_listings_count"], axis = 1)
    
    #z-square normalization
    df["maximum_nights"] = (df["maximum_nights"] - df["maximum_nights"].mean())/df["maximum_nights"].std()
    return df
    
def getAccommodationDistance(our_accomodation, df, feature):
    df["distance"] = abs(df[feature] - our_accomodation)
    return df

def randomize(df):
    np.random.seed(6)
    return df.loc[np.random.permutation(len(df))]
   
def findMdeanofDistanceZero(df, no_of_rows):
    meanvalue = df[(df["distance"] == 0)][:no_of_rows]["price"].mean()
    return meanvalue

def findError(df, mean):
    df["error_squered"] = (df["price"] - mean)**2
#Define all functions - ends here***************************************

meanPrice, rmserror = getHousePrice(3, 5, dc_listings_bkup, "bathrooms")
print(meanPrice, rmserror)