# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 22:12:19 2018

@author: Tushar Shah
"""

def fibonacci(n):
    if (n == 0 or n == 1): 
        return n
    else: 
        return (fibonacci(n-1) + fibonacci(n-2))

def isPrimeNumber(num):
    for i in range(2, num):
        if(num % i) == 0:
            break
    else:
        return True

number = int(input("Enter Total Number of Dice: "))
FebonacciOrPrime = int(input("Enter 1 for Febonacci Series or 2 for Prime Number : "))

if(number > 0):
    if(FebonacciOrPrime == 1):
        #Febonacci Series Call
        print("Febonacci Series :")
        for i in range(number):
            print(fibonacci(i))
    elif(FebonacciOrPrime == 2):    
        #Prime Number Call
        primrnumber = []
        for num in range(number, number*6+1):
            if(isPrimeNumber(num)):
                primrnumber.append(num)
        print(primrnumber)
    else:
        print("Please enter either 1 for Febonacci Series or 2 for Prime Number")
else:
    print("Entered Number is not a valid number")