
# import math
# make a variable for whether the user picks 'investment' or 'bond' under choice_of_calc
# using if statement - if choice of calculator == investment we proceed with investment calc 
# using elif statment - if choice of calculator == bond we proceed with bond calc
# ensure that program is not case sensitive with input of 'investment' or 'bond' with the use of lower() function 
# using else statement - output an appropriate error message if any other word is typed in 
# if their choice of calc is 'investment'
    # Ask user for deposit amount 
    # Ask user for interest rate 
    # Ask user how many years they're investing for 
# Ask user if they want simple or compound method
# if user asks for simple method:
    # Output the return on their investment using the forumla A = P * (1 + r * t)
# if user asks for compound method:
    # Output the return on their investment using the forumla A = P * math.pow((1 + r),t)
# if their choice of calc is 'bond'
    # Ask user how much is their houses current value 
    # Ask user the interest rate 
    # Ask user how many many months they plan to repay the bond in 
# Output the repayment figuire using the forumla repayment = (i * p) / (1 - (1 + i)**(-months))

###################################################################################################
# Start of program: 
import math

# putting each calculator into a variable depending on which one they pick with an if statement

choice_of_calc = input('''investment - to calculate the amount of interest you'll earn on your investment
bond       - to calculate the amount you'll have to pay on a home loan 
                       
all amounts are measured in GBP currency
                       
Enter either 'investment' or 'bond' from the menu above to proceed: ''').lower()                                                                 # Lower function to eradicate any program issues with user capitalisation in their selection 


if choice_of_calc == "investment" :
    deposit = int(input("Please enter the amount you are depositing for your investment (no commas interupting any digits): "))                  # deposit amounts are dealt in GBP currency 
    rate = int(input("Please enter the interest rate (without the % symbol):  "))                                                                # rate will be symbol for interest rate without being divided by 100 
    t = int(input("Please enter how many years you want to be in this investment for: "))                                                        
    interest = input("Please enter whether you want 'simple' or 'compound' interest here: ")
    if interest == "simple" :
        r = rate / 100                                                                                                                           # r will be symbol for the interest rate divided by 100
        simple_r = deposit *(1 + r*t)                                                                                                            # Variable that holds the simple interest formula 
        print(f"You will recieve {round(simple_r, 2)} GBP after {t} years at a {rate}% interest rate using the simple method!")                  # Rounding function to round figure to 2 decimal places to replicate a real financial figure
    elif interest == "compound" : 
        r = rate / 100 
        compound_r = deposit * math.pow((1+r),t)                                                                                                 # Variable that holds the compound interest formula 
        print(f"You will recieve {round(compound_r, 2)} GBP after {t} years at a {rate}% interest rate using the compounding method!")           
elif choice_of_calc == "bond" :
    p = int(input("Please enter the current value of the house (no commas interupting any digits): "))                                           # Valued in GBP currency
    rate = int(input("Please enter the interest rate (without the % symbol): "))
    months = int(input("Please enter the number of months you planned to repay the bond: "))
    i = (rate /100) / 12                                                                                                                         # This forumla in the variable of i is the calculate the rate on each month 
    repayment = (i * p)/(1 - (1 + i)**(-months))                                                                                                 # Formula for the monthly repyaments
    print(f"Your repayemnt of the bond each month will be {round(repayment, 2)} GBP for the next {months} months ")                              # rounding to 2 decimal places using round() function 
else:
    print("You have inputted invalid values, please try again. ")                                                                                # Error message for invalid inputs

                                                                                                                       

        
