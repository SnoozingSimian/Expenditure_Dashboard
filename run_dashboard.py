import datetime

import streamlit as st
import pandas as pd

from data_processors.Classifiers import ZeroShotClassifer


classifier = ZeroShotClassifer()

df = pd.read_csv("./data/expenditures_Apr-11-2022_Jul-13-2023.tsv", sep="\t")
df["Txn Date"] = pd.to_datetime(df["Txn Date"])
df.index = df["Txn Date"]

max_date = df["Txn Date"].max()
min_date = df["Txn Date"].min()

st.markdown(
    f'**Data available from <span style="color:DodgerBlue;">{min_date.strftime("%b %d, %Y")}</span> to <span style="color:DodgerBlue;">{max_date.strftime("%b %d, %Y")}</span>**',
    unsafe_allow_html=True,
)

st.sidebar.header("Expenditure Categories")
options = st.sidebar.multiselect(
    "**Select Expenditure Categories**",
    [
        "Transport",
        "Eating Out",
        "Swiggy",
        "Instamart",
        "Credit Card Bill",
        "Bill Payments",
        "Merchant Payment",
        "Point of Sales Payment",
    ],
)

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Select start date", min_date)
end_date = col2.date_input("Select end date", max_date)

if st.button("Build Dataframe"):
    classifier.categories = options

    df = df[start_date:end_date]

    categories = []
    for i, row in df.iterrows():
        categories.append(classifier.classify(row["Description"]))

    df["Categories"] = categories

    st.write(df)

# """General Categories
# Housing

# Mortgage or rent
# Property taxes
# Household repairs
# HOA fees
# Transportation

# Car payment
# Car warranty
# Gas
# Tires
# Maintenance and oil changes
# Parking fees
# Repairs
# Registration and DMV Fees
# Food

# Groceries
# Restaurants
# Pet food
# Utilities

# Electricity
# Water
# Garbage
# Phones
# Cable
# Internet
# Clothing

# Adults’ clothing
# Adults’ shoes
# Children’s clothing
# Children’s shoes
# Medical/Healthcare

# Primary care
# Dental care
# Specialty care (dermatologists, orthodontics, optometrists, etc.)
# Urgent care
# Medications
# Medical devices
# Insurance

# Health insurance
# Homeowner’s or renter’s insurance
# Home warranty or protection plan
# Auto insurance
# Life insurance
# Disability insurance
# Household Items/Supplies

# Toiletries
# Laundry detergent
# Dishwasher detergent
# Cleaning supplies
# Tools
# Personal

# Gym memberships
# Haircuts
# Salon services
# Cosmetics (like makeup or services like laser hair removal)
# Babysitter
# Subscriptions
# Debt

# Personal loans
# Student loans
# Credit cards
# Retirement

# Financial planning
# Investing
# Education

# Children’s college
# Your college
# School supplies
# Books
# Savings

# Emergency fund
# Big purchases like a new mattress or laptop
# Other savings
# Gifts/Donations

# Birthday
# Anniversary
# Wedding
# Christmas
# Special occasion
# Charities
# Entertainment

# Alcohol and/or bars
# Games
# Movies
# Concerts
# Vacations
# Subscriptions (Netflix, Amazon, Hulu, etc.)
# """
