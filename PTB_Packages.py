import pandas as pd
import os 
import re
from datetime import date
import json
import seaborn as sns
import matplotlib.pyplot as plt
import snowflake.connector
import numpy as np

# This is for Excel style 
heading_properties = [('font-size', '18px')]
cell_properties = [('font-size', '16px')]

dfstyle = [dict(selector="th", props=heading_properties),\
 dict(selector="td", props=cell_properties)]

        
# Target transformation
def Target_transformation(data,target_flag,tran_type):
    if tran_type=='log10':
        data=data[(data[target_flag]>0)]
        data['TARGET']=np.log10(data[target_flag])
        
    elif tran_type=='quantile_3':
        data=data[(data[target_flag]>0)]
        data['TARGET']=pd.qcut(data[target_flag],q=3, labels=['Low', 'Med','High'])
        
    elif tran_type=='IQR':
        data=data[(data[target_flag]>=0)]
        Target_Q1=np.quantile(data[target_flag], .25)
        Target_Q3=np.quantile(data[target_flag], .75)  
        data['TARGET']=np.where(data[target_flag]<Target_Q1, 'Low',
                                np.where(data[target_flag]<Target_Q3,'Med','High'))
        
    elif tran_type=='Normalization':
        data=data[(data[target_flag]>=0)]
        target_mean=np.mean(data[target_flag])
        target_sd=np.std(data[target_flag])
        data[target_flag]=(data[target_flag]-target_mean)/target_sd
    
    elif tran_type=='No Transformation':
        data['TARGET']=data[target_flag]
    
    if target_flag!='TARGET':
        data.drop(target_flag,axis=1,errors='ignore',inplace=True)
    
    print("Target: "+target_flag)
    return data

# Data_Prep function (Merge with Dnb, cleabit, missing value imputation)
def Data_Prep(base, dnb, clearbit, project_type, activity_include=True ,super_seg=None,product_type=None):
    #import pandas as pd
    #create dict for superseg_threshold
    superseg_threshold = {'strategic':{'overall':500000, 'ciam':300000, 'wf':300000},
                      'large':{'overall':200000, 'ciam':100000, 'wf':100000},
                      'commercial':{'overall':40000, 'ciam':40000, 'wf':25000}}

    #subset data to super segment if (specified) strategic, large, commerical, None
    if super_seg==None:
        base=base
    elif super_seg.find('strategic')!=-1: 
        # Strategic 
        base=base[base['ACCT_OWNER_SEGMENT'].isin(['Strategic'])]
        base['TARGET']=np.where(base['TARGET']>=superseg_threshold[super_seg][product_type], True, False)
    elif super_seg.find('large')!=-1:   
        #Large
        base=base[base['ACCT_OWNER_SEGMENT'].isin(['Enterprise-1', 'Enterprise-2', 'Federal'])]
        base['TARGET']=np.where(base['TARGET']>=superseg_threshold[super_seg][product_type], True, False)
    elif super_seg.find('commercial')!=-1:   
        #Commercial
        base=base[base['ACCT_OWNER_SEGMENT'].isin(['Central Gov','Corporate', 'Corporate-1', 'Corporate-2', 'Emerging',
                                            'Emerging-1','Emerging-2','Global', 'ICP', 'Mid-Market', 
                                            'NHS & Local Gov', 'Other', 'Regional', 'Renewals','SLED'])]
        base['TARGET']=np.where(base['TARGET']>=superseg_threshold[super_seg][product_type], True, False)
    elif super_seg==None:
        base=base
    
    # Define dnb columns to keep
    dnb_col=['Account ID 18', 'AccountToDnB: Employee Count Total', 'AccountToDnB: Revenue Trend Year', 
             'AccountToDnB: Sales Volume USD','AccountToDnB: Sales Volume Reliability Description', 
             'AccountToDnB: Parent Business Name', 'AccountToDnB: Parent D-U-N-S Number','AccountToDnB: SIC4 Code 1', 
             'AccountToDnB: SIC4 Code 1 Description','AccountToDnB: D-U-N-S Number', 'AccountToDnB: Location Type', 
             'AccountToDnB: Primary Address City', 'AccountToDnB: Primary Address County',
             'AccountToDnB: Primary Address State Province', 'AccountToDnB: Primary Address Country/Region', 
             'AccountToDnB: Web Address']
    
    # Merge base with dnb and clearbit data
    df=pd.merge(base, dnb[dnb_col], left_on='ACCOUNT_ID', right_on='Account ID 18', how='left')
    df=pd.merge(df, clearbit, left_on='ACCOUNT_ID', right_on='Account ID 18', how='left')
    
    #df['INDUSTRY']=np.where((df['INDUSTRY_NAICS']=='NA') | (df['INDUSTRY_NAICS'].isnull()), 
    #                    df['PRIMARY_INDUSTRY'], df['INDUSTRY_NAICS'] )
    
    # Data Cleansing 
    if project_type=='boolean':
        df['TARGET']=df['TARGET'].astype('bool')
    elif project_type=='multi-class':
        df['TARGET']=df['TARGET'].astype('str')
    elif project_type=='float':
        df['TARGET']=df['TARGET'].astype('float')
    else:
        df['TARGET']=df['TARGET'].astype('bool')
        print("Project Type is missing, will treat Target as boolean flag")
    
    df.drop(['Account ID 18_x', 'Account ID 18_y', 'HIGHEST_STAGE', 'REACHED_VP_SVP_IT', 
             'REACHED_DIRECTOR_IT','RESPONDED_VP_SVP_IT', 'RESPONDED_DIRECTOR_IT', 'SINCE_LAST_OPPTY',
            'X6SENSE_INTENT_SCORE_OKTA',   'X6SENSE_BUYING_STAGE','X6SENSE_PROFILE_FIT_OKTA',
             'TOTAL_AMOUNT', 'HAS_QUOTE','MIN_BOOKING_DATE',  
                   'ACCT_OWNER_SEGMENT', 'ACCT_OWNER_REGION', 'ACCT_OWNER_AREA', 'ACCT_OWNER_GEOGRAPHY'
             ,'PARTNER_INVOLVEMENT', 'CUSTOMER_ACQUISITION_DATE',
             'BUSINESS_TYPE', 'NUMBER_OF_PARTNERS','RNK',
             'ACCOUNT_ID_WFADP', 'ACCOUNT_ID_WFADP_F', 
             'ACCOUNT_ID_WFAC', 'ACCOUNT_ID_WFAC_F', 'ACCOUNT_ID_CIAMADP', 'ACCOUNT_ID_WFADP_G_F',
             'ACCOUNT_ID_CIAMAC_F','ACCOUNT_ID_WFADP_L', 'ACCOUNT_ID_WFAC_G_F', 'ACCOUNT_ID_CIAMAC_G_F',
             'ACCOUNT_ID_WFAC_L', 'ACCOUNT_ID_CIAMAC_L', 
            'UPSELL_RENEWAL_TOTAL_SINCE',    'CARR_USD_SNAPSHOT','WF_RENEWAL_TOTAL_SINCE', 'CIAM_RENEWAL_TOTAL_SINCE',
             'TARGET_WF', 'TARGET_CIAM' ,'WF_BOOKING','CIAM_BOOKING' , 'TOTAL_BOOKING',
             'PRIMARY_INDUSTRY','SUB_INDUSTRY','SKU_TYPE'
            ], axis=1,errors='ignore',inplace=True)

    if activity_include==False:
        df.drop(['QL','SMS','INQUIRY', 'EMAIL_QL', 'WEB_QL', 'FREE_TRIAL_QL','MKT_EMAIL_SENT', 'MKT_EMAIL_OPEN_RATE'
                 ,'MKT_EMAIL_CTR', 'EMAIL', 'CALL', 'EMAIL_RESPONSE_RATE', 'CALL_RESPONSE_RATE', 'CALL_RESPONDED',
                 'EMAIL_RESPONDED' ,'SINCE_LAST_SMS', 'SINCE_LAST_QL', 'ATTENDED_OKTANE', 'MAX_TITLE_REACHED',
                 'MAX_TITLE_RESPONDED','ATTENDED_FIELD_EVENT','NUM_OPPTY_WITH_PARTNER', 'NUM_CASES_CREATED',
                 'NB_OPPTY', 'RESPONDED_CIO_CISO', 'ATTENDED_EBC', 'SINCE_LAST_INQ', 'REACHED_CIO_CISO',
                 'ATTENDED_CONFERENCE'
            ], axis=1,errors='ignore',inplace=True)
        
    # Missing Value Imputation
    values = {#"EMAIL_RESPONSE_RATE": 0, 
              #"CALL_RESPONSE_RATE": 0,
              "EMPLOYEES_MASTER": 0, 
              "SMS": 0, 
              "Clearbit: Company Raised (converted)": 0,
             "NUM_OPPTY_WITH_PARTNER":0,
              "QL": 0,
              "AccountToDnB: Parent D-U-N-S Number":0,
              "INQUIRY": 0,
              "NB_OPPTY": 0,
              "AccountToDnB: Sales Volume USD": 0,
              "AccountToDnB: Employee Count Total": 0,
               "AccountToDnB: Revenue Trend Year": 0,
                "EMAIL": 0,
                "CALL":0
             }
    df.fillna(value=values, inplace=True)
    
    print("Input data row count: " + str(base.shape[0]))
    print("Output data row count: " + str(df.shape[0]))      
        
    return df