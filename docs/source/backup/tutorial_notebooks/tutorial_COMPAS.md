# Using *fairlib* with Structured Inputs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HanXudong/fairlib/blob/main/tutorial/COMPAS.ipynb)

In this tutorial we will:
- Show how to add a model for structural classification
- Show how to add a dataloader with structured data preprocessing

We will be using the Northpointe's Correctional Offender Management Profiling for Alternative Sanctions (COMPAS) score, used in states like California and Florida.

## Installation

Again, the first step will be installing our libarary


```python
!pip install fairlib
```

    Collecting fairlib
      Downloading fairlib-0.0.1-py3-none-any.whl (61 kB)
    [K     |████████████████████████████████| 61 kB 4.1 MB/s 
    [?25hRequirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from fairlib) (4.63.0)
    Collecting transformers
      Downloading transformers-4.18.0-py3-none-any.whl (4.0 MB)
    [K     |████████████████████████████████| 4.0 MB 16.9 MB/s 
    [?25hRequirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from fairlib) (1.3.5)
    Requirement already satisfied: docopt in /usr/local/lib/python3.7/dist-packages (from fairlib) (0.6.2)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.7/dist-packages (from fairlib) (1.0.2)
    Requirement already satisfied: seaborn in /usr/local/lib/python3.7/dist-packages (from fairlib) (0.11.2)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from fairlib) (3.2.2)
    Collecting pickle5
      Downloading pickle5-0.0.12-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (256 kB)
    [K     |████████████████████████████████| 256 kB 45.9 MB/s 
    [?25hRequirement already satisfied: torch in /usr/local/lib/python3.7/dist-packages (from fairlib) (1.10.0+cu111)
    Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from fairlib) (1.21.5)
    Requirement already satisfied: PyYAML in /usr/local/lib/python3.7/dist-packages (from fairlib) (3.13)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->fairlib) (0.11.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->fairlib) (3.0.7)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->fairlib) (1.4.0)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->fairlib) (2.8.2)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from kiwisolver>=1.0.1->matplotlib->fairlib) (3.10.0.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.1->matplotlib->fairlib) (1.15.0)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas->fairlib) (2018.9)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->fairlib) (1.1.0)
    Requirement already satisfied: scipy>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->fairlib) (1.4.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->fairlib) (3.1.0)
    Collecting sacremoses
      Downloading sacremoses-0.0.49-py3-none-any.whl (895 kB)
    [K     |████████████████████████████████| 895 kB 41.5 MB/s 
    [?25hCollecting huggingface-hub<1.0,>=0.1.0
      Downloading huggingface_hub-0.5.1-py3-none-any.whl (77 kB)
    [K     |████████████████████████████████| 77 kB 6.5 MB/s 
    [?25hRequirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.7/dist-packages (from transformers->fairlib) (2019.12.20)
    Collecting tokenizers!=0.11.3,<0.13,>=0.11.1
      Downloading tokenizers-0.11.6-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (6.5 MB)
    [K     |████████████████████████████████| 6.5 MB 28.3 MB/s 
    [?25hRequirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from transformers->fairlib) (2.23.0)
    Collecting PyYAML
      Downloading PyYAML-6.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (596 kB)
    [K     |████████████████████████████████| 596 kB 49.2 MB/s 
    [?25hRequirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from transformers->fairlib) (4.11.3)
    Requirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from transformers->fairlib) (3.6.0)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.7/dist-packages (from transformers->fairlib) (21.3)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata->transformers->fairlib) (3.7.0)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->transformers->fairlib) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->transformers->fairlib) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->transformers->fairlib) (2021.10.8)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->transformers->fairlib) (1.24.3)
    Requirement already satisfied: click in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers->fairlib) (7.1.2)
    Installing collected packages: PyYAML, tokenizers, sacremoses, huggingface-hub, transformers, pickle5, fairlib
      Attempting uninstall: PyYAML
        Found existing installation: PyYAML 3.13
        Uninstalling PyYAML-3.13:
          Successfully uninstalled PyYAML-3.13
    Successfully installed PyYAML-6.0 fairlib-0.0.1 huggingface-hub-0.5.1 pickle5-0.0.12 sacremoses-0.0.49 tokenizers-0.11.6 transformers-4.18.0
    


```python
import fairlib
```

## Download and preprocess the COMPAS dataset

https://github.com/google-research/google-research/blob/master/group_agnostic_fairness/data_utils/CreateCompasDatasetFiles.ipynb


```python
import os
```


```python
os.makedirs("data", exist_ok=True)
```


```python
!wget --no-check-certificate --content-disposition "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv" -O "data/compas-scores-two-years.csv"
```

    --2022-04-08 15:09:58--  https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 2546489 (2.4M) [text/plain]
    Saving to: ‘data/compas-scores-two-years.csv’
    
    data/compas-scores- 100%[===================>]   2.43M  --.-KB/s    in 0.03s   
    
    2022-04-08 15:09:58 (70.2 MB/s) - ‘data/compas-scores-two-years.csv’ saved [2546489/2546489]
    
    


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
```


```python
pd.options.display.float_format = '{:,.2f}'.format
dataset_base_dir = "data/"
dataset_file_name = 'compas-scores-two-years.csv'
```


```python
file_path = os.path.join(dataset_base_dir,dataset_file_name)
temp_df = pd.read_csv(file_path)

# Columns of interest
columns = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
                'age', 
                'c_charge_degree', 
                'c_charge_desc',
                'age_cat',
                'sex', 'race',  'is_recid']
target_variable = 'is_recid'
target_value = 'Yes'

# Drop duplicates
temp_df = temp_df[['id']+columns].drop_duplicates()
df = temp_df[columns].copy()

# Convert columns of type ``object`` to ``category`` 
df = pd.concat([
        df.select_dtypes(include=[], exclude=['object']),
        df.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
        ], axis=1).reindex(df.columns, axis=1)

# Binarize target_variable
df['is_recid'] = df.apply(lambda x: 'Yes' if x['is_recid']==1.0 else 'No', axis=1).astype('category')

# Process protected-column values
race_dict = {'African-American':'Black','Caucasian':'White'}
df['race'] = df.apply(lambda x: race_dict[x['race']] if x['race'] in race_dict.keys() else 'Other', axis=1).astype('category')
```


```python
df
```





  <div id="df-7d3a143f-c642-4228-9663-4fe7fafe02bd">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>juv_fel_count</th>
      <th>juv_misd_count</th>
      <th>juv_other_count</th>
      <th>priors_count</th>
      <th>age</th>
      <th>c_charge_degree</th>
      <th>c_charge_desc</th>
      <th>age_cat</th>
      <th>sex</th>
      <th>race</th>
      <th>is_recid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>69</td>
      <td>F</td>
      <td>Aggravated Assault w/Firearm</td>
      <td>Greater than 45</td>
      <td>Male</td>
      <td>Other</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>F</td>
      <td>Felony Battery w/Prior Convict</td>
      <td>25 - 45</td>
      <td>Male</td>
      <td>Black</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>24</td>
      <td>F</td>
      <td>Possession of Cocaine</td>
      <td>Less than 25</td>
      <td>Male</td>
      <td>Black</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>23</td>
      <td>F</td>
      <td>Possession of Cannabis</td>
      <td>Less than 25</td>
      <td>Male</td>
      <td>Black</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>43</td>
      <td>F</td>
      <td>arrest case no charge</td>
      <td>25 - 45</td>
      <td>Male</td>
      <td>Other</td>
      <td>No</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7209</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>23</td>
      <td>F</td>
      <td>Deliver Cannabis</td>
      <td>Less than 25</td>
      <td>Male</td>
      <td>Black</td>
      <td>No</td>
    </tr>
    <tr>
      <th>7210</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>23</td>
      <td>F</td>
      <td>Leaving the Scene of Accident</td>
      <td>Less than 25</td>
      <td>Male</td>
      <td>Black</td>
      <td>No</td>
    </tr>
    <tr>
      <th>7211</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>57</td>
      <td>F</td>
      <td>Aggravated Battery / Pregnant</td>
      <td>Greater than 45</td>
      <td>Male</td>
      <td>Other</td>
      <td>No</td>
    </tr>
    <tr>
      <th>7212</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>33</td>
      <td>M</td>
      <td>Battery on Law Enforc Officer</td>
      <td>25 - 45</td>
      <td>Female</td>
      <td>Black</td>
      <td>No</td>
    </tr>
    <tr>
      <th>7213</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>23</td>
      <td>F</td>
      <td>Possession of Ethylone</td>
      <td>Less than 25</td>
      <td>Female</td>
      <td>Other</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>7214 rows × 11 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7d3a143f-c642-4228-9663-4fe7fafe02bd')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-7d3a143f-c642-4228-9663-4fe7fafe02bd button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7d3a143f-c642-4228-9663-4fe7fafe02bd');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# Create splits
train_df, test_df = train_test_split(df, test_size=0.30, random_state=42)
train_df, dev_df = train_test_split(train_df, test_size=0.1, random_state=42)
```


```python
cat_cols = train_df.select_dtypes(include='category').columns
vocab_dict = {}
for col in cat_cols:
  vocab_dict[col] = list(set(train_df[col].cat.categories))
print(vocab_dict)
```

    {'c_charge_degree': ['F', 'M'], 'c_charge_desc': ['Aiding Escape', 'Cash Item w/Intent to Defraud', 'Lewd or Lascivious Molestation', 'Attempted Burg/Convey/Unocc', 'Aggravated Assault W/o Firearm', 'Disorderly Conduct', 'Crlty Twrd Child Urge Oth Act', 'Agg Battery Grt/Bod/Harm', 'Poss Contr Subst W/o Prescript', 'Soliciting For Prostitution', 'Tresspass Struct/Conveyance', 'Unlaw Lic Use/Disply Of Others', 'Aggrav Battery w/Deadly Weapon', 'Opert With Susp DL 2ND Offense', 'Deliver Cannabis', 'Battery', 'Intoxicated/Safety Of Another', 'Fail Sex Offend Report Bylaw', 'Attempted Deliv Control Subst', 'Lewd Act Presence Child 16-', 'Violation Of Boater Safety Id', 'Del Morphine at/near Park', 'Possession of LSD', 'Possession Of Diazepam', 'Possession Of Anabolic Steroid', 'Aggravated Battery (Firearm)', 'Possession Of Carisoprodol', 'Possession Of Cocaine', 'Purchase/P/W/Int Cannabis', 'Attempt Armed Burglary Dwell', 'Cause Anoth Phone Ring Repeat', 'Sexual Performance by a Child', 'Purchase Cannabis', 'Falsely Impersonating Officer', 'False Motor Veh Insurance Card', 'Strong Armed  Robbery', 'Deliver Cannabis 1000FTSch', 'Possession of Morphine', 'Possession of Methadone', 'Alcoholic Beverage Violation-FL', 'Battery On A Person Over 65', 'Lewd/Lasc Battery Pers 12+/<16', 'Sel/Pur/Mfr/Del Control Substa', 'Sound Articles Over 100', 'Petit Theft', 'DWLS Canceled Disqul 1st Off', 'Present Proof of Invalid Insur', 'Sex Batt Faml/Cust Vict 12-17Y', 'Felony Battery w/Prior Convict', 'Use Of 2 Way Device To Fac Fel', 'Possession of Oxycodone', 'Poss Firearm W/Altered ID#', 'Neglect Child / No Bodily Harm', 'Sex Battery Deft 18+/Vict 11-', 'Poss Unlaw Issue Driver Licenc', 'Att Burgl Struc/Conv Dwel/Occp', 'Robbery W/Firearm', 'Trans/Harm/Material to a Minor', 'Lve/Scen/Acc/Veh/Prop/Damage', 'Burglary Conveyance Occupied', 'Child Abuse', 'Robbery / Weapon', 'Offer Agree Secure/Lewd Act', 'Poss/Sell/Deliver Clonazepam', 'Giving False Crime Report', 'Possession Of Buprenorphine', 'Burglary Conveyance Assault/Bat', 'Prostitution/Lewdness/Assign', 'Aggress/Panhandle/Beg/Solict', 'Tampering with a Victim', 'Crim Use Of Personal Id Info', 'Use Computer for Child Exploit', 'Possession Child Pornography', 'Uttering Forged Bills', 'Fleeing or Eluding a LEO', 'Possession of Alcohol Under 21', 'Possess/Use Weapon 1 Deg Felon', 'Battery Spouse Or Girlfriend', 'DUI- Enhanced', 'Prostitution', 'DUI - Property Damage/Personal Injury', 'Traff In Cocaine <400g>150 Kil', 'Solicit Purchase Cocaine', 'Delivery of Heroin', 'Crimin Mischief Damage $1000+', 'Unlawful Use Of Police Badges', 'Fail Register Vehicle', 'Poss Pyrrolidinobutiophenone', 'Sexual Battery / Vict 12 Yrs +', 'False Ownership Info/Pawn Item', 'Forging Bank Bills/Promis Note', 'Crim Attempt/Solic/Consp', 'Aggravated Battery / Pregnant', 'Leave Accd/Attend Veh/Less $50', 'Theft/To Deprive', 'Grand Theft (motor Vehicle)', 'Compulsory Attendance Violation', 'Murder In 2nd Degree W/firearm', 'Poss/pur/sell/deliver Cocaine', 'Criminal Mischief Damage <$200', 'Possession Of Clonazepam', 'Burglary With Assault/battery', 'Poss Counterfeit Payment Inst', 'Possession of Cannabis', 'False Info LEO During Invest', 'Dealing In Stolen Property', 'Accessory After the Fact', 'Unlawful Conveyance of Fuel', 'Delivery of 5-Fluoro PB-22', 'Battery On Fire Fighter', 'Contradict Statement', 'Possess Controlled Substance', 'Computer Pornography', 'Grand Theft in the 3rd Degree', 'Sale/Del Cannabis At/Near Scho', 'Poss of Firearm by Convic Felo', 'DOC/Cause Public Danger', 'Attempted Burg/struct/unocc', 'Principal In The First Degree', 'Resist/Obstruct W/O Violence', 'Unlaw Use False Name/Identity', 'Felony Driving While Lic Suspd', 'Possession of Cocaine', 'Deliver Cocaine 1000FT Park', 'Del Cannabis At/Near Park', 'Failure To Pay Taxi Cab Charge', 'Traffick Oxycodone     4g><14g', 'Fel Drive License Perm Revoke', 'Murder in 2nd Degree', 'Possession Of Fentanyl', 'Poss F/Arm Delinq', 'Purchasing Of Alprazolam', 'Trespass Struct/Convey Occupy', 'Poss/Sell/Del Cocaine 1000FT Sch', 'False Name By Person Arrest', 'Fail To Redeliv Hire/Leas Prop', 'Del of JWH-250 2-Methox 1-Pentyl', 'Possession Of Heroin', 'Robbery W/Deadly Weapon', 'Trespass Structure/Conveyance', 'Leaving the Scene of Accident', 'Deliver Cocaine 1000FT Store', 'Driving License Suspended', 'Carry Open/Uncov Bev In Pub', 'Criminal Attempt 3rd Deg Felon', 'Deliver Alprazolam', 'Agg Assault W/int Com Fel Dome', 'Use Scanning Device to Defraud', 'Possess Weapon On School Prop', 'Fail Register Career Offender', 'Corrupt Public Servant', 'Poss Wep Conv Felon', 'Unlicensed Telemarketing', 'Del 3,4 Methylenedioxymethcath', 'Arson in the First Degree', 'Poss Similitude of Drivers Lic', 'Grand Theft (Motor Vehicle)', 'Delivery Of Drug Paraphernalia', 'Carjacking with a Firearm', 'Poss Pyrrolidinovalerophenone W/I/D/S', 'Offn Against Intellectual Prop', 'Burglary Assault/Battery Armed', 'Theft', 'DUI Level 0.15 Or Minor In Veh', 'Neglect Child / Bodily Harm', 'Harm Public Servant Or Family', 'Poss Tetrahydrocannabinols', 'Opert With Susp DL 2nd Offens', 'Robbery Sudd Snatch No Weapon', 'Assault', 'Grand Theft of the 2nd Degree', 'Unauth C/P/S Sounds>1000/Audio', 'Viol Pretrial Release Dom Viol', 'Obstruct Fire Equipment', 'Battery on Law Enforc Officer', 'Uttering Worthless Check +$150', 'Poss Unlaw Issue Id', 'Unlaw LicTag/Sticker Attach', 'Disrupting School Function', 'Fail Obey Driv Lic Restrictions', 'Felony Committing Prostitution', 'Agg Assault Law Enforc Officer', 'Throw In Occupied Dwell', 'Aggravated Assault W/Dead Weap', 'Sale/Del Counterfeit Cont Subs', 'Felony Battery (Dom Strang)', 'Criminal Mischief', 'Possession Of 3,4Methylenediox', 'Tresspass in Structure or Conveyance', 'Traffick Amphetamine 28g><200g', 'Throw Deadly Missile Into Veh', 'Burglary Conveyance Armed', 'Possess Cannabis/20 Grams Or Less', 'Lease For Purpose Trafficking', 'Tamper With Witness/Victim/CI', 'D.U.I. Serious Bodily Injury', 'Lewd/Lasciv Molest Elder Persn', 'Possession of Benzylpiperazine', 'Violation License Restrictions', 'Issuing a Worthless Draft', 'Throw Missile Into Pub/Priv Dw', 'Pos Cannabis For Consideration', 'Attempted Robbery  No Weapon', 'Possession of Ethylone', 'Aggravated Assault w/Firearm', 'Susp Drivers Lic 1st Offense', 'Purchase Of Cocaine', 'Possess Mot Veh W/Alt Vin #', 'Viol Injunct Domestic Violence', 'Felony DUI (level 3)', 'Sell or Offer for Sale Counterfeit Goods', 'Poss 3,4 MDMA (Ecstasy)', 'Fabricating Physical Evidence', 'Possess w/I/Utter Forged Bills', 'Threat Public Servant', 'Insurance Fraud', 'Extradition/Defendants', 'Traffic Counterfeit Cred Cards', 'Petit Theft $100- $300', 'Att Burgl Unoccupied Dwel', 'Poss Cntrft Contr Sub w/Intent', 'Abuse Without Great Harm', 'arrest case no charge', 'DUI - Enhanced', 'Grand Theft Dwell Property', 'Resist Officer w/Violence', 'False 911 Call', 'Poss Pyrrolidinovalerophenone', 'Trespass Other Struct/Conve', 'Fighting/Baiting Animals', 'Possess Cannabis 1000FTSch', 'Possession of Hydromorphone', 'Grand Theft on 65 Yr or Older', 'Interfere W/Traf Cont Dev RR', 'Exhibition Weapon School Prop', 'Imperson Public Officer or Emplyee', 'Agg Fleeing and Eluding', 'Viol Injunction Protect Dom Vi', 'Traffick Hydrocodone   4g><14g', 'Refuse to Supply DNA Sample', 'Pos Methylenedioxymethcath W/I/D/S', 'Trespass Struct/Conveyance', 'Restraining Order Dating Viol', 'Tamper With Victim', 'Compulsory Sch Attnd Violation', 'Deliver Cocaine', 'Reckless Driving', 'Fail To Obey Police Officer', 'Deliver 3,4 Methylenediox', 'Contribute Delinquency Of A Minor', 'Open Carrying Of Weapon', 'Burglary Conveyance Unoccup', 'Grand Theft in the 1st Degree', 'Possession Of Amphetamine', 'Burglary Structure Assault/Batt', 'Possession of Codeine', 'Harass Witness/Victm/Informnt', 'Aggr Child Abuse-Torture,Punish', 'Poss of Vessel w/Altered ID NO', 'Carrying Concealed Firearm', 'Escape', 'Poss Meth/Diox/Meth/Amp (MDMA)', 'Aggravated Battery', 'Criminal Mischief>$200<$1000', 'Att Tamper w/Physical Evidence', 'Possession Of Methamphetamine', 'Robbery / No Weapon', 'Possession Of Alprazolam', 'Refuse Submit Blood/Breath Test', 'Trespass Structure w/Dang Weap', 'Bribery Athletic Contests', 'Felon in Pos of Firearm or Amm', 'Posses/Disply Susp/Revk/Frd DL', 'Exposes Culpable Negligence', 'Leave Acc/Attend Veh/More $50', 'Agg Abuse Elderlly/Disabled Adult', 'Interference with Custody', 'Tampering With Physical Evidence', 'Counterfeit Lic Plates/Sticker', 'Trespass On School Grounds', 'PL/Unlaw Use Credit Card', 'Possess Drug Paraphernalia', 'Cruelty Toward Child', 'Poss Oxycodone W/Int/Sell/Del', 'Exploit Elderly Person 20-100K', 'Solic to Commit Battery', 'Stalking (Aggravated)', 'Depriv LEO of Protect/Communic', 'Defrauding Innkeeper $300/More', 'Lewd/Lasc Exhib Presence <16yr', 'Prostitution/Lewd Act Assignation', 'Poss Alprazolam W/int Sell/Del', 'Poss Of RX Without RX', 'Unauthorized Interf w/Railroad', 'False Imprisonment', 'Burglary Structure Unoccup', 'Unemployment Compensatn Fraud', 'Flee/Elude LEO-Agg Flee Unsafe', 'DWI w/Inj Susp Lic / Habit Off', 'DUI/Property Damage/Persnl Inj', 'Conspiracy to Deliver Cocaine', 'Fail To Secure Load', 'Misuse Of 911 Or E911 System', 'Pos Cannabis W/Intent Sel/Del', 'Money Launder 100K or More Dols', 'Leaving Acc/Unattended Veh', 'Burglary Dwelling Armed', 'Burglary Dwelling Occupied', 'Trespassing/Construction Site', 'Culpable Negligence', 'Burgl Dwel/Struct/Convey Armed', 'Consp Traff Oxycodone  4g><14g', 'Battery Emergency Care Provide', 'Shoot In Occupied Dwell', 'Hiring with Intent to Defraud', 'Poss Of 1,4-Butanediol', 'Fraud Obtain Food or Lodging', 'License Suspended Revoked', 'Murder in the First Degree', 'Deliver Cocaine 1000FT School', 'Kidnapping / Domestic Violence', 'Attempted Robbery  Weapon', 'Aggrav Stalking After Injunctn', 'Unl/Disturb Education/Instui', 'Fleeing Or Attmp Eluding A Leo', 'Manslaughter W/Weapon/Firearm', 'Possess Tobacco Product Under 18', 'Viol Prot Injunc Repeat Viol', 'Battery On Parking Enfor Speci', 'DUI Property Damage/Injury', 'Tamper With Witness', 'Aide/Abet Prostitution Lewdness', 'Possession Of Lorazepam', 'Poss3,4 Methylenedioxymethcath', 'Burglary Unoccupied Dwelling', 'Carjacking w/o Deadly Weapon', 'Sell Cannabis', 'Sell Conterfeit Cont Substance', 'Fail To Redeliver Hire Prop', 'Attempted Robbery Firearm', 'Grand Theft Firearm', 'Consp Traff Oxycodone 28g><30k', 'Agg Fleeing/Eluding High Speed', 'Consume Alcoholic Bev Pub', 'Grand Theft of a Fire Extinquisher', 'Possession Firearm School Prop', 'Possession of Hydrocodone', 'Voyeurism', 'Deliver Cocaine 1000FT Church', 'Aggravated Battery (Firearm/Actual Possession)', 'Poss/Sell/Del/Man Amobarbital', 'Uttering Forged Credit Card', 'Grand Theft In The 3Rd Degree', 'Aggrav Child Abuse-Causes Harm', 'Poss Cocaine/Intent To Del/Sel', 'Structuring Transactions', 'Conspiracy Dealing Stolen Prop', 'Burglary Dwelling Assault/Batt', 'Expired DL More Than 6 Months', 'Fraudulent Use of Credit Card', 'Possession of Butylone', 'Offer Agree Secure For Lewd Act', 'Battery on a Person Over 65', 'Att Burgl Conv Occp', 'Felony DUI - Enhanced', 'Ride Tri-Rail Without Paying', 'Simulation of Legal Process', 'Introduce Contraband Into Jail', 'Poss Trifluoromethylphenylpipe', 'Lewdness Violation', 'Dealing in Stolen Property', 'Driving Under The Influence', 'Gambling/Gamb Paraphernalia', 'Obstruct Officer W/Violence', 'Assault Law Enforcement Officer', 'Violation of Injunction Order/Stalking/Cyberstalking', 'Manufacture Cannabis', 'Crim Use of Personal ID Info', 'Aggravated Battery On 65/Older', 'Carrying A Concealed Weapon', 'Attempt Burglary (Struct)', 'Del Cannabis For Consideration', 'Unauth Poss ID Card or DL', 'Solicit Deliver Cocaine', 'Poss Drugs W/O A Prescription', 'Sel Etc/Pos/w/Int Contrft Schd', 'Obtain Control Substance By Fraud', 'Burglary Structure Occupied', 'Sex Offender Fail Comply W/Law', 'Retail Theft $300 1st Offense', 'Uttering a Forged Instrument', 'Cruelty to Animals', 'Felony Batt(Great Bodily Harm)', 'Oper Motorcycle W/O Valid DL', 'Trespass Property w/Dang Weap', 'Possession Burglary Tools', 'Disorderly Intoxication', 'Stalking', 'Aggravated Assault W/dead Weap', 'Video Voyeur-<24Y on Child >16', 'Poss of Cocaine W/I/D/S 1000FT Park', 'False Bomb Report', 'Operating W/O Valid License', 'Neglect/Abuse Elderly Person', 'Felony Battery', 'Littering', 'Possession Of Phentermine', 'Shoot Into Vehicle', 'Sell/Man/Del Pos/w/int Heroin', 'Felony/Driving Under Influence', 'Retail Theft $300 2nd Offense', 'Prowling/Loitering', 'Possession of XLR11', 'Solicit To Deliver Cocaine', 'Possession Of Paraphernalia', 'Crim Attempt/Solicit/Consp', 'DUI Blood Alcohol Above 0.20', 'Driving While License Revoked', 'Possess Countrfeit Credit Card', 'Poss Anti-Shoplifting Device', 'Felony Petit Theft', 'Failure To Return Hired Vehicle', 'Use of Anti-Shoplifting Device', 'Trespass Private Property', 'Aggravated Assault', 'Poss of Methylethcathinone', 'Manage Busn W/O City Occup Lic', 'DWLS Susp/Cancel Revoked', 'Drivg While Lic Suspd/Revk/Can', 'Poss Of Controlled Substance', 'Armed Trafficking in Cannabis', 'Defrauding Innkeeper', 'Live on Earnings of Prostitute', 'Aggrav Child Abuse-Agg Battery', 'Solicitation On Felony 3 Deg', 'Discharge Firearm From Vehicle', 'Arson II (Vehicle)'], 'age_cat': ['Less than 25', '25 - 45', 'Greater than 45'], 'sex': ['Male', 'Female'], 'race': ['Black', 'White', 'Other'], 'is_recid': ['Yes', 'No']}
    


```python
temp_dict = train_df.describe().to_dict()
mean_std_dict = {}
for key, value in temp_dict.items():
  mean_std_dict[key] = [value['mean'],value['std']]
print(mean_std_dict)
```

    {'juv_fel_count': [0.0721830985915493, 0.5187066204966256], 'juv_misd_count': [0.09793133802816902, 0.5348148993571356], 'juv_other_count': [0.10629401408450705, 0.47329289176000755], 'priors_count': [3.504181338028169, 4.9829540064651585], 'age': [34.875220070422536, 11.929411671055068]}
    


```python
def preprocessing(tmp_df):
    features = {}
    # Normalize numberiacal columns
    for col_name in mean_std_dict.keys():
        _mean, _std = mean_std_dict[col_name]
        features[col_name] = ((tmp_df[col_name]-_mean)/_std)
    # Encode categorical columns as indices
    for col_name in vocab_dict.keys():
        features[col_name] = tmp_df[col_name].map(
            {
                j:i for i,j in enumerate(vocab_dict[col_name])
            }
        )
    # One-hot encoding categorical features
    for col_name in ["c_charge_degree", "c_charge_desc", "age_cat"]:
        features[col_name] = pd.get_dummies(features[col_name], prefix=col_name)
    return pd.concat(features.values(), axis=1)
```


```python
train_df = preprocessing(train_df)
dev_df =  preprocessing(dev_df)
test_df = preprocessing(test_df)
```


```python
train_df.to_pickle(os.path.join(dataset_base_dir, "train.pkl"))
dev_df.to_pickle(os.path.join(dataset_base_dir, "dev.pkl"))
test_df.to_pickle(os.path.join(dataset_base_dir, "test.pkl"))
```

## Train a Vanilla Model

```python
from fairlib import networks, BaseOptions, dataloaders
import torch
```


```python
Shared_options = {
    # The name of the dataset, correponding dataloader will be used,
    "dataset":  "COMPAS",

    # Specifiy the path to the input data
    "data_dir": "./data",

    # Device for computing, -1 is the cpu
    "device_id": -1,

    # The default path for saving experimental results
    "results_dir":  r"results",

    # The same as the dataset
    "project_dir":  r"dev",

    # We will focusing on TPR GAP, implying the Equalized Odds for binay classification.
    "GAP_metric_name":  "TPR_GAP",

    # The overall performance will be measured as accuracy
    "Performance_metric_name":  "accuracy",

    # Model selections are based on DTO
    "selection_criterion":  "DTO",

    # Default dirs for saving checkpoints
    "checkpoint_dir":   "models",
    "checkpoint_name":  "checkpoint_epoch",


    "n_jobs":   1,
}
```


```python
args = {
    "dataset":Shared_options["dataset"], 
    "data_dir":Shared_options["data_dir"],
    "device_id":Shared_options["device_id"],

    # Give a name to the exp, which will be used in the path
    "exp_id":"vanilla",

    "emb_size": 450-3,
    "lr": 0.001,
    "batch_size": 128,
    "hidden_size": 32,
    "n_hidden": 1,
    "activation_function": "ReLu",

    "num_classes": 2,
    "num_groups": 3, # Balck; White; and Other
}

# Init the argument
options = BaseOptions()
state = options.get_state(args=args, silence=True)
```

    INFO:root:Unexpected args: ['-f', '/root/.local/share/jupyter/runtime/kernel-2b94b31d-201f-46ea-842f-e5f6c625b168.json']
    INFO:root:Logging to ./results/dev/COMPAS/vanilla/output.log
    

    2022-04-08 15:10:45 [INFO ]  ======================================== 2022-04-08 15:10:45 ========================================
    2022-04-08 15:10:45 [INFO ]  Base directory is ./results/dev/COMPAS/vanilla
    Not implemented
    2022-04-08 15:10:45 [INFO ]  dataloaders need to be initialized!
    


```python
class CustomizedDataset(dataloaders.utils.BaseDataset):

    def load_data(self):

        self.data_dir = os.path.join(self.args.data_dir, "{}.pkl".format(self.split))

        data = pd.read_pickle(self.data_dir)

        self.X = data.drop(['sex', 'race', 'is_recid'], axis=1).to_numpy().astype(np.float32)
        self.y = list(data["is_recid"])
        self.protected_label = list(data["race"])
```


```python
customized_train_data = CustomizedDataset(args=state, split="train")
customized_dev_data = CustomizedDataset(args=state, split="dev")
customized_test_data = CustomizedDataset(args=state, split="test")

# DataLoader Parameters
tran_dataloader_params = {
        'batch_size': state.batch_size,
        'shuffle': True,
        'num_workers': state.num_workers}

eval_dataloader_params = {
        'batch_size': state.test_batch_size,
        'shuffle': False,
        'num_workers': state.num_workers}

# init dataloader
customized_training_generator = torch.utils.data.DataLoader(customized_train_data, **tran_dataloader_params)
customized_validation_generator = torch.utils.data.DataLoader(customized_dev_data, **eval_dataloader_params)
customized_test_generator = torch.utils.data.DataLoader(customized_test_data, **eval_dataloader_params)
```

    Loaded data shapes: (4544, 447), (4544,), (4544,)
    Loaded data shapes: (505, 447), (505,), (505,)
    Loaded data shapes: (2165, 447), (2165,), (2165,)
    


```python
model = networks.classifier.MLP(state)
```

    2022-04-08 15:10:53 [INFO ]  MLP( 
    2022-04-08 15:10:53 [INFO ]    (output_layer): Linear(in_features=32, out_features=2, bias=True)
    2022-04-08 15:10:53 [INFO ]    (AF): ReLU()
    2022-04-08 15:10:53 [INFO ]    (hidden_layers): ModuleList(
    2022-04-08 15:10:53 [INFO ]      (0): Linear(in_features=447, out_features=32, bias=True)
    2022-04-08 15:10:53 [INFO ]      (1): ReLU()
    2022-04-08 15:10:53 [INFO ]    )
    2022-04-08 15:10:53 [INFO ]    (criterion): CrossEntropyLoss()
    2022-04-08 15:10:53 [INFO ]  )
    2022-04-08 15:10:53 [INFO ]  Total number of parameters: 14402 
    
    


```python
model.train_self(
    train_generator = customized_training_generator,
    dev_generator = customized_validation_generator,
    test_generator = customized_test_generator,
)
```

    2022-04-08 15:10:56 [INFO ]  Epoch:    0 [      0/   4544 ( 0%)]	Loss: 0.6934	 Data Time: 0.05s	Train Time: 0.18s
    2022-04-08 15:10:56 [INFO ]  Evaluation at Epoch 0
    2022-04-08 15:10:56 [INFO ]  Validation accuracy: 65.35	macro_fscore: 64.74	micro_fscore: 65.35	TPR_GAP: 29.34	FPR_GAP: 29.34	PPR_GAP: 31.29	
    2022-04-08 15:10:56 [INFO ]  Test accuracy: 67.16	macro_fscore: 66.42	micro_fscore: 67.16	TPR_GAP: 30.35	FPR_GAP: 30.35	PPR_GAP: 35.61	
    2022-04-08 15:10:56 [INFO ]  Epoch:    1 [      0/   4544 ( 0%)]	Loss: 0.6347	 Data Time: 0.00s	Train Time: 0.00s
    2022-04-08 15:10:56 [INFO ]  Evaluation at Epoch 1
    2022-04-08 15:10:56 [INFO ]  Validation accuracy: 66.53	macro_fscore: 66.34	micro_fscore: 66.53	TPR_GAP: 28.61	FPR_GAP: 28.61	PPR_GAP: 31.18	
    2022-04-08 15:10:56 [INFO ]  Test accuracy: 67.39	macro_fscore: 67.07	micro_fscore: 67.39	TPR_GAP: 31.21	FPR_GAP: 31.21	PPR_GAP: 36.49	
    2022-04-08 15:10:56 [INFO ]  Epoch:    2 [      0/   4544 ( 0%)]	Loss: 0.5875	 Data Time: 0.00s	Train Time: 0.00s
    2022-04-08 15:10:57 [INFO ]  Evaluation at Epoch 2
    2022-04-08 15:10:57 [INFO ]  Validation accuracy: 66.53	macro_fscore: 66.42	micro_fscore: 66.53	TPR_GAP: 30.15	FPR_GAP: 30.15	PPR_GAP: 32.84	
    2022-04-08 15:10:57 [INFO ]  Test accuracy: 68.73	macro_fscore: 68.44	micro_fscore: 68.73	TPR_GAP: 32.73	FPR_GAP: 32.73	PPR_GAP: 38.85	
    2022-04-08 15:10:57 [INFO ]  Epoch:    3 [      0/   4544 ( 0%)]	Loss: 0.6307	 Data Time: 0.00s	Train Time: 0.00s
    2022-04-08 15:10:57 [INFO ]  Epochs since last improvement: 1
    2022-04-08 15:10:57 [INFO ]  Evaluation at Epoch 3
    2022-04-08 15:10:57 [INFO ]  Validation accuracy: 64.95	macro_fscore: 64.75	micro_fscore: 64.95	TPR_GAP: 30.37	FPR_GAP: 30.37	PPR_GAP: 32.78	
    2022-04-08 15:10:57 [INFO ]  Test accuracy: 68.96	macro_fscore: 68.58	micro_fscore: 68.96	TPR_GAP: 32.78	FPR_GAP: 32.78	PPR_GAP: 38.32	
    2022-04-08 15:10:57 [INFO ]  Epoch:    4 [      0/   4544 ( 0%)]	Loss: 0.5862	 Data Time: 0.00s	Train Time: 0.00s
    2022-04-08 15:10:57 [INFO ]  Epochs since last improvement: 2
    2022-04-08 15:10:57 [INFO ]  Evaluation at Epoch 4
    2022-04-08 15:10:57 [INFO ]  Validation accuracy: 65.54	macro_fscore: 65.52	micro_fscore: 65.54	TPR_GAP: 35.21	FPR_GAP: 35.21	PPR_GAP: 38.30	
    2022-04-08 15:10:57 [INFO ]  Test accuracy: 68.87	macro_fscore: 68.76	micro_fscore: 68.87	TPR_GAP: 33.56	FPR_GAP: 33.56	PPR_GAP: 40.54	
    2022-04-08 15:10:57 [INFO ]  Epoch:    5 [      0/   4544 ( 0%)]	Loss: 0.5576	 Data Time: 0.00s	Train Time: 0.00s
    2022-04-08 15:10:57 [INFO ]  Epochs since last improvement: 3
    2022-04-08 15:10:57 [INFO ]  Evaluation at Epoch 5
    2022-04-08 15:10:57 [INFO ]  Validation accuracy: 65.94	macro_fscore: 65.67	micro_fscore: 65.94	TPR_GAP: 33.95	FPR_GAP: 33.95	PPR_GAP: 37.23	
    2022-04-08 15:10:57 [INFO ]  Test accuracy: 68.91	macro_fscore: 68.41	micro_fscore: 68.91	TPR_GAP: 29.95	FPR_GAP: 29.95	PPR_GAP: 35.99	
    2022-04-08 15:10:57 [INFO ]  Epoch:    6 [      0/   4544 ( 0%)]	Loss: 0.5662	 Data Time: 0.00s	Train Time: 0.00s
    2022-04-08 15:10:57 [INFO ]  Epochs since last improvement: 4
    2022-04-08 15:10:57 [INFO ]  Evaluation at Epoch 6
    2022-04-08 15:10:57 [INFO ]  Validation accuracy: 64.75	macro_fscore: 64.62	micro_fscore: 64.75	TPR_GAP: 35.02	FPR_GAP: 35.02	PPR_GAP: 37.66	
    2022-04-08 15:10:57 [INFO ]  Test accuracy: 68.91	macro_fscore: 68.66	micro_fscore: 68.91	TPR_GAP: 30.55	FPR_GAP: 30.55	PPR_GAP: 37.57	
    2022-04-08 15:10:57 [INFO ]  Epoch:    7 [      0/   4544 ( 0%)]	Loss: 0.6291	 Data Time: 0.00s	Train Time: 0.00s
    2022-04-08 15:10:57 [INFO ]  Epochs since last improvement: 5
    2022-04-08 15:10:57 [INFO ]  Evaluation at Epoch 7
    2022-04-08 15:10:57 [INFO ]  Validation accuracy: 65.15	macro_fscore: 65.13	micro_fscore: 65.15	TPR_GAP: 33.15	FPR_GAP: 33.15	PPR_GAP: 36.60	
    2022-04-08 15:10:57 [INFO ]  Test accuracy: 68.73	macro_fscore: 68.63	micro_fscore: 68.73	TPR_GAP: 30.15	FPR_GAP: 30.15	PPR_GAP: 37.46	
    

## Bias Mitigation

```python
debiasing_args = {
    "dataset":Shared_options["dataset"], 
    "data_dir":Shared_options["data_dir"],
    "device_id":Shared_options["device_id"],

    # Give a name to the exp, which will be used in the path
    "exp_id":"BT_Adv",

    "emb_size": 450-3,
    "lr": 0.001,
    "batch_size": 128,
    "hidden_size": 32,
    "n_hidden": 1,
    "activation_function": "ReLu",

    "num_classes": 2,
    "num_groups": 3, # Balck; White; and Other

    # Perform adversarial training if True
    "adv_debiasing":True,

    # Specify the hyperparameters for Balanced Training
    "BT":"Resampling",
    "BTObj":"EO",
}

# Init the argument
debias_options = BaseOptions()
debias_state = debias_options.get_state(args=debiasing_args, silence=True)

customized_train_data = CustomizedDataset(args=debias_state, split="train")
customized_dev_data = CustomizedDataset(args=debias_state, split="dev")
customized_test_data = CustomizedDataset(args=debias_state, split="test")

# DataLoader Parameters
tran_dataloader_params = {
        'batch_size': state.batch_size,
        'shuffle': True,
        'num_workers': state.num_workers}

eval_dataloader_params = {
        'batch_size': state.test_batch_size,
        'shuffle': False,
        'num_workers': state.num_workers}

# init dataloader
customized_training_generator = torch.utils.data.DataLoader(customized_train_data, **tran_dataloader_params)
customized_validation_generator = torch.utils.data.DataLoader(customized_dev_data, **eval_dataloader_params)
customized_test_generator = torch.utils.data.DataLoader(customized_test_data, **eval_dataloader_params)

debias_model = networks.classifier.MLP(debias_state)
```

    2022-04-08 15:11:04 [INFO ]  Unexpected args: ['-f', '/root/.local/share/jupyter/runtime/kernel-2b94b31d-201f-46ea-842f-e5f6c625b168.json']
    2022-04-08 15:11:04 [INFO ]  Logging to ./results/dev/COMPAS/BT_Adv/output.log
    2022-04-08 15:11:04 [INFO ]  ======================================== 2022-04-08 15:11:04 ========================================
    2022-04-08 15:11:04 [INFO ]  Base directory is ./results/dev/COMPAS/BT_Adv
    Not implemented
    2022-04-08 15:11:04 [INFO ]  dataloaders need to be initialized!
    2022-04-08 15:11:04 [INFO ]  SubDiscriminator( 
    2022-04-08 15:11:04 [INFO ]    (grad_rev): GradientReversal()
    2022-04-08 15:11:04 [INFO ]    (output_layer): Linear(in_features=300, out_features=3, bias=True)
    2022-04-08 15:11:04 [INFO ]    (AF): ReLU()
    2022-04-08 15:11:04 [INFO ]    (hidden_layers): ModuleList(
    2022-04-08 15:11:04 [INFO ]      (0): Linear(in_features=32, out_features=300, bias=True)
    2022-04-08 15:11:04 [INFO ]      (1): ReLU()
    2022-04-08 15:11:04 [INFO ]      (2): Linear(in_features=300, out_features=300, bias=True)
    2022-04-08 15:11:04 [INFO ]      (3): ReLU()
    2022-04-08 15:11:04 [INFO ]    )
    2022-04-08 15:11:04 [INFO ]    (criterion): CrossEntropyLoss()
    2022-04-08 15:11:04 [INFO ]  )
    2022-04-08 15:11:04 [INFO ]  Total number of parameters: 101103 
    
    2022-04-08 15:11:04 [INFO ]  Discriminator built!
    Loaded data shapes: (2001, 447), (2001,), (2001,)
    Loaded data shapes: (252, 447), (252,), (252,)
    Loaded data shapes: (939, 447), (939,), (939,)
    2022-04-08 15:11:04 [INFO ]  MLP( 
    2022-04-08 15:11:04 [INFO ]    (output_layer): Linear(in_features=32, out_features=2, bias=True)
    2022-04-08 15:11:04 [INFO ]    (AF): ReLU()
    2022-04-08 15:11:04 [INFO ]    (hidden_layers): ModuleList(
    2022-04-08 15:11:04 [INFO ]      (0): Linear(in_features=447, out_features=32, bias=True)
    2022-04-08 15:11:04 [INFO ]      (1): ReLU()
    2022-04-08 15:11:04 [INFO ]    )
    2022-04-08 15:11:04 [INFO ]    (criterion): CrossEntropyLoss()
    2022-04-08 15:11:04 [INFO ]  )
    2022-04-08 15:11:04 [INFO ]  Total number of parameters: 14402 
    
    


```python
debias_model.train_self(
    train_generator = customized_training_generator,
    dev_generator = customized_validation_generator,
    test_generator = customized_test_generator,
)
```

    2022-04-08 15:11:07 [INFO ]  Epoch:    0 [      0/   2001 ( 0%)]	Loss: -0.3949	 Data Time: 0.01s	Train Time: 0.04s
    2022-04-08 15:11:08 [INFO ]  Evaluation at Epoch 0
    2022-04-08 15:11:08 [INFO ]  Validation accuracy: 55.16	macro_fscore: 43.87	micro_fscore: 55.16	TPR_GAP: 9.77	FPR_GAP: 9.77	PPR_GAP: 8.73	
    2022-04-08 15:11:08 [INFO ]  Test accuracy: 66.24	macro_fscore: 47.19	micro_fscore: 66.24	TPR_GAP: 7.53	FPR_GAP: 7.53	PPR_GAP: 5.11	
    2022-04-08 15:11:08 [INFO ]  Epoch:    1 [      0/   2001 ( 0%)]	Loss: -0.4219	 Data Time: 0.00s	Train Time: 0.01s
    2022-04-08 15:11:08 [INFO ]  Evaluation at Epoch 1
    2022-04-08 15:11:08 [INFO ]  Validation accuracy: 55.95	macro_fscore: 45.35	micro_fscore: 55.95	TPR_GAP: 7.64	FPR_GAP: 7.64	PPR_GAP: 7.14	
    2022-04-08 15:11:08 [INFO ]  Test accuracy: 66.67	macro_fscore: 48.27	micro_fscore: 66.67	TPR_GAP: 10.50	FPR_GAP: 10.50	PPR_GAP: 6.18	
    2022-04-08 15:11:08 [INFO ]  Epoch:    2 [      0/   2001 ( 0%)]	Loss: -0.4536	 Data Time: 0.00s	Train Time: 0.01s
    2022-04-08 15:11:08 [INFO ]  Evaluation at Epoch 2
    2022-04-08 15:11:08 [INFO ]  Validation accuracy: 58.73	macro_fscore: 50.64	micro_fscore: 58.73	TPR_GAP: 19.20	FPR_GAP: 19.20	PPR_GAP: 16.67	
    2022-04-08 15:11:08 [INFO ]  Test accuracy: 67.52	macro_fscore: 51.29	micro_fscore: 67.52	TPR_GAP: 13.91	FPR_GAP: 13.91	PPR_GAP: 8.09	
    2022-04-08 15:11:08 [INFO ]  Epoch:    3 [      0/   2001 ( 0%)]	Loss: -0.4428	 Data Time: 0.00s	Train Time: 0.02s
    2022-04-08 15:11:09 [INFO ]  Evaluation at Epoch 3
    2022-04-08 15:11:09 [INFO ]  Validation accuracy: 59.13	macro_fscore: 52.36	micro_fscore: 59.13	TPR_GAP: 22.48	FPR_GAP: 22.48	PPR_GAP: 20.63	
    2022-04-08 15:11:09 [INFO ]  Test accuracy: 68.37	macro_fscore: 54.69	micro_fscore: 68.37	TPR_GAP: 15.95	FPR_GAP: 15.95	PPR_GAP: 10.86	
    2022-04-08 15:11:09 [INFO ]  Epoch:    4 [      0/   2001 ( 0%)]	Loss: -0.4942	 Data Time: 0.00s	Train Time: 0.01s
    2022-04-08 15:11:09 [INFO ]  Evaluation at Epoch 4
    2022-04-08 15:11:09 [INFO ]  Validation accuracy: 61.11	macro_fscore: 56.25	micro_fscore: 61.11	TPR_GAP: 24.95	FPR_GAP: 24.95	PPR_GAP: 23.81	
    2022-04-08 15:11:09 [INFO ]  Test accuracy: 69.22	macro_fscore: 57.71	micro_fscore: 69.22	TPR_GAP: 22.75	FPR_GAP: 22.75	PPR_GAP: 14.27	
    2022-04-08 15:11:09 [INFO ]  Epoch:    5 [      0/   2001 ( 0%)]	Loss: -0.4361	 Data Time: 0.00s	Train Time: 0.01s
    2022-04-08 15:11:09 [INFO ]  Evaluation at Epoch 5
    2022-04-08 15:11:09 [INFO ]  Validation accuracy: 61.51	macro_fscore: 57.77	micro_fscore: 61.51	TPR_GAP: 31.73	FPR_GAP: 31.73	PPR_GAP: 30.95	
    2022-04-08 15:11:09 [INFO ]  Test accuracy: 69.54	macro_fscore: 59.98	micro_fscore: 69.54	TPR_GAP: 23.34	FPR_GAP: 23.34	PPR_GAP: 15.34	
    2022-04-08 15:11:09 [INFO ]  Epoch:    6 [      0/   2001 ( 0%)]	Loss: -0.4565	 Data Time: 0.00s	Train Time: 0.01s
    2022-04-08 15:11:10 [INFO ]  Epochs since last improvement: 1
    2022-04-08 15:11:10 [INFO ]  Evaluation at Epoch 6
    2022-04-08 15:11:10 [INFO ]  Validation accuracy: 62.70	macro_fscore: 59.38	micro_fscore: 62.70	TPR_GAP: 31.73	FPR_GAP: 31.73	PPR_GAP: 30.95	
    2022-04-08 15:11:10 [INFO ]  Test accuracy: 69.54	macro_fscore: 60.41	micro_fscore: 69.54	TPR_GAP: 23.16	FPR_GAP: 23.16	PPR_GAP: 16.19	
    2022-04-08 15:11:10 [INFO ]  Epoch:    7 [      0/   2001 ( 0%)]	Loss: -0.5079	 Data Time: 0.01s	Train Time: 0.03s
    2022-04-08 15:11:10 [INFO ]  Evaluation at Epoch 7
    2022-04-08 15:11:10 [INFO ]  Validation accuracy: 64.68	macro_fscore: 62.17	micro_fscore: 64.68	TPR_GAP: 33.00	FPR_GAP: 33.00	PPR_GAP: 32.54	
    2022-04-08 15:11:10 [INFO ]  Test accuracy: 70.18	macro_fscore: 62.32	micro_fscore: 70.18	TPR_GAP: 26.50	FPR_GAP: 26.50	PPR_GAP: 19.17	
    2022-04-08 15:11:10 [INFO ]  Epoch:    8 [      0/   2001 ( 0%)]	Loss: -0.5115	 Data Time: 0.00s	Train Time: 0.03s
    2022-04-08 15:11:11 [INFO ]  Evaluation at Epoch 8
    2022-04-08 15:11:11 [INFO ]  Validation accuracy: 65.08	macro_fscore: 62.83	micro_fscore: 65.08	TPR_GAP: 33.30	FPR_GAP: 33.30	PPR_GAP: 32.54	
    2022-04-08 15:11:11 [INFO ]  Test accuracy: 70.07	macro_fscore: 62.94	micro_fscore: 70.07	TPR_GAP: 27.85	FPR_GAP: 27.85	PPR_GAP: 18.74	
    2022-04-08 15:11:11 [INFO ]  Epoch:    9 [      0/   2001 ( 0%)]	Loss: -0.5244	 Data Time: 0.01s	Train Time: 0.02s
    2022-04-08 15:11:11 [INFO ]  Epochs since last improvement: 1
    2022-04-08 15:11:11 [INFO ]  Evaluation at Epoch 9
    2022-04-08 15:11:11 [INFO ]  Validation accuracy: 65.48	macro_fscore: 63.17	micro_fscore: 65.48	TPR_GAP: 33.92	FPR_GAP: 33.92	PPR_GAP: 33.33	
    2022-04-08 15:11:11 [INFO ]  Test accuracy: 70.29	macro_fscore: 63.03	micro_fscore: 70.29	TPR_GAP: 25.90	FPR_GAP: 25.90	PPR_GAP: 18.32	
    2022-04-08 15:11:11 [INFO ]  Epoch:   10 [      0/   2001 ( 0%)]	Loss: -0.5170	 Data Time: 0.00s	Train Time: 0.03s
    2022-04-08 15:11:11 [INFO ]  Epochs since last improvement: 2
    2022-04-08 15:11:11 [INFO ]  Evaluation at Epoch 10
    2022-04-08 15:11:11 [INFO ]  Validation accuracy: 64.68	macro_fscore: 62.33	micro_fscore: 64.68	TPR_GAP: 32.35	FPR_GAP: 32.35	PPR_GAP: 30.95	
    2022-04-08 15:11:11 [INFO ]  Test accuracy: 70.18	macro_fscore: 62.86	micro_fscore: 70.18	TPR_GAP: 26.16	FPR_GAP: 26.16	PPR_GAP: 17.89	
    2022-04-08 15:11:11 [INFO ]  Epoch:   11 [      0/   2001 ( 0%)]	Loss: -0.4809	 Data Time: 0.00s	Train Time: 0.01s
    2022-04-08 15:11:11 [INFO ]  Epochs since last improvement: 3
    2022-04-08 15:11:11 [INFO ]  Evaluation at Epoch 11
    2022-04-08 15:11:11 [INFO ]  Validation accuracy: 65.48	macro_fscore: 63.33	micro_fscore: 65.48	TPR_GAP: 35.23	FPR_GAP: 35.23	PPR_GAP: 34.13	
    2022-04-08 15:11:11 [INFO ]  Test accuracy: 69.86	macro_fscore: 62.85	micro_fscore: 69.86	TPR_GAP: 26.16	FPR_GAP: 26.16	PPR_GAP: 17.89	
    2022-04-08 15:11:11 [INFO ]  Epoch:   12 [      0/   2001 ( 0%)]	Loss: -0.5162	 Data Time: 0.00s	Train Time: 0.01s
    2022-04-08 15:11:12 [INFO ]  Epochs since last improvement: 4
    2022-04-08 15:11:12 [INFO ]  Evaluation at Epoch 12
    2022-04-08 15:11:12 [INFO ]  Validation accuracy: 64.29	macro_fscore: 61.98	micro_fscore: 64.29	TPR_GAP: 33.67	FPR_GAP: 33.67	PPR_GAP: 32.54	
    2022-04-08 15:11:12 [INFO ]  Test accuracy: 69.65	macro_fscore: 62.75	micro_fscore: 69.65	TPR_GAP: 26.87	FPR_GAP: 26.87	PPR_GAP: 17.68	
    2022-04-08 15:11:12 [INFO ]  Epoch:   13 [      0/   2001 ( 0%)]	Loss: -0.4658	 Data Time: 0.00s	Train Time: 0.01s
    2022-04-08 15:11:12 [INFO ]  Epochs since last improvement: 5
    2022-04-08 15:11:12 [INFO ]  Evaluation at Epoch 13
    2022-04-08 15:11:12 [INFO ]  Validation accuracy: 63.89	macro_fscore: 61.64	micro_fscore: 63.89	TPR_GAP: 34.87	FPR_GAP: 34.87	PPR_GAP: 34.13	
    2022-04-08 15:11:12 [INFO ]  Test accuracy: 69.01	macro_fscore: 62.31	micro_fscore: 69.01	TPR_GAP: 26.53	FPR_GAP: 26.53	PPR_GAP: 17.89	
    