import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#Load the datasets
pop=pd.read_csv("counties_data.csv")
votes=pd.read_csv("votes.csv")

# Data Preparation

# Calculate the percentage of votes for each party
votes['Percent party'] = (votes['candidatevotes'] / votes['totalvotes']) * 100
# Concatenate county name and state for accurate merging later
votes['county_name'] = votes['county_name'] + ', ' + votes['state']
# Exclude rows where the political party is OTHER
votes = votes[votes['party'] != 'OTHER']
# Drop unnecessary columns after calculating Percent party
votes = votes.drop(columns=['candidatevotes', 'totalvotes'])
# Reshape the dataset so each party becomes a column with the percentage as values
votes = votes.pivot_table(index=['state', 'county_name', ], columns='party',
    values='Percent party', aggfunc='first').reset_index()
# Determine the leading political party for each county
votes['Leading Political Party'] = votes.apply(
    lambda row: 'DEMOCRAT' if row['DEMOCRAT'] > row['REPUBLICAN'] else 'REPUBLICAN', axis=1)
# Rename columns
votes.rename(columns={'DEMOCRAT': 'Democrat', 'REPUBLICAN': 'Republican', 'state':'State',
                      'county_name':'County'}, inplace=True)
# Extract only the county name and capitalize
votes['County']=votes['County'].str.split(',').str[0]
votes = votes.apply(lambda col: col.str.title() if col.dtype == 'object' else col)

# Clean the County column in the population dataset
pop['County'] = (pop['County']
    .str.split(',').str[0]  
    .str.replace('County', '', case=False).str.strip())
# Transform the crime rate to crime per 100000 residents
pop['Crime'] = pop['Crime'] / 100

# Merge the votes data into the population dataset
pop = pd.merge(pop, votes, on=["State", "County"], how="left")


#Data exploration

pop.info()

# Calculate summary statistics 

summary_df = pop.select_dtypes(include='float64').describe()
summary_df = summary_df.round(1).transpose()
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                 rowLabels=summary_df.index, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(summary_df.columns))))
plt.savefig('statistics.jpg', bbox_inches='tight')
plt.close()


# Check for missing values in the entire dataset
missing_values = pop.isnull().sum()
print(missing_values)
# Remove rows where 'Leading Political Party' is NaN
pop = pop.dropna(subset=['Leading Political Party'])
# Replace missing values in numeric columns with their mean 
numeric_cols = pop.select_dtypes(include=['number']).columns
pop[numeric_cols] = pop[numeric_cols].fillna(pop[numeric_cols].mean())
# Check for missing values in the entire dataset
missing_values = pop.isnull().sum()
print(missing_values)

# Map 'Democrat' to 1 and 'Republican' to 0 for numerical representation
pop['Political Binary'] = pop['Leading Political Party'].map({"Democrat": 1, "Republican": 0})

# Pairplots to visualize relationships between different variables
sns.pairplot(pop, vars=["Population", "White", "African American", "Hispanic", 
                         "Foreign born", "Leading Political Party"])
plt.show()

sns.pairplot(pop, vars=["Less Than HighSchool", "At Least High School", "Median age",
                        "Bachelors Degree and Higher", "Leading Political Party"])
plt.show()

sns.pairplot(pop, vars=["Poverty",  "Homeownership", "Household income",
                        "Crime", "Unemployment", "Leading Political Party"])
plt.show()

# Box plots 

fig, axs = plt.subplots(2, 2, figsize=(14, 12))
# Box plot for crime rate
data_to_plot = [pop[pop['Leading Political Party'] == 'Democrat']['Crime'],
                pop[pop['Leading Political Party'] == 'Republican']['Crime']]
axs[0, 0].boxplot(data_to_plot, labels=['Democrat', 'Republican'], patch_artist=True, 
                  boxprops=dict(facecolor='lightblue', color='blue'), 
                  medianprops=dict(color='purple', linewidth=3), widths=0.8)
axs[0, 0].set_ylabel('Crime Rate (Per 100,000 residents)')
axs[0, 0].set_title('Democrat counties have higher median crime rate \n compared to Republican counties')

# Box plot for income
data_to_plot = [pop[pop['Leading Political Party'] == 'Democrat']['Household income'],
                pop[pop['Leading Political Party'] == 'Republican']['Household income']]
axs[0, 1].boxplot(data_to_plot, labels=['Democrat', 'Republican'], patch_artist=True, 
                  boxprops=dict(facecolor='lightblue', color='blue'), 
                  medianprops=dict(color='purple', linewidth=3), widths=0.8)
axs[0, 1].set_ylabel('Median Household Income ($)')
axs[0, 1].set_title('Median husehold income is similar between parties, \n with greater variability in Democrat counties')
axs[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${int(x / 1000)}K"))

# Box plot for median age
data_to_plot = [pop[pop['Leading Political Party'] == 'Democrat']['Median age'],
    pop[pop['Leading Political Party'] == 'Republican']['Median age']]
axs[1, 0].boxplot(data_to_plot, labels=['Democrat', 'Republican'], patch_artist=True, 
                  boxprops=dict(facecolor='lightblue', color='blue'), 
                  medianprops=dict(color='purple', linewidth=3), widths=0.8)
axs[1, 0].set_ylabel('Median Age')
axs[1, 0].set_title('Democrat counties have higher median age \n compared to Republican counties')

# Box plot for unemployment
data_to_plot = [pop[pop['Leading Political Party'] == 'Democrat']['Unemployment'],
    pop[pop['Leading Political Party'] == 'Republican']['Unemployment']]
axs[1, 1].boxplot(data_to_plot, labels=['Democrat', 'Republican'], patch_artist=True, 
                  boxprops=dict(facecolor='lightblue', color='blue'), 
                  medianprops=dict(color='purple', linewidth=3), widths=0.8)
axs[1, 1].set_ylabel('Unemployment Percentage')
axs[1, 1].set_title('Democrat counties show higher median unemployment \n compared to Republican counties')
axs[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

plt.tight_layout()
plt.show()


# Bar plots
education = pop.groupby('Leading Political Party')[['Less Than HighSchool', 'At Least High School', 
                                                    'Bachelors Degree and Higher']].mean().T
education.plot(kind='bar', figsize=(12, 8), color=['cornflowerblue', 'lightcoral'])
plt.title('Political affiliation differs across education levels')
plt.ylabel('Average Percentage')
plt.legend(title='Leading Political Party', loc='upper right', frameon=True)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


race = pop.groupby('Leading Political Party')[['White', 'African American','Asian',
                                               'Hispanic','Foreign born']].mean().T
race.plot(kind='bar', figsize=(12, 8), color=['cornflowerblue', 'lightcoral'])
plt.title('Political affiliation varies across different racial groups')
plt.ylabel('Average Percentage')
plt.legend(title='Leading Political Party', loc='upper right', frameon=True)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#Heat map for correlation

columns = ['Political Binary'] + [col for col in numeric_cols if col not in ['Democrat', 'Republican']]
correlation_matrix = pop[columns].corr()
correlation_with_political = correlation_matrix[['Political Binary']].drop('Political Binary')
plt.figure(figsize=(4, 9))
sns.heatmap(correlation_with_political, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, 
            cbar_kws={'shrink': 0.9}, vmin=-1,  vmax=1)
plt.title('Correlation with Political Affiliation')
plt.show()


# Hypothesis testing

formula = ("Crime ~ np.log(Population)")
# Fit the linear regression model
model = smf.ols(formula=formula, data=pop).fit()
print(model.summary())

# Hypothesis testing

population_threshold = 800000
large_population_data = pop[pop['Population'] >= population_threshold]
group_democrat = large_population_data[large_population_data['Political Binary'] == 1]['Crime']
group_republican = large_population_data[large_population_data['Political Binary'] == 0]['Crime']

t_stat, p_value_two_tailed = ttest_ind(group_democrat, group_republican, equal_var=False)
p_value_one_tailed = p_value_two_tailed / 2 if t_stat > 0 else 1

print(f"One-Tailed P-Value: {p_value_one_tailed:.4f}")

# Check multicolinearity

variables = pop[['Population','White','African American','Asian','Hispanic',
'Foreign born','Less Than HighSchool','At Least High School',
'Bachelors Degree and Higher','Median age','Household income',
'Homeownership','Poverty','Crime','Unemployment']]

vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) 
              for i in range(variables.shape[1])]
vif["Features"] = variables.columns
vif

# New column Diversity
pop['Diversity'] = (pop['African American'] + pop['Asian'] + pop['Hispanic'] + pop['Foreign born'])


variables = pop[['Population','White','Diversity', 
'Bachelors Degree and Higher','Unemployment']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) 
              for i in range(variables.shape[1])]
vif["Features"] = variables.columns
vif


# Linear Regression
formula = ("Democrat ~ np.log(Population)*Q('Median age')*Homeownership + White * Diversity +"
    "Q('Bachelors Degree and Higher') * Q('Less Than HighSchool') + np.sqrt(Unemployment)")
# Split the dataset into training and testing sets
train_data, test_data = train_test_split(pop, test_size=0.2, random_state=365)
# Fit the model on the training data
train_model = smf.ols(formula=formula, data=train_data).fit()
print(train_model.summary())

# Extract the actual target values for the training set
y_train = train_data['Democrat']
# Predict using the training data
y_hat = train_model.predict(train_data)  
# Plot the actual vs predicted values
plt.figure(figsize=(16, 12))
plt.scatter(y_train, y_hat, alpha=0.7, s=150)
plt.plot([y_train.min(), y_train.max()],
         [y_train.min(), y_train.max()],
         color='red', linestyle='--')
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Predictions (y_hat)', size=18)
plt.show()


# Extract predictors (x_test) and target (y_test) from the test dataset
y_test = test_data['Democrat']  
x_test = test_data  
# Predict on the test dataset
y_hat_test = train_model.predict(x_test)  
# Plot actual vs predicted values
plt.figure(figsize=(16, 12))
plt.scatter(y_test, y_hat_test, alpha=0.7, s=150)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linestyle='--')
plt.xlabel('Targets (y_test)', size=18)
plt.ylabel('Predictions (y_hat_test)', size=18)
plt.savefig("ytest_plot.png", dpi=300, bbox_inches='tight')
plt.show()


#Logistic Regression
columns_to_use = ['Population', 'White', 'Diversity', 'Less Than HighSchool','Median age',
                  'Bachelors Degree and Higher', 'Homeownership', 'Unemployment']
# Subset the data
X = pop[columns_to_use]
y = pop['Political Binary']  

# Apply log transformation
log_transform = 'Population'
X[log_transform] = np.log(X[log_transform])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logit_model_reg = LogisticRegression(max_iter=1000)
logit_model_reg.fit(X_train, y_train)

X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Fit the logistic regression model
logit_model = sm.Logit(y_train, X_train_sm).fit()
print(logit_model.summary())

cv_scores = cross_val_score(logit_model_reg, X_train, y_train, cv=15, scoring='accuracy')

print("Cross-Validation Accuracy Scores:", cv_scores)
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation of CV Accuracy: {cv_scores.std():.4f}")






