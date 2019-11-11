import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing



sns.set_style('whitegrid')

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
train_df_org = pd.read_csv('data/train.csv')
test_df_org = pd.read_csv('data/test.csv')
test_df_org['Survived'] = 0
combined_train_test = train_df_org.append(test_df_org)
PassengerId = test_df_org['PassengerId']


################analyze data#################
#sex and survival
def analyze_sex():
	train_data.groupby(['Sex','Survived'])['Survived'].count()
	train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()

#pclass and survival
def analyze_pclass():
	train_data.groupby(['Pclass','Survived'])['Pclass'].count()
	train_data[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()


#age and survival
def analyze_age():
	fig, ax = plt.subplots(1, 2, figsize = (18, 8))
	sns.violinplot("Pclass", "Age", hue="Survived", data=train_data, split=True, ax=ax[0])
	ax[0].set_title('Pclass and Age vs Survived')
	ax[0].set_yticks(range(0, 110, 10))

	sns.violinplot("Sex", "Age", hue="Survived", data=train_data, split=True, ax=ax[1])
	ax[1].set_title('Sex and Age vs Survived')
	ax[1].set_yticks(range(0, 110, 10))

	plt.show()

	plt.figure(figsize=(12,5))
	plt.subplot(121)
	train_data['Age'].hist(bins=70)
	plt.xlabel('Age')
	plt.ylabel('Num')

	plt.subplot(122)
	train_data.boxplot(column='Age', showfliers=False)
	plt.show()


#title and survival
def analyze_title():
	train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
	pd.crosstab(train_data['Title'], train_data['Sex'])

	train_data[['Title','Survived']].groupby(['Title']).mean().plot.bar()

#sibling and survival
def analyze_sibling():
	sibsp_df = train_data[train_data['SibSp'] != 0]
	no_sibsp_df = train_data[train_data['SibSp'] == 0]
	plt.figure(figsize=(10,5))
	plt.subplot(121)
	sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
	plt.xlabel('sibsp')

	plt.subplot(122)
	no_sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
	plt.xlabel('no_sibsp')

	plt.show()

#parch and survival
def analyze_parch():
	parch_df = train_data[train_data['Parch'] != 0]
	no_parch_df = train_data[train_data['Parch'] == 0]

	plt.figure(figsize=(10,5))
	plt.subplot(121)
	parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
	plt.xlabel('parch')

	plt.subplot(122)
	no_parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
	plt.xlabel('no_parch')

	plt.show()

	train_data['Family_Size'] = train_data['Parch'] + train_data['SibSp'] + 1
	train_data[['Family_Size','Survived']].groupby(['Family_Size']).mean().plot.bar()


#fare and survival
def analyze_fare():
	plt.figure(figsize=(10,5))
	train_data['Fare'].hist(bins = 70)

	train_data.boxplot(column='Fare', by='Pclass', showfliers=False)
	plt.show()

	train_data['Fare'].describe()


#Cabin and survival
def analyze_cabin():
	# Replace missing values with "U0"
	train_data.loc[train_data.Cabin.isnull(), 'Cabin'] = 'U0'
	train_data['Has_Cabin'] = train_data['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)
	train_data[['Has_Cabin','Survived']].groupby(['Has_Cabin']).mean().plot.bar()
	# create feature for the alphabetical part of the cabin number
	train_data['CabinLetter'] = train_data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
	# convert the distinct cabin letters with incremental integer values
	train_data['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]
	train_data[['CabinLetter','Survived']].groupby(['CabinLetter']).mean().plot.bar()

	#embarked and survival
	sns.countplot('Embarked', hue='Survived', data=train_data)
	plt.title('Embarked and Survived')
	sns.factorplot('Embarked', 'Survived', data=train_data, size=3, aspect=2)
	plt.title('Embarked and Survived rate')
	plt.show()


def analyze_data():
	analyze_sex()
	analyze_pclass()
	analyze_cabin()
	analyze_fare()
	analyze_sibling()
	analyze_parch()
	analyze_title()
	analyze_age()




###########feature engineering###############3

def embark_variable():
	global combined_train_test
	#embarked
	combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().iloc[0], inplace=True)
	# 为了后面的特征分析，这里我们将 Embarked 特征进行facrorizing
	combined_train_test['Embarked'] = pd.factorize(combined_train_test['Embarked'])[0]

	# 使用 pd.get_dummies 获取one-hot 编码
	emb_dummies_df = pd.get_dummies(combined_train_test['Embarked'], prefix=combined_train_test[['Embarked']].columns[0])
	combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)

def sex_variable():
	global combined_train_test
	#sex
	combined_train_test['Sex'] = pd.factorize(combined_train_test['Sex'])[0]
	sex_dummies_df = pd.get_dummies(combined_train_test['Sex'], prefix=combined_train_test[['Sex']].columns[0])
	combined_train_test = pd.concat([combined_train_test, sex_dummies_df], axis=1)

def title_variable():
	global combined_train_test
	#name
	# what is each person's title? 
	combined_train_test['Title'] = combined_train_test['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
	title_Dict = {}
	title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
	title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
	title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
	title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
	title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
	title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))

	combined_train_test['Title'] = combined_train_test['Title'].map(title_Dict)

	# 为了后面的特征分析，这里我们也将 Title 特征进行facrorizing
	combined_train_test['Title'] = pd.factorize(combined_train_test['Title'])[0]

	title_dummies_df = pd.get_dummies(combined_train_test['Title'], prefix=combined_train_test[['Title']].columns[0])
	combined_train_test = pd.concat([combined_train_test, title_dummies_df], axis=1)

	combined_train_test['Name_length'] = combined_train_test['Name'].apply(len)

def fare_variable():
	global combined_train_test
	#fare
	combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(combined_train_test.groupby('Pclass').transform(np.mean))
	combined_train_test['Group_Ticket'] = combined_train_test['Fare'].groupby(by=combined_train_test['Ticket']).transform('count')
	combined_train_test['Fare'] = combined_train_test['Fare'] / combined_train_test['Group_Ticket']
	combined_train_test.drop(['Group_Ticket'], axis=1, inplace=True)

	combined_train_test['Fare_bin'] = pd.qcut(combined_train_test['Fare'], 5)
	combined_train_test['Fare_bin_id'] = pd.factorize(combined_train_test['Fare_bin'])[0]

	fare_bin_dummies_df = pd.get_dummies(combined_train_test['Fare_bin_id']).rename(columns=lambda x: 'Fare_' + str(x))
	combined_train_test = pd.concat([combined_train_test, fare_bin_dummies_df], axis=1)
	combined_train_test.drop(['Fare_bin'], axis=1, inplace=True)

#pclass

# 建立PClass Fare Category
def pclass_fare_category(df, pclass1_mean_fare, pclass2_mean_fare, pclass3_mean_fare):
    if df['Pclass'] == 1:
        if df['Fare'] <= pclass1_mean_fare:
            return 'Pclass1_Low'
        else:
            return 'Pclass1_High'
    elif df['Pclass'] == 2:
        if df['Fare'] <= pclass2_mean_fare:
            return 'Pclass2_Low'
        else:
            return 'Pclass2_High'
    elif df['Pclass'] == 3:
        if df['Fare'] <= pclass3_mean_fare:
            return 'Pclass3_Low'
        else:
            return 'Pclass3_High'

def pclass_variable():
	global combined_train_test
	Pclass1_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([1]).values[0]
	Pclass2_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([2]).values[0]
	Pclass3_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([3]).values[0]

	# 建立Pclass_Fare Category
	combined_train_test['Pclass_Fare_Category'] = combined_train_test.apply(pclass_fare_category, args=(
	 Pclass1_mean_fare, Pclass2_mean_fare, Pclass3_mean_fare), axis=1)
	pclass_level = LabelEncoder()

	# 给每一项添加标签
	pclass_level.fit(np.array(
	 ['Pclass1_Low', 'Pclass1_High', 'Pclass2_Low', 'Pclass2_High', 'Pclass3_Low', 'Pclass3_High']))

	# 转换成数值
	combined_train_test['Pclass_Fare_Category'] = pclass_level.transform(combined_train_test['Pclass_Fare_Category'])

	# dummy 转换
	pclass_dummies_df = pd.get_dummies(combined_train_test['Pclass_Fare_Category']).rename(columns=lambda x: 'Pclass_' + str(x))
	combined_train_test = pd.concat([combined_train_test, pclass_dummies_df], axis=1)
	combined_train_test['Pclass'] = pd.factorize(combined_train_test['Pclass'])[0]

#parch and sibsp
def family_size_category(family_size):
    if family_size <= 1:
        return 'Single'
    elif family_size <= 4:
        return 'Small_Family'
    else:
        return 'Large_Family'
def parch_variable():
	global combined_train_test
	combined_train_test['Family_Size'] = combined_train_test['Parch'] + combined_train_test['SibSp'] + 1
	combined_train_test['Family_Size_Category'] = combined_train_test['Family_Size'].map(family_size_category)

	le_family = LabelEncoder()
	le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
	combined_train_test['Family_Size_Category'] = le_family.transform(combined_train_test['Family_Size_Category'])

	family_size_dummies_df = pd.get_dummies(combined_train_test['Family_Size_Category'],
	                                     prefix=combined_train_test[['Family_Size_Category']].columns[0])
	combined_train_test = pd.concat([combined_train_test, family_size_dummies_df], axis=1)
print(combined_train_test.head())
def age_variable():
	global combined_train_test
	#age
	missing_age_df = pd.DataFrame(combined_train_test[['Age', 'Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size', 'Family_Size_Category','Fare', 'Fare_bin_id', 'Pclass']])

	missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
	missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]
	missing_age_test.head()

	def fill_missing_age(missing_age_train, missing_age_test):
	    missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
	    missing_age_Y_train = missing_age_train['Age']
	    missing_age_X_test = missing_age_test.drop(['Age'], axis=1)

	    # model 1  gbm
	    gbm_reg = GradientBoostingRegressor(random_state=42)
	    gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [4], 'learning_rate': [0.01], 'max_features': [3]}
	    gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
	    gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
	    print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))
	    print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))
	    print('GB Train Error for "Age" Feature Regressor:' + str(gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
	    missing_age_test.loc[:, 'Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)
	    print(missing_age_test['Age_GB'][:4])
	    
	    # model 2 rf
	    rf_reg = RandomForestRegressor()
	    rf_reg_param_grid = {'n_estimators': [200], 'max_depth': [5], 'random_state': [0]}
	    rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
	    rf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
	    print('Age feature Best RF Params:' + str(rf_reg_grid.best_params_))
	    print('Age feature Best RF Score:' + str(rf_reg_grid.best_score_))
	    print('RF Train Error for "Age" Feature Regressor' + str(rf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
	    missing_age_test.loc[:, 'Age_RF'] = rf_reg_grid.predict(missing_age_X_test)
	    print(missing_age_test['Age_RF'][:4])

	    # two models merge
	    print('shape1', missing_age_test['Age'].shape, missing_age_test[['Age_GB', 'Age_RF']].mode(axis=1).shape)
	    # missing_age_test['Age'] = missing_age_test[['Age_GB', 'Age_LR']].mode(axis=1)

	    missing_age_test.loc[:, 'Age'] = np.mean([missing_age_test['Age_GB'], missing_age_test['Age_RF']])
	    print(missing_age_test['Age'][:4])

	    missing_age_test.drop(['Age_GB', 'Age_RF'], axis=1, inplace=True)

	    return missing_age_test

	combined_train_test.loc[(combined_train_test.Age.isnull()), 'Age'] = fill_missing_age(missing_age_train, missing_age_test)

def ticket_variable():
	global combined_train_test
	#ticket
	combined_train_test['Ticket_Letter'] = combined_train_test['Ticket'].str.split().str[0]
	combined_train_test['Ticket_Letter'] = combined_train_test['Ticket_Letter'].apply(lambda x: 'U0' if x.isnumeric() else x)

	# combined_train_test['Ticket_Number'] = combined_train_test['Ticket'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
	# combined_train_test['Ticket_Number'].fillna(0, inplace=True)

	combined_train_test['Ticket_Letter'] = pd.factorize(combined_train_test['Ticket_Letter'])[0]


#cabin
def cabin_variable():
	global combined_train_test
	combined_train_test.loc[combined_train_test.Cabin.isnull(), 'Cabin'] = 'U0'
	combined_train_test['Cabin'] = combined_train_test['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)


def change_variables():
	embark_variable()
	sex_variable()
	title_variable()
	fare_variable()
	pclass_variable()
	parch_variable()
	age_variable()	
	ticket_variable()
	cabin_variable()
	
	



def data_processing():
	global combined_train_test

	# analyze_data()
	change_variables()

	Correlation = pd.DataFrame(combined_train_test[
	 ['Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size', 'Family_Size_Category','Fare', 'Fare_bin_id', 'Pclass', 
	  'Pclass_Fare_Category', 'Age', 'Ticket_Letter', 'Cabin']])

	colormap = plt.cm.viridis
	plt.figure(figsize=(14,12))
	plt.title('Pearson Correlation of Features', y=1.05, size=15)
	sns.heatmap(Correlation.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

	g = sns.pairplot(combined_train_test[[u'Survived', u'Pclass', u'Sex', u'Age', u'Fare', u'Embarked',
	    u'Family_Size', u'Title', u'Ticket_Letter']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
	g.set(xticklabels=[])


	scale_age_fare = preprocessing.StandardScaler().fit(combined_train_test[['Age','Fare', 'Name_length']])
	combined_train_test[['Age','Fare', 'Name_length']] = scale_age_fare.transform(combined_train_test[['Age','Fare', 'Name_length']])

	combined_data_backup = combined_train_test
	combined_train_test.drop(['PassengerId', 'Embarked', 'Sex', 'Name', 'Title', 'Fare_bin_id', 'Pclass_Fare_Category', 
	                       'Parch', 'SibSp', 'Family_Size_Category', 'Ticket'],axis=1,inplace=True)
	return combined_train_test

