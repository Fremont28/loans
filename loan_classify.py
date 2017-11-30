#import dataset
loans=pd.read_csv("LoanStats_2017Q1.csv",encoding='latin-1')

#fill is n/a with averages
loans.fillna(loans.mean()).dropna(axis=1,how='all')
#fill in missing values (with median)
loans=loans.fillna(method='ffill') #maye this works better?

#assign numeric to loan grades? source: https://chrisalbon.com/python/convert_categorical_to_numeric.html
def grade_to_numeric(x):
    if x=='A':
        return 1
    else:
        return 0 

#apply the function to the 'loan grade' variable
loans['grade_score']=loans['grade'].apply(grade_to_numeric)

#if loans exceeds 10,000?
loans['high_loan']=np.where(loans['loan_amnt']>10000,0,1)

#subset loans dataframe 'loan_sub'
loans_sub=loans[['loan_amnt','int_rate','installment','delinq_2yrs','inq_last_6mths','pub_rec','grade_score']]
loans_sub=loans_sub.fillna(method='ffill')
loans_sub.dropna(subset=['grade_score'],how="all",inplace=True)
#split the data into test and train datasets 
sample_loans=np.random.rand(len(loans_sub))<0.8
train_loans=loans_sub[sample_loans] 
test_loans=loans_sub[~sample_loans]

X_train1=train_loans[['loan_amnt','int_rate','installment','delinq_2yrs','inq_last_6mths','pub_rec']]
X_test1=test_loans[['loan_amnt','int_rate','installment','delinq_2yrs','inq_last_6mths','pub_rec']]
y_train1=train_loans['grade_score']
y_test1=test_loans['grade_score']

#normalize dataset (features)
from sklearn.preprocessing import scale
X_train_scale=(X_train1)
X_test_scale=(X_test1)

#1. Adaboost algorithm source: https://stackoverflow.com/questions/13413590/how-to-drop-rows-of-pandas-dataframe-whose-value-in-certain-columns-is-nan
from sklearn.ensemble import AdaBoostClassifier #For Classification
from sklearn.ensemble import AdaBoostRegressor #For Regression
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier() #using decision tree as a base estimator 
clf=AdaBoostClassifier(n_estimators=100,base_estimator=dt,learning_rate=1) 
clf.fit(X_train1,y_train1)
#predictions
predict_ada=clf.predict(X_test1)
predict_ada 
#check the model accuracy
accuracy_score(y_test1,clf.predict(X_test1)) #99.5% accuracy 

#2. boosting algorithm
from sklearn.ensemble import GradientBoostingClassifier #For Classification
from sklearn.ensemble import GradientBoostingRegressor #For Regression

clf_gbm=GradientBoostingClassifier(n_estimators=150,learning_rate=0.5,max_depth=3)
clf_gbm.fit(X_train,y_train)
#predictions
predict_gbm=clf_gbm.predict(X_test)
predict_gbm
#check accuracy
import collections, numpy 
collections.Counter(predict_gbm)
accuracy_score(y_test1,clf_gbm.predict(X_test1)) #not sure here?

#3. xgboost
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model_xg=XGBClassifier() 

#other data exploration
loans.describe() 
loans.info() 

#int rate vs. installment?
plt.scatter(loans.int_rate,loans.installment)
plt.title('')
plt.xlabel('installment payment')
plt.ylabel('interest rate')
plt.show() 
loans['int_rate'].corr(loans['installment']) r=0.18 

#installment vs. grade
loans['installment'].hist(by=loans['grade'])
plt.show() 
plt.xlabel('installment payment')

installment_grade=loans['installment'].groupby(loans['grade']).mean() 
installment_grade 

int_rate_grade=loans['int_rate'].groupby(loans['grade']).mean()
int_rate_grade 
#count loan grades
loans["grade"].value_counts() #39.7% C-grade, 34% B-grade,17.7% A-grade

#term (length of the loan)
loans['term'].value_counts() # 74.8% 36 months 
length_installments=loans['installment'].groupby(loans['term']).mean() 
length_installments #pay lower installments with 36 months 

#home_ownership vs. installment
home_own_installments=loans['installment'].groupby(loans['home_ownership']).mean() 
home_own_installments 

#purpose vs. installment
purpose_installments=loans['installment'].groupby(loans['purpose']).mean() 
purpose_installments 
loans["purpose"].value_counts() #debt consolidation and credit card 

#annual income vs. home ownership
ann_income_home=loans['annual_inc'].groupby(loans['home_ownership']).mean() 
ann_income_home 

#annual income vs grade?
income_grade=loans.boxplot(column=['annual_inc'],by=['grade'])
income_grade.set_ylabel("annual income")
income_grade.set_xlabel("Loan Grade")
plt.title('')
pylab.show() 

#pattern matching?
titles=loans[loans['emp_title'].str.contains("SECURITY|Security",na=False)]
titles_count=titles.groupby('emp_title')['emp_title'].count()
titles_count
titles.shape #dimensions of dataframe 

#security salary vs. others?

#what about states?
#interest rates 
state_int=loans['int_rate'].groupby(loans['addr_state']).mean() 
state_int #use tableu to visualize? #test for significance 

#state count
loans["addr_state"].value_counts() 

#loans visualizations-scatter
plt.scatter(loans.installment,loans.int_rate)
plt.title('Do Installment Payments Influence Interest Rates?')
plt.xlabel('installment')
plt.ylabel('interest rate')
plt.show()
plt.savefig('installment.png')

#interest rates (grade)
sns.boxplot(loans.)






















