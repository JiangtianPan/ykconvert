#!/usr/bin/env python
# coding: utf-8

# In[133]:


#读取数据
import pandas as pd
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")
print(f"训练数据量为{test_df.shape[0]}，测试数据量为{test_df.shape[0]}。")
train_df.head()


# In[134]:


#1.1 缺失值填充
train_df = train_df.copy().fillna(0)
test_df = train_df.copy().fillna(0)
#查缺失值所在行并确认所有空缺值已被填充
train_df[train_df.isnull().any(axis = 1)]


# In[3]:


#1.2 去除重复行
train_df.drop_duplicates(inplace = True)
test_df.drop_duplicates(inplace = True)
print(f"去除重复行后，训练数据量为{train_df.shape[0]}，测试数据量为{test_df.shape[0]}。")


# In[4]:


def grade_replace(x, replace_num):
    if x < 0 or x > 200:
        return replace_num
    else:
        return x

grade_replace(2, 100)


# In[5]:


#1.3 对Previous qualification (grade)、Admission grade列进行异常值处理（异常情况说明：不在指定范围内的值均为异常值）
def grade_replace(x, replace_num):
    if x < 0 or x > 200:
        return replace_num
    else:
        return x
previous_qualification_grade_median = train_df["Previous qualification (grade)"].median()
admission_grade_median = train_df["Admission grade"].median()
train_df["Previous qualification (grade)"] = train_df["Previous qualification (grade)"].apply(lambda x: grade_replace(x, previous_qualification_grade_median))
test_df["Previous qualification (grade)"] = test_df["Previous qualification (grade)"].apply(lambda x: grade_replace(x, previous_qualification_grade_median))
train_df["Admission grade"] = train_df["Admission grade"].apply(lambda x: grade_replace(x, admission_grade_median))
test_df["Admission grade"] = test_df["Admission grade"].apply(lambda x: grade_replace(x, admission_grade_median))
#确认异常值清洗完毕 #train_df["Previous qualification (grade)"].describe()


# In[6]:


#确认异常值清洗完毕
train_df["Previous qualification (grade)"].describe()


# In[65]:


#1.4 筛选表格中Previous qualification的取值为1, 9, 10, 12, 14, 15, 19, 38的记录作为后续操作步骤的数据集并打印前五行记录。
train_df_selected = train_df[train_df["Previous qualification"].isin([1, 9, 10, 12, 14, 15, 19, 38])]
train_df_selected.head(5)


# **问题2**  
# （1）请根据问题1筛选输出的数据集绘制各个特征的核密度估计（KDE）图；  
# （2）绘制各个特征的箱型图（x轴为Target）；  
# （3）绘制各个特征的相关性矩阵热力图。（20分）

# In[22]:


#2.1 根据问题1筛选输出的数据集绘制各个特征的核密度估计（KDE）图
import seaborn as sns
import matplotlib.pyplot as plt

plt_list = train_df_selected.columns.tolist()
plt_list.remove("id")
plt_list.remove("Target")
fig, axs = plt.subplots(nrows = 6, ncols = 5, figsize=(20, 18))

for i, (col_name, ax) in enumerate(zip(plt_list, axs.flatten())):
    sns.kdeplot(data = train_df_selected, x= col_name, fill=True, ax=ax)
    ax.set_title(f'KDE Plot for {col_name}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')

plt.tight_layout()
plt.show()


# In[21]:


#2.2 绘制各个特征的箱型图（x轴为Target）
fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(20, 18))
for i, (col_name, ax) in enumerate(zip(plt_list, axes.flatten())):
    train_df_selected.boxplot(column=[col_name], by = 'Target', ax=ax)
    ax.set_yticklabels([]) #移除y轴刻度标签

plt.title("")
plt.tight_layout()
plt.show()


# In[38]:


#2.3 绘制各个特征的相关性矩阵热力图
import numpy as np
data_matrix = train_df_selected.drop(columns = ["Target", "id"]).corr()
sns.heatmap(data_matrix)


# **问题3**  
# 高校学生画像是指根据学生的在校表现、社会经济背景等信息而抽象出来的标签化模型。通俗说就是给高校学生打标签，而标签是通过对对象信息分析而来的高度精炼的特征标识，通过打标签可以利用一些高度概括、容易理解的特征来描述高校学生。  
# 请根据问题1筛选输出的数据集，基于Curricular units 1nd sem (grade)、Curricular units 2nd sem (grade)列构建学生的在校表现标签School Performance（两学期均绩均值≥15为优秀，15＞两学期均绩均值≥10为良好，10＞两学期均绩均值≥5为一般，5＞两学期均绩均值≥0为差）；  
# 基于Unemployment rate、Inflation rate、GDP构建学生所在地区的社会经济背景标签Social Economy（Unemployment rate、Inflation rate、GDP的权重分别为0.4、0.3、0.3，综合得分≤0.33为富裕，0.33＜综合得分≤0.66为普通，0.66＜综合得分为贫穷）。（10分）

# In[66]:


def performance_func(x):
    if x >= 15:
        return "优秀"
    elif x >= 10:
        return "良好"
    elif x >= 5:
        return "一般"
    else:
        return "差"

def economy_fun(x):
    if x <= 0.33:
        return "富裕"
    elif x <= 0.66:
        return "普通"
    else:
        return "贫穷"
        
train_df_selected["School Performance"] = train_df_selected[["Curricular units 1st sem (grade)"
                                                             ,"Curricular units 2nd sem (grade)"]].mean(axis = 1).apply(lambda x: performance_func(x))
train_df_selected["Social Economy"] = (train_df_selected["Unemployment rate"]/100 * 0.4 + 
                                       train_df_selected["Inflation rate"]/100 * 0.3 + 
                                       train_df_selected["GDP"] * 0.3).apply(lambda x: economy_fun(x))


# In[85]:


train_df_selected.groupby(["Target"]).count()


# **问题4**
# 高等教育机构拟统计不同本科生的毕业情况，通过使用数据分析技术在学术道路的早期阶段识别风险，从而有助于减少高等教育中的学术辍学和失败。  
# 请根据问题1筛选输出的数据集，完成学生毕业情况模型构建（请至少使用3个以上的模型完成此题）。（20分）

# In[99]:


#分模型训练用train&test
from sklearn.model_selection import train_test_split

def target_func(x):
    if x == "Graduate":
        return 1
    elif x == "Enrolled":
        return 2
    else:
        return 0

Y = train_df_selected["Target"].apply(lambda x: target_func(x)) 
train_X, test_X, train_y, test_y = train_test_split(train_df_selected.drop(columns = ["Target", "id"]),
                                                    Y,
                                                    test_size = 0.3, random_state = 2024, stratify = Y)


# In[100]:


disc_list = ["School Performance", "Social Economy"]
train_X[disc_list] = train_X[disc_list].astype("category")
test_X[disc_list] = test_X[disc_list].astype("category")
train_data = lgb.Dataset(data=train_X,label=train_y, categorical_feature = disc_list)
test_data = lgb.Dataset(data=test_X,label=test_y, categorical_feature = disc_list)


# In[128]:


#构建lightgbm模型
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
from sklearn.metrics import f1_score

params = {'num_leaves': 30, #叶子节点数 #结果对最终效果影响较大，越大值越好，太大会出现过拟合
          'min_data_in_leaf': 30,
          'objective': 'multiclass', #定义的目标函数，可以定regression
          'max_depth': 6,
          'num_class': 3,
          'learning_rate': 0.1,#学习速率
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt", #设置提升类型
          "feature_fraction": 0.9, #建树的特征选择比例
          "bagging_freq": 1, #k意味着每k次迭代进行bagging
          "bagging_fraction": 0.8, #建树的样本采样比例
          "bagging_seed": 11,
          "lambda_l1": 10, #l1正则
          'lambda_l2': 10,
          "verbosity": -1, #<0显示致命的，=0显示错误，1显示信息
          "nthread": -1, #线程数量，-1表示全部线程，线程越多，运行的速度越快
          'metric': 'multi_logloss' #评估函数
          }

num_round = 1000
callbacks = [log_evaluation(period = 20), early_stopping(stopping_rounds = 50)]
clf_lgb = lgb.train(params,train_data,num_round,valid_sets=[test_data], categorical_feature = disc_list,
                callbacks = callbacks)

y_pred_train = clf_lgb.predict(train_X, num_iteration = clf.best_iteration)
y_pred_test = clf_lgb.predict(test_X, num_iteration = clf.best_iteration)


# In[129]:


from sklearn.metrics import accuracy_score, classification_report
y_pred = clf_lgb.predict(test_X, num_iteration=clf.best_iteration)
y_pred_class_clf = np.argmax(y_pred, axis=1)

print("Accuracy:", accuracy_score(test_y, y_pred_class_clf))
print("Classification Report:\n", classification_report(test_y, y_pred_class_clf))


# In[121]:


from sklearn.linear_model import LogisticRegression

clf_multinomial = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
clf_multinomial.fit(train_X.drop(columns = disc_list), train_y)
y_pred_multinomial = clf_multinomial.predict(test_X.drop(columns = disc_list))

print("\nMultinomial Logistic Regression:")
print("Accuracy:", accuracy_score(test_y, y_pred_multinomial))
print(classification_report(test_y, y_pred_multinomial))


# In[127]:


from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(train_X.drop(columns = disc_list), train_y)
y_pred_test_rf = clf_rf.predict(test_X.drop(columns = disc_list))

print("\nRandom Forest:")
print("Accuracy:", accuracy_score(test_y, y_pred_test_rf))
print(classification_report(test_y, y_pred_test_rf))

