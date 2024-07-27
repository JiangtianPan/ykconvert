#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/Users/tianjiaoli/miniconda3/bin/python -m pip install lightgbm


# In[1]:


#!D:\Conda\miniconda38\envs\ltj_jupyter\python -m pip install scorecardpy


# In[2]:


#!which python


# ### 加载包

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

import re
import string
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score,ConfusionMatrixDisplay, confusion_matrix,f1_score
import lightgbm as lgb

from scipy.sparse import hstack

def displayConfusionMatrix(y_true, y_pred, dataset):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Not Disaster","Disaster"],
        cmap=plt.cm.Blues
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1_score = tp / (tp+((fn+fp)/2))

    disp.ax_.set_title("Confusion Matrix on " + dataset + " Dataset -- F1 Score: " + str(f1_score.round(2)))


# ### 问题1. 首先对train.csv中的数据进行缺失值填充，keyword列缺少的值用 no _ key 关键词填充，location列缺少的值用 no _ location 关键词填充；统计与真实灾难、虚假灾难有关的推文数量占比，并进行数据分析和饼图可视化。(10分)

# In[6]:


df_train = pd.read_csv('train_data.csv', dtype={'id': np.int16, 'target': np.int8}, encoding='ISO-8859-1')
df_test = pd.read_csv('test_data.csv', dtype={'id': np.int16}, encoding='ISO-8859-1')

df_train = pd.read_csv('train_data.csv')
df_test = pd.read_csv('test_data.csv')

# Filling missing values
df_train['keyword'].fillna('no_key', inplace=True)
df_train['location'].fillna('no_location', inplace=True)

# Checking if missing values are filled
missing_values = df_train.isnull().sum()
missing_values


# In[7]:


# 进行数据分析和饼图可视化
# Classes are almost equally separated so they don't require any stratification by target in cross-validation.

# 统计与真实灾难、虚假灾难有关的推文数量占比
disaster_counts = df_train['target'].value_counts(normalize=True) * 100
tweet_counts = df_train['target'].value_counts()

# 设置绘图区域
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 统一标签和颜色
labels = ['Fake Disaster ('+str(tweet_counts[0]) + ')','Real Disaster ('+str(tweet_counts[1]) + ')']
colors = ['lightcoral', 'lightskyblue']

# 绘制饼图
axes[0].pie(disaster_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 15})
axes[0].axis('equal')
axes[0].set_title('Proportion of Real and Fake Disaster Tweets',fontsize=15)

# 绘制柱形图
sns.countplot(ax=axes[1], x=df_train['target'], hue=df_train['target'], palette=colors)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(labels,fontsize=15)
axes[1].set_xlabel('Tweet Category',fontsize=15)
axes[1].set_ylabel('Count',fontsize=15)
axes[1].set_title('Count of Real and Fake Disaster Tweets',fontsize=15)

# axes[1].tick_params(axis='x', labelsize=15)
# axes[1].tick_params(axis='y', labelsize=15)

# 显示图表
plt.tight_layout()
plt.show()


# ### 问题2. 请统计分析与真实灾难有关的推文内容，进行分词、清洗、词量统计并绘制词云图，分析推文文本中最常见的灾难关联关键词数量排名前10 是哪些？（20分）

# In[8]:


# 清洗文本数据
def clean_text(text):
    # 去除标点符号和数字
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r"http\S+", " ", text) # remove urls
    text = re.sub(r"RT ", " ", text) # remove rt
    text = re.sub(r"[^a-zA-Z\'\.\,\d\s]", " ", text) # remove special character except # @ . ,
    text = re.sub(r"[0-9]", " ", text) # remove number
    text = re.sub(r'\t', ' ', text) # remove tabs
    text = re.sub(r'\n', ' ', text) # remove line jump
    text = re.sub(r"\s+", " ", text) # remove extra white space
    # 去除单个字母的单词
    # \b 是单词边界，\w 匹配字母或数字，{1} 表示匹配一次
    text = re.sub(r'\b\w{1}\b', '', text)
    text = text.lower()
    text = text.strip()
    return text

# 合并所有推文内容
all_texts = ' '.join(df_train[df_train['target'] == 1]['text'].apply(clean_text))


#---------------------------------------------------丰富停词---------------------------------------------------
# 定义一些自定义停词
custom_stopwords = {'amp', 'u', 'im','look','will','https','http','one','two','hundred','new','news','via','say','year','co','yr','old'}

# 去掉虚假新闻中的高频词汇
# 使用 CountVectorizer 计算词频
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform([' '.join(df_train[df_train['target'] == 0]['text'].apply(clean_text))])
word_freq = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0]))

# 设定一个频率阈值
frequency_threshold = pd.Series(list(word_freq.values())).quantile(0.95)  # 设置为 90% 分位数
high_freq_words = {word for word, freq in word_freq.items() if freq > frequency_threshold}

# 定义停用词
stopwordlist = set(STOPWORDS).union(custom_stopwords).union(high_freq_words)

#------------------------------------------------------------------------------------------------------------
# 生成词云
wordcloud = WordCloud(width=1200, height=800, 
                      background_color='white', 
                      stopwords=stopwordlist, 
                      colormap='Set2',
                      min_font_size=10).generate(all_texts)

# 显示词云
plt.figure(figsize=(12, 8)) 
plt.imshow(wordcloud, interpolation='bilinear') 
plt.axis("off") 
plt.tight_layout(pad=0) 

plt.show()


# In[9]:


# 提取词云中的词及其频率
word_freq = wordcloud.words_

# 获取词频前十的词
top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

# 打印词频前十的词
print("Top 10 words and their frequencies:")
for word, freq in top_words:
    print(f"{word}: {freq}")


# In[9]:


# sns.barplot(y=df_train['keyword'].value_counts()[:10].index,x=df_train['keyword'].value_counts()[:10],
#             orient='h')
# plt.title("Top ten keywords ")
# plt.show()


# In[10]:


# 分析推文文本中最常见的灾难关联关键词数量排名前10 是哪些？
df_train['target_mean'] = df_train.groupby('keyword')['target'].transform('mean')

# 对每个关键词按target_mean降序排列，并计算排名
df_train['rank'] = df_train['target_mean'].rank(ascending=False, method = 'dense')

# 选择排名前十的关键词
df_train_top10 = df_train[df_train['rank'] <= 10]

# 创建图像
fig = plt.figure(figsize=(8, 6), dpi=100)

# 绘制关键词的计数图，只显示排名前十的关键词
sns.countplot(y=df_train_top10.sort_values(by='target_mean', ascending=False)['keyword'],
              hue=df_train_top10.sort_values(by='target_mean', ascending=False)['target'],
              palette= colors )

plt.title('Target Distribution in Top 10 Keywords')

# 显示图像
plt.show()


# ### 问题3. 请根据数据中的location统计来源于中国、美国、印度、英国的推文数量，并绘制推文数量的直方图（有效的location必须为“China”、“USA”、“India”、“UK”）。（10分）

# In[11]:


# 过滤数据，只保留指定的国家
filtered_locations = df_train['location'].loc[df_train['location'].isin(['China', 'USA', 'India', 'UK'])]

# 计算各个国家的推文数量
location_counts = filtered_locations.value_counts()

# 绘制推文数量的直方图
plt.figure(figsize=(10, 6))
sns.histplot(filtered_locations, bins=len(location_counts), kde=False, color='skyblue')

# 设置图像属性
plt.xlabel('Location')
plt.ylabel('Frequency')
plt.title('Distribution of Tweet Counts by Location')

# 设置x轴的刻度为国家名称
plt.xticks(ticks=range(len(location_counts)), labels=location_counts.index, rotation=45)

# 显示图像
plt.tight_layout()  # 自动调整子图参数
plt.show()


# ### 问题4. 建立一个与真实灾难有关的推文识别模型，基于模型对附件中的测试数据test.csv进行评测，将评测结果按submission_sample.csv的格式输出submission.csv。(30分)

# In[87]:


# 选择特征和标签
X = df_train['text'].apply(clean_text)
y = df_train['target']

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[88]:


# 初始化 TF-IDF 向量化器
tfidf_vectorizer = TfidfVectorizer(stop_words = list(stopwordlist), max_features=10000)

# 训练 TF-IDF 向量化器并转换训练和测试数据
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# In[89]:


# 初始化 SVM 模型
# Sigmoid
svm_model = SVC(kernel='sigmoid', probability=True, random_state=42)

# 训练SVM模型
svm_model.fit(X_train_tfidf , y_train)

# 在测试集上进行预测
y_pred_svm = svm_model.predict_proba(X_test_tfidf)

y_pred_svm_train = svm_model.predict_proba(X_train_tfidf)

# 评估模型
print("SVM Model Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))


# SVM Model Accuracy: 0.7893939393939394
# SVM Classification Report:
#                precision    recall  f1-score   support

#            0       0.80      0.85      0.82       770
#            1       0.77      0.71      0.74       550

#     accuracy                           0.79      1320
#    macro avg       0.79      0.78      0.78      1320
# weighted avg       0.79      0.79      0.79      1320

# SVM Model Accuracy: 0.8
# SVM Classification Report:
#                precision    recall  f1-score   support

#            0       0.82      0.84      0.83       770
#            1       0.77      0.74      0.76       550

#     accuracy                           0.80      1320
#    macro avg       0.80      0.79      0.79      1320
# weighted avg       0.80      0.80      0.80      1320


# SVM Model Accuracy: 0.8068181818181818
# SVM Classification Report:
#                precision    recall  f1-score   support

#            0       0.81      0.87      0.84       770
#            1       0.80      0.72      0.76       550

#     accuracy                           0.81      1320
#    macro avg       0.80      0.79      0.80      1320
# weighted avg       0.81      0.81      0.81      1320


# In[15]:


displayConfusionMatrix(y_test, y_pred_svm,'svm test results')


# In[92]:


# 定义自定义的f1_score评估函数
def feval_custom(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    y_pred = np.round(y_pred)
    return 'f1', f1_score(y_true, y_pred), True

# 训练 LightGBM 模型
lgb_train = lgb.Dataset(X_train_tfidf, label=y_train)
lgb_eval = lgb.Dataset(X_test_tfidf, label=y_test, reference=lgb_train)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01
}

lgb_model = lgb.train(params, lgb_train, num_boost_round=1000,feval=feval_custom)

# 在测试集上进行预测
y_pred_lgb = (lgb_model.predict(X_test_tfidf, num_iteration=lgb_model.best_iteration) > 0.423046).astype(int)

# 评估模型
print("LightGBM Model Accuracy:", accuracy_score(y_test, y_pred_lgb))
print("LightGBM Classification Report:\n", classification_report(y_test, y_pred_lgb))


# In[86]:


displayConfusionMatrix(y_test, y_pred_lgb,'lgb test results')


# ### 'https://tfhub.dev/google/universal-sentence-encoder-large/5' 尝试

# In[33]:


import tensorflow_hub as hub 
import tensorflow as tf 
from tqdm.notebook import tqdm 


# In[34]:


def transfrom(text_train, text_test):
    large_use = 'https://tfhub.dev/google/universal-sentence-encoder-large/5'
    embed = hub.load(large_use)

    vector_train = [tf.reshape(embed([line]), [-1]).numpy() for line in tqdm(text_train)]
    vector_test = [tf.reshape(embed([line]), [-1]).numpy() for line in tqdm(text_test)]

    return vector_train, vector_test


# In[43]:


vector_train, vector_test = transfrom(df_train.text,df_test.text)


# In[44]:


model = SVC(kernel='sigmoid', probability=True, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(vector_train, df_train.target, test_size=0.2, random_state=2020)
model.fit(X_train, y_train)

preds = model.predict(X_val)
print('Accuracy score', accuracy_score(y_val, preds))
print('f1_score', f1_score(y_val, preds))


# In[45]:


displayConfusionMatrix(y_val, preds,'universal-sentence-encoder-large')


# In[53]:


import scorecardpy as sc


# In[62]:


pd.DataFrame({"obs": y_train,"model": y_pred_svm_train})


# In[73]:


sc.perf_eva(y_train,y_pred_svm_train[:, 1])


# In[91]:


sc.perf_eva(y_test, y_pred_svm[:, 1])


# In[49]:


y_pred_svm_train = svm_model.predict(X_train_tfidf)


# In[81]:


y_pred_svm = y_pred_svm[:, 1]


# In[83]:


#输出confusion matrix
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from sklearn.metrics import confusion_matrix

precision, recall, thresholds = precision_recall_curve(y_train, y_pred_svm_train)
fscore = (2 * precision * recall)/(precision + recall)
ix = np.nanargmax(fscore)
threshold_train = thresholds[ix]

def PR_function(actual_y, y_pred, threshold):
    y_pred_01 = []
    for i in range(len(y_pred)):
        y_pred_01.append(0 if y_pred[i] < threshold else 1)
    C2= confusion_matrix(actual_y, y_pred_01, labels=[0, 1])
    TN = C2[0,0]
    TP = C2[1,1]
    FP = C2[0,1]
    FN = C2[1,0]
    precision = round(TP/(TP+FP), 3)
    recall = round(TP/(TP+FN), 3)
    return precision, recall, y_pred_01

precision_train, recall_train, y_pred_01_train = PR_function(y_train, y_pred_svm_train, threshold_train)
precision_test, recall_test, y_pred_01_test = PR_function(y_test, y_pred_svm, threshold_train)

print(f"根据train data所选的最优threshold为{round(threshold_train, 6)}；")
print(f"训练集的precision为{precision_train}, recall为{recall_train}；")
print(f"测试集的precision为{precision_test}, recall为{recall_test}；")


# In[ ]:


print(y_train, y_pred_svm_train)

