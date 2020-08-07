# Disaster Response Pipeline Project

### requirements:

* python==3.6.5
* pandas==0.25.3
* scikit-learn==0.21.3
* SQLAlchemy==1.3.10
* Flask==1.1.1
* nltk==3.4.5
* numpy==1.17.3
* plotly==4.2.1

### 一、项目背景

灾难发生时需要对消息进行分类并准确地传到到灾难应急机构。

### 二、数据源

* data/disaster_categories.csv
* data/disaster_messages.csv 

### 三、项目结构

1.ELT pipeline
process_data.py: 数据处理管道，主要实现：

* 加载messages与categories数据；
* 合并两个数据集
* 处理数据
* 存储数据到SQLite数据库

2.ML pipeline
train_classifier.py: ML pipeline，主要实现：

* 从SQLite数据库加载数据
* 将数据集划分为train 和 test
* 搭建文本处理和机器学习管道
* 用GridSearchCV对模型进行参数调优和训练
* 输出test数据集结果
* 将best model输出为pickle文件

3.Flash服务端部署文件: 
用于以网页的形式展示结果。部署步骤：

1) Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans data and stores in database
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterData.db`
    
- To run ML pipeline that trains classifier and saves
    `python models/train_classifier.py data/DisasterData.db models/pipeline.pkl`

2) Run the following command in the app's directory to run your web app.
`python run.py`

3) Go to http://127.0.0.1:3001/

