# Disaster Response Pipeline Project

### 一、项目背景
灾难发生时需要对消息进行分类并准确地传到到灾难应急机构。

### 二、项目结构
1.ELT pipeline
process_data.py: 数据处理管道，主要实现：
* 加载messages与categories数据；
* 合并两个数据集
* 处理数据
* 存储数据到SQLite数据库

2.Machine Learning pipeline
train_classifier.py: 机器学习管道，主要实现：
* 从SQLite数据库加载数据
* 将数据集划分为train 和 test
* 搭建文本处理和机器学习管道
* 用GridSearchCV对模型进行参数调优和训练
* 输出test数据集结果
* 将best model输出为pickle文件

3.Flash服务端部署文件: run.py
用于以网页的形式展示结果。

### 三、数据源
* data/disaster_categories.csv
* data/disaster_messages.csv

### 三、notebook文件夹
* ETL Pipeline Preparation-zh.html
* ETL Pipeline Preparation-zh.ipynb
* ML Pipeline Preparation-zh.html
* ML Pipeline Preparation-zh.ipynb
主要用于构建pipeline。

### 四、Flask部署:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterData.db`
        
        Run the commands:
        
        cd /data/ python process_data.py disaster_messages.csv  disaster_categories.csv /models/DisasterData.db '''
        
    - To run ML pipeline that trains classifier and saves
        `python ../models/train_classifier.py ../data/DisasterData.db models/pipeline.pkl`
        
        Run the cmd:
        
        cd /models/ python train_classifier.py DisasterData.db pipeline.pkl

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/
