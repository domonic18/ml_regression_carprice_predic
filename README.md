## 项目背景
在当今的汽车市场中，二手车的价格受到多种因素的影响。通过机器学习技术，我们可以分析历史数据，建立模型，从而更准确地预测二手车的价格。这不仅可以帮助消费者做出更明智的购买决策，也可以为卖家提供合理的定价参考。

## 项目目录结构
```
ml_regression_carprice_predic
├── README.md
├── requirements.txt
├── src
│   ├── app.py
│   ├── data_processing
│   ├── datasets
│   ├── models
│   ├── training
│   ├── utils
├── .gitignore
```

### 数据集简介
#### 下载地址
https://tianchi.aliyun.com/dataset/175540

#### 内容简介
这是阿里天池上的一个数据集，该数据集为二手车交易价格数据集，数据来自某交易平台的二手车交易记录，总数据量超过40w，包含31列变量信息，其中15列为匿名变量。

#### 数据情况

| 数据名称| 上传日期 | 大小  |
|:--------------------------|:------------|:---------|
| used_car_testB_20200421.csv | 2024-04-16 | 17.06MB |
| used_car_train_20200313.csv | 2024-04-16 | 51.77MB |

#### 数据字段
![](doc/数据集截图.png)


## 使用方法
1. 创建虚拟环境
```
conda create --name deeplearning python=3.10
```

2. 激活虚拟环境
```
conda activate deeplearning
```

3. 安装依赖
```
pip install -r requirements.txt
```

4. 下载数据集
访问下载地址，将数据集下载到data目录下

5. 运行项目
```
streamlit run src/app.py
```

### 数据分析
1. 在streamlit页面中，点击“数据分析”按钮。
2. 选择data目录下的used_car_train_20200313.csv文件，点击“开始分析”按钮，即可查看数据分析结果。

![数据分析截图](doc/数据分析截图.png)

### 数据清洗
1. 在streamlit页面中，上传data目录下的used_car_train_20200313.csv文件。
2. 点击“开始数据清洗”按钮，开始数据清洗操作。
3. 数据清洗完毕后，点击保存数据按钮，将清理后的数据保存到data目录下的used_car_train_20200313_cleaned.csv文件中。
![数据清洗截图](doc/数据清洗截图.png)

### 模型训练
1. 在streamlit页面中，选择清理后的数据文件used_car_train_20200313_cleaned.csv
2. 配置模型训练的参数，包括训练轮数、学习率。
3. 点击“开始训练”按钮，即可开始训练。
4. 模型训练完毕后，系统会输出模型文件保存路径和训练结果。

![模型训练截图](doc/模型训练截图.png)

### 模型预测
1. 在streamlit页面中，点击“模型预测”按钮。
2. 选择data目录下的used_car_train_20200313.csv文件，点击“开始预测”按钮，即可查看模型预测结果。
![模型预测截图](doc/模型预测截图.png)

## 声明
本项目仅供学习交流使用，原创内容不易，转载请注明出处。