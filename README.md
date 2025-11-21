## This project is called Information aggregation function and model debugging WebUI (Original Erlangshen Launcher <abbreviated as 原神启动器 in Chinese>), which is created for an NLP presentation project in USST.
## 该项目名为“信息聚合功能与模型调试WebUI”（原二郎神启动器<中文简称原神启动器>），是为上海理工大学的一个自然语言处理演示项目而创建的。

### You only need to place the Chinese NLU model into the "Models" folder to quickly test some Chinese NLU task results.
### 你只需要将中文NLU的模型放入Models文件夹，就可以快速测试一些中文NLU任务结果。

### The project was inspired by Erlangshen Model, namely https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B.
### 该项目灵感来源于Erlangshen模型，即https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B。

### But now the project is independent of Erlangshen, it could be utilized to test any Chinese NLU model.
### 但现在该项目已独立于二郎神，可用于测试任何中文自然语言理解（NLU）模型。

### Erlangshen Model is a Model from project Fengshenbang, https://github.com/IDEA-CCNL/Fengshenbang-LM?tab=readme-ov-file#封神榜科技成果.
### Erlangshen模型来自封神榜项目，https://github.com/IDEA-CCNL/Fengshenbang-LM?tab=readme-ov-file#封神榜科技成果。

### The spider "baidu_news_spider" in folder "AllSpider", was reconstructed on the basis of Python3Spiders's project "AllNewsSpider", https://github.com/Python3Spiders/AllNewsSpider/tree/master.
### 在“AllSpider”文件夹中的爬虫程序“baidu_news_spider”是在Python3Spiders的“AllNewsSpider”项目的基础上重构的，该项目位于https://github.com/Python3Spiders/AllNewsSpider/tree/master。

#### File Structure:
#### ├── Program
#### │   ├── Models
#### │   ├── Spiders
#### │   │   ├── AllSpider
#### │   │   │   ├── baidu_news_spider.py
#### │   │   │   ├── lenovo_app_spider.py
#### │   │   │   └── research_paper_spider.py
#### │   │   ├── Information
#### │   │   │   ├── AppDescriptions
#### │   │   │   ├── Literature
#### │   │   │   └── News
#### │   │   └── spider_launcher.py
#### │   ├── quick_start.py
#### │   └── webui_app.py


