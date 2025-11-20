This project is called Information aggregation function and model debugging WebUI (Original Erlangshen Launcher <abbreviated as 原神启动器 in Chinese>), which is created for an NLP presentation project in USST.
该项目名为“信息聚合功能与模型调试WebUI”（原二郎神启动器<中文简称原神启动器>），是为上海理工大学的一个自然语言处理演示项目而创建的。

You only need to place the Chinese NLU model into the "Models" folder to quickly test some Chinese NLU task results.
你只需要将中文NLU的模型放入Models文件夹，就可以快速测试一些中文NLU任务结果。

File Structure:
├── Program
│   ├── Models
│   ├── Spiders
│   │   ├── AllSpider
│   │   │   ├── baidu_news_spider.py
│   │   │   ├── lenovo_app_spider.py
│   │   │   └── research_paper_spider.py
│   │   ├── Information
│   │   │   ├── AppDescriptions
│   │   │   ├── Literature
│   │   │   └── News
│   │   └── spider_launcher.py
│   ├── quick_start.py
│   └── webui_app.py
