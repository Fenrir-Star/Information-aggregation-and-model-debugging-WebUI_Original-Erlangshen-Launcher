import os
import sys
import subprocess
import time

def clear_screen():
    """清空屏幕"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """打印欢迎头"""
    print("=" * 50)
    print("🕷️  通用爬虫启动管理器 🕷️")
    print("=" * 50)

def get_script_path(folder, filename):
    """获取脚本的绝对路径"""
    # 获取启动器所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, folder, filename)

def run_spider(script_path, spider_name):
    """运行指定的爬虫脚本"""
    if not os.path.exists(script_path):
        print(f"\n❌ 错误：找不到文件 -> {script_path}")
        print(f"请确保 '{os.path.basename(script_path)}' 文件位于子文件夹中。")
        input("\n按回车键返回菜单...")
        return

    print(f"\n🚀 正在启动 [{spider_name}] ...")
    print("-" * 50)
    
    try:
        # 使用当前运行启动器的同一个 Python 解释器来运行子脚本
        # 使用 cwd 参数确保子脚本在自己的目录环境下运行（虽然你的脚本里处理了绝对路径，但这更稳妥）
        script_dir = os.path.dirname(script_path)
        subprocess.run([sys.executable, script_path], cwd=script_dir)
        
        print("-" * 50)
        print(f"✅ [{spider_name}] 运行结束")
    except KeyboardInterrupt:
        print(f"\n⚠️ 用户强制中断了 [{spider_name}]")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
    
    input("\n按回车键返回主菜单...")

def main():
    # 定义脚本路径配置
    # 格式: "显示名称": ("子文件夹名", "脚本文件名")
    spiders = {
        "1": {
            "name": "百度新闻爬虫 (Baidu News)",
            "folder": "AllSpider",
            "file": "baidu_news_spider.py"
        },
        "2": {
            "name": "联想应用商店爬虫 (Lenovo App)",
            "folder": "AllSpider",
            "file": "lenovo_app_spider.py"
        },
        "3": {
            "name": "学术论文爬虫 (Research Paper)",
            "folder": "AllSpider",
            "file": "research_paper_spider.py"
        }
    }

    while True:
        clear_screen()
        print_header()
        print("\n请选择要运行的爬虫任务：\n")
        
        for key, info in spiders.items():
            print(f"  [{key}] {info['name']}")
            
        print("\n  [0] 退出系统")
        print("-" * 50)
        
        choice = input("请输入选项编号: ").strip()
        
        if choice == '0':
            print("\n👋 感谢使用，再见！")
            break
            
        if choice in spiders:
            target = spiders[choice]
            script_path = get_script_path(target['folder'], target['file'])
            run_spider(script_path, target['name'])
        else:
            print("\n❌ 无效的选项，请重新输入。")
            time.sleep(1)

if __name__ == "__main__":
    main()