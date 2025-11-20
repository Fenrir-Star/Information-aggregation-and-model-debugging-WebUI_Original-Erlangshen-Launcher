import os
import time
import csv
import random
from DrissionPage import ChromiumPage, ChromiumOptions


class LenovoSpider:
    """联想应用商店爬虫类
    
    用于爬取联想应用商店中指定关键词的应用信息，包括应用名称、简介和详情链接，
    并将结果保存到CSV文件中。
    """
    
    def __init__(self):
        """初始化爬虫实例，配置浏览器选项并创建页面对象"""
        co = ChromiumOptions()

        # 配置浏览器路径
        browser_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

        if os.path.exists(browser_path):
            co.set_paths(browser_path=browser_path)
        else:
            print(f"⚠️ 警告: 路径 {browser_path} 不存在，尝试使用默认配置...")

        # 浏览器参数配置
        co.set_argument('--disable-blink-features=AutomationControlled')
        co.set_argument('--no-sandbox')

        self.page = ChromiumPage(co)
        self.app_folder = self.init_folder()

    def init_folder(self):
        """初始化应用数据保存目录
        
        尝试创建指定的保存目录，若失败则回退到脚本所在目录创建默认文件夹。
        
        Returns:
            str: 实际的保存目录路径
        """
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        base_dir = os.path.dirname(script_dir)
        folder = os.path.join(base_dir, "Information", "AppDescriptions")

        # 尝试创建目录
        if not os.path.exists(folder):
            try:
                os.makedirs(folder, exist_ok=True)
                print(f"✅ 已创建目录: {folder}")
            except Exception as e:
                print(f"❌ 创建目录失败: {e}")
                # 回退到脚本所在目录
                script_dir = os.path.dirname(os.path.abspath(__file__))
                folder = os.path.join(script_dir, "Lenovo_Apps")
                os.makedirs(folder, exist_ok=True)
                print(f"⚠️ 已回退到默认目录: {folder}")

        print(f"\n📂 数据保存目录: {folder}")
        return folder

    def save_to_csv(self, data, filename):
        """将应用信息保存到CSV文件
        
        Args:
            data (dict): 包含应用信息的字典，需包含'name'、'desc'、'url'键
            filename (str): 保存的文件名（不含扩展名）
        """
        filepath = os.path.join(self.app_folder, f"{filename}.csv")
        file_exists = os.path.exists(filepath)

        try:
            with open(filepath, 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                # 写入表头（仅首次）
                if not file_exists:
                    writer.writerow(['应用名称', '应用简介', '详情链接'])

                writer.writerow([
                    data.get('name', '未知'),
                    data.get('desc', ''),
                    data.get('url', '')
                ])
            print(f"✅ 成功保存: {data.get('name')}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")

    def crawl_lenovo(self, keyword, max_apps=5):
        """爬取联想应用商店中指定关键词的应用
        
        Args:
            keyword (str): 搜索关键词
            max_apps (int): 最大爬取应用数量，默认5
        """
        print(f"🚀 开始搜索: {keyword}")

        search_url = f"https://lestore.lenovo.com/search?k={keyword}"
        self.page.get(search_url)
        time.sleep(3)

        # 滚动页面加载更多内容
        self.page.scroll.to_bottom()
        time.sleep(1)

        print("🔍 提取应用链接...")
        link_elements = self.page.eles('xpath://a[contains(@href, "/detail/")]')

        # 去重处理应用链接
        unique_urls = set()
        for ele in link_elements:
            if ele.link:
                unique_urls.add(ele.link)

        target_urls = list(unique_urls)[:max_apps]

        if not target_urls:
            print("❌ 未找到任何应用链接。")
            return

        print(f"📊 找到 {len(target_urls)} 个应用，开始抓取...")

        # 逐个处理应用详情页
        count = 0
        for url in target_urls:
            count += 1
            print(f"\n[{count}/{len(target_urls)}] 正在处理: {url}")
            self.parse_detail_page(url, keyword)
            time.sleep(1)

    def parse_detail_page(self, url, keyword):
        """解析应用详情页，提取应用信息
        
        Args:
            url (str): 应用详情页链接
            keyword (str): 搜索关键词（用于保存文件）
        """
        tab = None
        try:
            tab = self.page.new_tab(url)

            # 等待页面加载完成
            start_time = time.time()
            while time.time() - start_time < 10:
                if tab.title and "联想" in tab.title:
                    break
                time.sleep(0.5)

            current_title = tab.title
            if not current_title or "404" in current_title:
                print(f"⚠️ 页面无效，跳过: {url}")
                return

            # 提取应用名称
            name = "未知应用"
            # 策略A: 尝试从h1标签提取
            try:
                h1 = tab.ele('tag:h1', timeout=2)
                if h1: 
                    name = h1.text
            except:
                pass

            # 策略B (兜底): 从网页标题提取
            if name == "未知应用" and "-" in current_title:
                name = current_title.split("-")[0].strip()
                print(f"💡 从标题提取到名称: {name}")

            # 提取应用简介
            desc = "暂无简介"
            try:
                # 策略A: 从meta标签提取
                meta_desc = tab.ele('xpath://meta[@name="description"][2]', timeout=2)
                if meta_desc:
                    desc = meta_desc.attr('content').strip("<p>")

                # 策略B: 从页面元素提取
                if not desc or len(desc) < 5:
                    desc_ele = tab.ele('css:.detail-description')
                    if desc_ele: 
                        desc = desc_ele.text
            except:
                pass

            # 整理数据并保存
            data = {
                "name": name,
                "desc": desc[:150].replace('\n', ' '),  # 清洗换行符
                "url": url
            }

            self.save_to_csv(data, keyword)

        except Exception as e:
            print(f"❌ 处理异常: {e}")
        finally:
            if tab:
                tab.close()

    def close(self):
        """关闭浏览器页面"""
        self.page.quit()


if __name__ == "__main__":
    spider = None
    try:
        spider = LenovoSpider()

        # 用户输入配置
        kw = input("请输入关键词（默认：电脑管家）: ").strip() or "电脑管家"
        limit = input("请输入爬取数量（默认：2）: ").strip()
        limit = int(limit) if limit.isdigit() else 2

        spider.crawl_lenovo(kw, max_apps=limit)
        print(f"\n✨ 全部任务完成！")

    except Exception as e:
        print(f"❌ 程序异常: {e}")
    finally:
        if spider:
            spider.close()