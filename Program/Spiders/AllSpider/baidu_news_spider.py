import requests
from datetime import datetime, timedelta
from lxml import etree
import csv
import os
import time
import random
import sys
import io


# 设置标准输出编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def get_news_folder():
    """获取新闻数据保存目录
    
    若目录不存在则自动创建，确保数据有正确的保存位置。
    
    Returns:
        str: 新闻保存目录路径
    """
    # 获取当前脚本路径
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    base_dir = os.path.dirname(script_dir)
    news_folder = os.path.join(base_dir, "Information", "News")
    
    # 确保目录存在
    if not os.path.exists(news_folder):
        os.makedirs(news_folder)
        print(f"✅ 已创建News文件夹: {news_folder}")
    
    return news_folder


def parse_time(unformated_time):
    """解析相对时间为标准格式
    
    将"X分钟前"、"X小时前"等相对时间转换为绝对时间字符串。
    
    Args:
        unformated_time (str): 原始时间字符串
        
    Returns:
        str: 格式化后的时间字符串（YYYY-MM-DD HH:MM）
    """
    if '分钟' in unformated_time:
        minute = unformated_time[:unformated_time.find('分钟')]
        minute = timedelta(minutes=int(minute))
        return (datetime.now() - minute).strftime('%Y-%m-%d %H:%M')
    elif '小时' in unformated_time:
        hour = unformated_time[:unformated_time.find('小时')]
        hour = timedelta(hours=int(hour))
        return (datetime.now() - hour).strftime('%Y-%m-%d %H:%M')
    else:
        return unformated_time


def deal_html(html, file_name):
    """处理HTML页面，提取新闻信息并保存
    
    Args:
        html (lxml.etree._Element): 解析后的HTML对象
        file_name (str): 保存文件路径
    """
    # 尝试多种选择器匹配新闻结果（应对页面结构变化）
    results = []
    results = html.xpath('//div[contains(@class, "result-op") and contains(@class, "c-container")]')
    
    if not results:
        results = html.xpath('//div[@class="result-op c-container xpath-log new-pmd"]')
    
    if not results:
        results = html.xpath('//div[contains(@class, "news-result")]')
    
    if not results:
        results = html.xpath('//div[@class="result"]')
    
    print(f"📊 找到 {len(results)} 条新闻结果")
    
    save_data = []

    for i, result in enumerate(results):
        try:
            # 提取标题
            title_elements = (result.xpath('.//h3/a') or 
                             result.xpath('.//h3') or 
                             result.xpath('.//a[@class="news-title"]'))
            if not title_elements:
                continue
                
            title = title_elements[0].xpath('string(.)').strip()

            # 提取摘要
            summary_elements = (result.xpath('.//span[@class="c-font-normal c-color-text"]') or 
                              result.xpath('.//div[contains(@class, "c-span-last")]') or 
                              result.xpath('.//div[contains(@class, "c-gap-top-xsmall")]'))
            summary = summary_elements[0].xpath('string(.)').strip() if summary_elements else ""

            # 提取来源和时间
            source = ""
            date_time = ""
            
            # 尝试多种信息选择器
            info_elements = (result.xpath('.//div[contains(@class, "news-source")]') or 
                           result.xpath('.//span[contains(@class, "c-color-gray")]') or 
                           result.xpath('.//div[contains(@class, "c-span-last")]//span'))
            
            if info_elements:
                info_text = info_elements[0].xpath('string(.)').strip()
                # 分离来源和时间
                if '·' in info_text:
                    parts = info_text.split('·')
                    if len(parts) >= 2:
                        source = parts[0].strip()
                        date_time = parse_time(parts[1].strip())
                else:
                    source = info_text
                    date_time = "未知时间"
            
            # 单独提取时间（兜底）
            if not date_time:
                time_elements = result.xpath('.//span[@class="c-color-gray2 c-font-normal c-gap-right-xsmall"]/text()')
                if time_elements:
                    date_time = parse_time(time_elements[0])
                else:
                    date_time = "未知时间"

            print(f'第{i+1}条新闻:')
            print(f'标题: {title}')
            print(f'来源: {source}')
            print(f'时间: {date_time}')
            print(f'概要: {summary}')
            print('-' * 50)

            save_data.append({
                'title': title,
                'source': source,
                'time': date_time,
                'summary': summary
            })
            
        except Exception as e:
            print(f"❌ 解析第{i+1}条新闻时出错: {e}")
            continue
    
    # 写入CSV文件
    if save_data:
        file_exists = os.path.exists(file_name)
        with open(file_name, 'a+', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            # 写入表头（仅首次）
            if not file_exists or os.path.getsize(file_name) == 0:
                writer.writerow(['标题', '来源', '时间', '概要'])
            for row in save_data:
                writer.writerow([row['title'], row['source'], row['time'], row['summary']])
        print(f"✅ 成功保存 {len(save_data)} 条新闻到文件")
    else:
        print("❌ 没有提取到任何新闻数据")


# 请求配置
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.baidu.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache'
}

URL = 'https://www.baidu.com/s'

PARAMS = {
    'ie': 'utf-8',
    'medium': 0,
    # rtt=4 按时间排序；rtt=1 按焦点排序
    'rtt': 1,
    'bsst': 1,
    'rsv_dl': 'news_t_sk',
    'cl': 2,
    'tn': 'news',
    'rsv_bp': 1,
    'oq': '',
    'rsv_btype': 't',
    'f': 8,
    'wd': ''  # 搜索关键词（后续填充）
}


def do_spider(keyword, sort_by='focus'):
    """执行百度新闻爬虫
    
    Args:
        keyword (str): 搜索关键词
        sort_by (str): 排序方式，'focus'（按焦点）或'time'（按时间），默认'focus'
    """
    # 获取保存目录
    news_folder = get_news_folder()
    file_name = os.path.join(news_folder, f'{keyword}.csv')

    print(f"📁 文件将保存到: {file_name}")

    # 配置请求参数
    PARAMS['wd'] = keyword
    PARAMS['rtt'] = 4 if sort_by == 'time' else 1

    try:
        print(f"🔍 开始请求百度新闻，关键词: {keyword}")
        response = requests.get(url=URL, params=PARAMS, headers=HEADERS, timeout=10)
        response.encoding = 'utf-8'
        
        # 保存网页用于调试
        debug_html_path = os.path.join(news_folder, 'debug_baidu_news.html')
        with open(debug_html_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"💾 网页已保存到: {debug_html_path}")
        
        html = etree.HTML(response.text)
        
        # 检查是否有结果
        no_results = html.xpath('//div[contains(text(), "没有找到")]') or html.xpath('//div[contains(text(), "未找到")]')
        if no_results:
            print("❌ 百度返回：没有找到相关新闻")
            return
        
        deal_html(html, file_name)

        # 尝试获取总页数并爬取多页
        try:
            total_element = (html.xpath('//div[@id="header_top_bar"]/span/text()') or 
                           html.xpath('//span[@class="nums_text"]/text()') or 
                           html.xpath('//div[contains(@class, "nums")]//text()'))
            
            if total_element:
                total_text = total_element[0] if total_element else ""
                print(f"总结果信息: {total_text}")
                
                # 提取总条数
                import re
                numbers = re.findall(r'\d+', total_text.replace(',', ''))
                if numbers:
                    total = int(numbers[0])
                    page_num = min(total // 10, 5)  # 限制最多5页
                
                    print(f"📄 总共约 {total} 条结果，计划爬取 {page_num} 页")
                    
                    for page in range(1, page_num + 1):
                        print(f'\n第 {page} 页\n')
                        HEADERS['Referer'] = response.url
                        PARAMS['pn'] = page * 10

                        response = requests.get(url=URL, headers=HEADERS, params=PARAMS, timeout=10)
                        response.encoding = 'utf-8'
                        
                        html = etree.HTML(response.text)
                        deal_html(html, file_name)

                        time.sleep(random.randint(2, 4))
                else:
                    print("无法解析总页数，只爬取第一页")
            else:
                print("未找到总页数信息，只爬取第一页")
                
        except Exception as e:
            print(f"❌ 分页爬取出错: {e}，继续处理第一页数据")
            
    except requests.RequestException as e:
        print(f"❌ 网络请求出错: {e}")
    except Exception as e:
        print(f"❌ 爬取过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """百度新闻爬虫主入口"""
    try:
        # 用户输入配置
        keyword = input("请输入搜索关键词（默认：特朗普）: ").strip()
        if not keyword:
            keyword = '特朗普'
        
        sort_option = input("请选择排序方式：1-按焦点排序，2-按时间排序（默认：1）: ").strip()
        sort_by = 'focus' if sort_option != '2' else 'time'
        
        print(f"🔍 开始爬取关键词: {keyword}")
        print(f"📊 排序方式: {'按焦点排序' if sort_by == 'focus' else '按时间排序'}")
        
        do_spider(keyword=keyword, sort_by=sort_by)
        
        # 显示保存结果
        news_folder = get_news_folder()
        file_path = os.path.join(news_folder, f"{keyword}.csv")
        print(f"✅ 百度新闻爬取完成！")
        print(f"💾 文件已保存至: {file_path}")
        
        # 简单验证文件内容
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
                lines = content.strip().split('\n')
                print(f"📊 文件包含 {len(lines)} 行数据")
                if lines:
                    print("前几行内容:")
                    for i, line in enumerate(lines[:3]):
                        print(f"{i+1}: {line}")
    except Exception as e:
        print(f"❌ 爬取过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def run():
    """运行入口（兼容外部调用）"""
    main()


if __name__ == "__main__":
    main()