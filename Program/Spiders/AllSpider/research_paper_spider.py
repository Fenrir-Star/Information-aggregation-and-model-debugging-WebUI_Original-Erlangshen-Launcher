import requests
import json, ssl, time, math, warnings, os, re
from urllib.parse import unquote, urljoin
from pyquery import PyQuery as pq
import pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore")


def save_2_csv(df, file_path, check_keys=[], str_col=[]):
    # 保存至csv
    """
    将数据添加到对应的文件中，原文件中有内容则追加在后面。
    使用 utf-8-sig 编码，确保中文在 Excel 中打开时不乱码。
    """
    try:
        if type(df) == dict:
            df = pd.DataFrame(df, index=[0])
        # 确保目录存在
        save_dir = get_file_path(file_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not os.path.exists(file_path):
            # 首次创建文件
            for col in str_col:
                df[col] = df[col].astype(str)

            if len(check_keys) > 0:
                df.drop_duplicates(check_keys,
                                   keep='first',
                                   inplace=True)
            # 第一次写入，包含列头 (header=True)，使用 utf-8-sig 编码
            df.to_csv(file_path, index=False, encoding='utf-8-sig', header=True)
        else:
            # 文件已存在，读取原有数据进行合并
            row_df = pd.read_csv(file_path, encoding='utf-8-sig')
            for col in str_col:
                row_df[col] = row_df[col].astype(str)
            for col in str_col:
                df[col] = df[col].astype(str)
            
            has_row_num = row_df.shape[0]
            row_num = has_row_num + df.shape[0]

            final_df = pd.concat([row_df, df], ignore_index=True)
            if len(check_keys) > 0:
                final_df.drop_duplicates(check_keys,
                                         keep='first',
                                         inplace=True)
            # 重新写入整个文件，包含列头 (header=True)
            final_df.to_csv(file_path, index=False, encoding='utf-8-sig', header=True)
    except Exception as e:
        time.sleep(5)
        print(f'程序正在保存内容，请关闭CSV文件{file_path}！！！')
        save_2_csv(df, file_path, check_keys, str_col)


def get_file_path(full_path):
    # 根据文件路径，获取所在目录
    """
    根据文件名称获取目录
    :param full_path:
    :return:
    """
    return full_path[0:full_path.rfind(os.path.sep) + 1]


class DzSpider(object):
    def __init__(self, search_word, total_pages):
        # 过程数据缓存
        self.data = {}
        # 数据存储目录修改为 Information/Literature
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        base_dir = os.path.dirname(script_dir)
        self.folder = os.path.join(base_dir, 'Information', 'Literature')
        # 数据库
        self.db_file = fr'{self.folder}\data.db'
        # 开始页码
        self.start_page = 1
        # 结束页码，使用传入的参数
        self.end_page = total_pages
        # 采集数量标识
        self.spider_num = 1
        # 页码大小
        self.page_size = 20
        # 是否已结束
        self.has_finish = False
        # 是否自动获取页码
        # 如果 total_pages 是 None，说明需要自动获取总页数，否则使用传入的值
        self.reset_end_page = True if total_pages is None else False
        # 搜索关键词
        self.search_word = search_word
        # 保存的文件名（根据关键词动态生成），已修改为 .csv
        self.output_filename = f'{self.search_word}.csv'

        # 请求头信息
        self.headers_str = '''
        Accept: application/json, text/javascript, */*; q=0.01
		Accept-Language: zh-CN,zh;q=0.9,en;q=0.8
		Cache-Control: no-cache
		Connection: keep-alive
		Content-Type: application/x-www-form-urlencoded; charset=UTF-8
		Origin: https://cje.ustb.edu.cn
		Pragma: no-cache
		Referer: https://cje.ustb.edu.cn/search
		Sec-Fetch-Dest: empty
		Sec-Fetch-Mode: cors
		Sec-Fetch-Site: same-origin
		sec-ch-ua: "Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"
		sec-ch-ua-mobile: ?0
		sec-ch-ua-platform: "Windows"
		User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36
		X-Requested-With: XMLHttpRequest
		Cookie: JSESSIONID=6BCD7774406051C090EE61879FC5E5C1; Hm_lvt_d166eac910ebb55d72749ba6226ab99e=1763616921; Hm_lpvt_d166eac910ebb55d72749ba6226ab99e=1763616921; HMACCOUNT=5DB1396920E78934; _sowise_user_sessionid_=c99bb6ea-0eab-47a3-ba94-b7f5afe8c8e1
        '''
        self.headers = dict(
            [[y.strip() for y in x.strip().split(':', 1)] for x in self.headers_str.strip().split('\n') if x.strip()])

    def run_task(self):
        # 遍历所有的页码，采集数据
        page_index = self.start_page

        # 如果 total_pages 未指定 (即 self.end_page 是 None)，则在第一次请求后获取总页数
        if self.end_page is None:
             # 设置一个足够大的初始值，确保能执行第一次请求
            self.end_page = 1000
            self.reset_end_page = True

        self.has_finish = False
        while page_index < self.end_page + 1 and not self.has_finish:
            self.get_one_page(page_index)
            page_index += 1
            time.sleep(3)

    def get_one_page(self, page_index):
        req_url = 'https://cje.ustb.edu.cn/data/search/searchResult'
        data = {
            "searchWord": self.search_word, # 使用传入的关键词
            "searchField": "",
            "searchType": "0",
            "orderBy": "",
            "otherConditions": "",
            "analyzeWord": self.search_word, # 使用传入的关键词
            "actualSearchField": "",
            "language": "cn",
            "maxresult": self.page_size,
            "currentpage": page_index,
            "journalId": "ff007540-a7c7-4752-b593-efa08309babb"
        }

        params = {}
        res = requests.post(req_url, headers=self.headers, data=data, params=params, verify=False).json()

        # 如果是第一次请求且需要自动获取总页数
        if self.reset_end_page:
            total_records = res.get('data').get('pagerFilter').get('totalrecord')
            self.end_page = math.ceil(total_records / self.page_size)
            print(f"\n--- 爬虫信息 ---")
            print(f"检测到总记录数: {total_records}，总页数设置为: {self.end_page}")
            print(f"------------------\n")
            self.reset_end_page = False

        df_page = pd.DataFrame()
        for item in res.get('data').get('pagerFilter').get('records'):
            one_data = {
                "标题": item.get('titleCn'),
                "作者": ",".join([a.get('authorNameCn') for a in item.get('authors')]),
                "摘要": item.get('abstractinfoCn'),
                "关键词": ",".join([a.get('keywordCn') for a in item.get('keywords')])
            }
            for k,v in one_data.items():
                one_data[k] = remove_html_el(v)
            print(f'========================================\n'
                  f'第{page_index}/{self.end_page}页，第{self.spider_num}条\n'
                  f'标题: {one_data["标题"]}')
            df_one = pd.DataFrame(one_data, index=[0])
            df_page = pd.concat([df_page, df_one])
            self.spider_num += 1


        file_path = os.path.join(self.folder, self.output_filename)
        if df_page.shape[0] > 0:

            save_2_csv(df_page, file_path)


def remove_html_el(s):
    return pq(f'<div>{s}</div>').text().replace('\n','')


if __name__ == '__main__':
    # 交互式输入逻辑
    print("--- 文献爬虫启动 ---")

    # 获取搜索关键词
    search_word = input("请输入要搜索的关键词 (直接回车默认: 卷积): ").strip()
    if not search_word:
        search_word = "卷积"

    # 获取页数 (可选)
    pages_input = input(f"请输入要爬取的总页数 (直接回车默认: 2): ").strip()

    total_pages = 2  # 默认值设置为 2
    if pages_input:
        try:
            total_pages = int(pages_input)
            if total_pages <= 0:
                print("页数必须是正整数，已使用默认值 2。")
                total_pages = 2
        except ValueError:
            print("页数输入无效，已使用默认值 2。")
            total_pages = 2
    
    print(f"\n开始爬取：关键词='{search_word}'，总页数='{total_pages}'")

    # 实例化爬虫并运行
    if search_word:
        spider = DzSpider(search_word=search_word, total_pages=total_pages)
        spider.run_task()
    else:
        print("关键词不能为空，程序退出。")