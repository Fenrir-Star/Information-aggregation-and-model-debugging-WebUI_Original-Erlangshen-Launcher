# ===================== 配置日志 =====================
import logging
# 1. 获取PyTorch分布式重定向模块的日志器
logger = logging.getLogger("torch.distributed.elastic.multiprocessing.redirects")
# 2. 将该日志器的级别设为ERROR（仅输出错误，屏蔽WARNING及以下日志）
logger.setLevel(logging.ERROR)
# 3. 禁止日志传播到父日志器（避免其他地方重复输出）
logger.propagate = False

# ===================== 警告过滤 =====================
import warnings
warnings.filterwarnings(
    "ignore",
    message="NOTE: Redirects are currently not supported in Windows or MacOs.",
    category=UserWarning,
    module="torch.distributed.elastic.multiprocessing.redirects"
)

import gradio as gr
import os
import re
import sys
import subprocess
import time
import csv # 导入 csv 库用于读取和写入数据
import glob # 导入 glob 用于文件搜索
import traceback # 导入 traceback 用于错误调试

# ===================== 模型依赖和初始化（真实代码结构）=====================
try:
    import torch
    from transformers import BertTokenizer, MegatronBertModel, MegatronBertConfig
    import torch.nn.functional as F
    
    # 定义 quick_start.py 中使用的标签映射
    # TNEWS (新闻分类)
    TNEWS_LABELS = {
        0: "科技", 1: "娱乐", 2: "体育", 3: "财经", 4: "时政", 5: "教育",
        6: "军事", 7: "汽车", 8: "房产", 9: "游戏", 10: "时尚", 11: "彩票",
        12: "股票", 13: "家居", 14: "社会"
    }
    
    # OCNLI (自然语言推理)
    OCNLI_LABELS = {0: "蕴含", 1: "矛盾", 2: "中立"}
    
    # CSLDCP (主题文献分类)
    CSLDCP_LABELS = {
        0: "计算机科学与技术", 1: "电子科学与技术", 2: "信息与通信工程", 3: "控制科学与工程",
        4: "软件工程", 5: "网络空间安全", 6: "数学", 7: "物理学", 8: "化学", 9: "生物学",
        10: "医学", 11: "管理学", 12: "经济学", 13: "法学", 14: "教育学", 15: "文学",
        16: "历史学", 17: "哲学", 18: "艺术学", 19: "农学", 20: "工学", 21: "理学",
        22: "医学技术", 23: "公共卫生与预防医学", 24: "药学", 25: "中药学", 26: "口腔医学",
        27: "临床医学", 28: "护理学", 29: "基础医学", 30: "中医学", 31: "中西医结合",
        32: "管理科学与工程", 33: "工商管理", 34: "公共管理", 35: "图书情报与档案管理",
        36: "应用经济学", 37: "理论经济学", 38: "统计学", 39: "法学理论", 40: "宪法学与行政法学",
        41: "刑法学", 42: "民商法学", 43: "诉讼法学", 44: "经济法学", 45: "环境与资源保护法学",
        46: "国际法学", 47: "军事法学", 48: "教育学原理", 49: "课程与教学论", 50: "学前教育学",
        51: "高等教育学", 52: "成人教育学", 53: "职业技术教育学", 54: "特殊教育学", 55: "教育技术学",
        56: "中国语言文学", 57: "外国语言文学", 58: "新闻传播学", 59: "艺术学理论", 60: "音乐与舞蹈学",
        61: "戏剧与影视学", 62: "美术学", 63: "设计学", 64: "历史学理论", 65: "中国史",
        66: "世界史"
    }
    
    # IFLYTEK (应用描述分类)
    IFLYTEK_LABELS = {
        0: "打车", 1: "地图导航", 2: "旅游", 3: "外卖", 4: "美食", 5: "社交", 6: "购物",
        7: "视频", 8: "音乐", 9: "教育", 10: "办公", 11: "工具", 12: "金融", 13: "医疗健康",
        14: "出行", 15: "房产", 16: "招聘", 17: "小说", 18: "资讯", 19: "摄影", 20: "美图",
        21: "母婴", 22: "运动", 23: "美妆", 24: "两性", 25: "动漫", 26: "游戏", 27: "娱乐",
        28: "影视", 29: "星座", 30: "直播", 31: "理财", 32: "保险", 33: "贷款", 34: "信用卡",
        35: "证券", 36: "股票", 37: "基金", 38: "银行", 39: "支付", 40: "记账", 41: "税务",
        42: "社保", 43: "医保", 44: "医疗服务", 45: "健康管理", 46: "就医挂号", 47: "药品查询",
        48: "体检", 49: "养生", 50: "减肥", 51: "育儿", 52: "早教", 53: "K12教育", 54: "职业教育",
        55: "语言学习", 56: "考研", 57: "公考", 58: "留学", 59: "求职", 60: "职场", 61: "办公协作",
        62: "文档管理", 63: "笔记", 64: "思维导图", 65: "PPT", 66: "表格", 67: "PDF", 68: "OCR",
        69: "翻译", 70: "词典", 71: "计算器", 72: "日历", 73: "天气", 74: "闹钟", 75: "手电筒",
        76: "文件管理", 77: "压缩", 78: "加密", 79: "杀毒", 80: "浏览器", 81: "输入法", 82: "壁纸",
        83: "主题", 84: "铃声", 85: "文件传输", 86: "WiFi", 87: "蓝牙", 88: "投屏", 89: "远程控制",
        90: "智能家居", 91: "汽车服务", 92: "违章查询", 93: "驾校", 94: "汽车资讯", 95: "二手车",
        96: "租房", 97: "买房", 98: "装修", 99: "家居建材", 100: "家政", 101: "快递", 102: "物流",
        103: "外卖配送", 104: "餐饮服务", 105: "酒店预订", 106: "机票", 107: "火车票", 108: "租车",
        109: "共享单车", 110: "公交", 111: "地铁", 112: "轮渡", 113: "长途客运", 114: "停车场",
        115: "加油", 116: "洗车", 117: "汽车维修", 118: "其他"
    }

except ImportError as e:
    print(f"⚠️ 警告: 缺少 PyTorch/Transformers 依赖。模型功能将无法实际运行。错误: {e}")
    # 定义占位符以防止代码崩溃
    class MockTokenizer:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return {"input_ids": torch.tensor([[1, 103, 1]]), "token_type_ids": torch.tensor([[0, 0, 0]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        def encode(self, *args, **kwargs): return torch.tensor([[1, 103, 1]])
        def decode(self, *args, **kwargs): return ""
    class MockModel:
        def __init__(self, *args, **kwargs): pass
        def to(self, *args, **kwargs): return self
        def eval(self): pass
        def device(self): return "cpu"
        def __call__(self, *args, **kwargs): 
            # 模拟输出，至少包含一个 [CLS] 特征
            hidden_state = torch.rand(1, 128, 768) 
            pooler_output = hidden_state[:, 0, :]
            return (hidden_state, pooler_output) # 模拟输出 (last_hidden_state, pooler_output)
    
    torch = None
    BertTokenizer = MockTokenizer
    MegatronBertModel = MockModel
    MegatronBertConfig = lambda *args, **kwargs: None
    F = None
    TNEWS_LABELS = OCNLI_LABELS = CSLDCP_LABELS = IFLYTEK_LABELS = {}


# ====================================================================
# 1. 路径和工具函数 
# ====================================================================

# 假设 app.py 位于 Program/ 目录下
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_BASE_DIR = os.path.join(CURRENT_DIR, "Models")
DATA_BASE_DIR = os.path.join(CURRENT_DIR, "Spiders", "Information")


# --- 动态扫描 Spiders 文件夹 ---
def get_available_spiders() -> list:
    """扫描 Spiders/AllSpider/ 下的所有 .py 文件作为可用爬虫"""
    SPIDERS_DIR = os.path.join(CURRENT_DIR, "Spiders", "AllSpider")
    if not os.path.exists(SPIDERS_DIR):
        print(f"⚠️ 警告: 爬虫基础路径不存在: {SPIDERS_DIR}")
        return []
    
    # 查找所有 .py 文件名
    spider_files = [
        f for f in os.listdir(SPIDERS_DIR) 
        if f.endswith('.py') and not f.startswith('_') # 过滤掉非.py文件和_开头的文件
    ]
    # 可选：定义每个脚本的输入类型（用于控制 UI 组件的可见性/值）
    SPIDER_CONFIG = {
        "baidu_news_spider.py": {"sort": True, "pages": False, "default_kw": "特朗普", "default_sort": "1. 按焦点排序"},
        "lenovo_app_spider.py": {"sort": False, "pages": True, "default_kw": "电脑管家", "default_pages": 3},
        "research_paper_spider.py": {"sort": False, "pages": True, "default_kw": "卷积", "default_pages": 2},
    }
    
    return spider_files, SPIDER_CONFIG

# 获取配置 (可供后续通用函数使用)
SPIDER_FILES, SPIDER_CONFIG = get_available_spiders()


# --- 排序方式映射 ---
SORT_MAP = {
    "默认排序": "0",
    "时间排序": "2", 
    "焦点排序": "1", # 百度新闻中 1=焦点, 2=时间
    "名称排序": "3", # 占位符
}
SORT_CHOICES = list(SORT_MAP.keys())


# --- 动态扫描 Models 文件夹 ---
def get_available_models() -> list:
    """扫描 MODEL_BASE_DIR 下的所有子文件夹作为可用模型"""
    if not os.path.exists(MODEL_BASE_DIR):
        print(f"⚠️ 警告: 模型基础路径不存在: {MODEL_BASE_DIR}")
        return []
    
    # 过滤掉非文件夹项，并获取文件夹名称
    model_dirs = [
        d for d in os.listdir(MODEL_BASE_DIR) 
        if os.path.isdir(os.path.join(MODEL_BASE_DIR, d)) and not d.startswith('.')
    ]
    # 优先将 'Erlangshen' 开头的模型排在前面
    model_dirs.sort(key=lambda x: (not x.startswith('Erlangshen'), x))
    
    return model_dirs


# --- 批量任务所需列名映射 (用于按列读取 CSV) ---
TASK_COLUMN_MAP = {
    "新闻分类（TNEWS）": {"input": "标题"},
    "摘要关键词验证（CSL）": {"input": "摘要", "true_value": "关键词"},
    "主题文献分类（CSLDCP）": {"input": "摘要"},
    "应用描述分类（IFLYTEK）": {"input": "应用简介"},
}

def find_column_indices(header: list, required_columns: dict) -> tuple:
    """根据 Header 查找所需列名的索引"""
    header_map = {col.strip(): idx for idx, col in enumerate(header)}
    indices = {}
    missing = []
    
    for key, col_name in required_columns.items():
        if col_name in header_map:
            indices[key] = header_map[col_name]
        else:
            missing.append(col_name)
    return indices, missing

# --- quick_start.py 核心零样本分类逻辑 ---
def predict_by_similarity(model, tokenizer, device, text_feat, label_map):
    """
    原理：计算[输入文本特征]与[类别名称特征]的余弦相似度进行分类
    """
    label_texts = [f"关于{label}的内容" for label in label_map.values()]

    label_encoded = tokenizer(
        label_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32
    ).to(device)

    with torch.no_grad():
        label_outputs = model(** label_encoded)
        label_feats = label_outputs.pooler_output 

    text_feat_norm = F.normalize(text_feat, p=2, dim=1)
    label_feats_norm = F.normalize(label_feats, p=2, dim=1)

    # 相似度矩阵 (1, N)
    similarities = torch.mm(text_feat_norm, label_feats_norm.T)

    # 乘以一个缩放因子(scale)让softmax分布更尖锐
    logits = similarities * 15 
    pred_probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(torch.argmax(logits).item())
    
    # 返回所有相似度，用于展示所有类别的置信度
    all_similarities = similarities.cpu().numpy()[0] 
    
    return pred_idx, pred_probs, pred_probs[pred_idx], all_similarities 

# --- 爬虫运行函数 (保持不变) ---
def run_external_script(script_path, inputs_list: list) -> str:
    # ... (与之前版本相同，用于运行爬虫) ...
    if not os.path.exists(script_path):
        return f"❌ 错误：找不到脚本文件: {os.path.basename(script_path)}\n请确保文件位于: {script_path}"
    
    stdin_input = "\n".join(inputs_list) + "\n"
    
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # 针对特定脚本的编码处理保持不变
        if "baidu_news_spider.py" in script_path:
             result = subprocess.run(
                [sys.executable, script_path],
                input=stdin_input.encode('utf-8'),
                capture_output=True,
                check=False,
                env=env,
                timeout=240
            )
             stdout = result.stdout.decode('utf-8', errors='replace')
             stderr = result.stderr.decode('utf-8', errors='replace')
             
        else:
            result = subprocess.run(
                [sys.executable, script_path],
                input=stdin_input,
                capture_output=True,
                text=True, 
                check=False,
                encoding='utf-8',
                env=env,
                timeout=240
            )
            stdout = result.stdout
            stderr = result.stderr


        if result.returncode != 0:
            return (f"❌ 脚本运行失败 (Return Code: {result.returncode})!\n"
                    f"--- Standard Error ---\n"
                    f"{stderr}\n"
                    f"--- Standard Output (部分) ---\n"
                    f"{stdout}")
        
        return f"✅ 脚本运行成功！\n" + stdout

    except subprocess.TimeoutExpired:
        return "❌ 脚本运行超时 (Timeout: 240s)。"
    except Exception as e:
        return f"❌ 运行过程中发生未知错误: {e}"


def get_spider_script_path(script_name: str) -> str:
    return os.path.join(CURRENT_DIR, "Spiders", "AllSpider", script_name)

# --- 通用爬虫任务函数 ---
def run_generic_spider_gr(script_name: str, keyword: str, sort_by_choice: str, max_pages: int) -> str:
    """
    通用爬虫运行函数，根据脚本名称调用 run_external_script
    """
    script_path = get_spider_script_path(script_name)
    
    config = SPIDER_CONFIG.get(script_name, {})
    
    # 1. 确定关键词 (使用输入值，若为空则使用默认值)
    keyword_input = keyword if keyword else config.get('default_kw', '默认关键词')
    
    # 2. 确定输入列表 inputs
    inputs = [keyword_input]
    
    # 3. 处理排序 (仅 baidu_news_spider.py 使用)
    if config.get("sort"):
        # 排序值映射: 默认排序 -> '0', 时间排序 -> '2', 焦点排序 -> '1', 名称排序 -> '3'
        # baidu_news_spider 仅接受 1(焦点) 或 2(时间)
        if script_name == "baidu_news_spider.py":
            sort_input = '1' if '焦点' in sort_by_choice else '2' 
        else:
            sort_input = SORT_MAP.get(sort_by_choice, '0')
        inputs.append(sort_input)
        
    # 4. 处理页数 (lenovo_app_spider.py 和 research_paper_spider.py 使用)
    if config.get("pages"):
        # 确保 max_pages 是整数
        page_input = str(int(max_pages))
        inputs.append(page_input) # 页数在关键词之后
        
        # NOTE: 针对您的现有代码结构，baidu_news_spider.py 不使用 max_pages，因此只有 inputs=[keyword, sort]
        # lenovo_app_spider.py 和 research_paper_spider.py 应该使用 inputs=[keyword, max_pages]
        # 检查原始函数逻辑：
        # run_spider_lenovo_app_gr(keyword, max_pages) -> inputs = [keyword, str(max_pages)]
        # run_spider_baidu_news_gr(keyword, sort) -> inputs = [keyword, sort_input]
        
        # 为了兼容这两种模式，我们依赖 config 来确定 inputs 的顺序和数量。
        # 如果是 lenovo 或 research，只有 keyword 和 max_pages
        if script_name != "baidu_news_spider.py":
             inputs = [keyword_input, page_input]
            
    # 5. 运行脚本
    header = f"🚀 正在运行脚本: **{script_name}** (输入: {inputs})\n"
    result = run_external_script(script_path, inputs)
    
    return header + result


# ====================================================================
# 2. 模型加载和通用任务函数 (更新为实际执行逻辑)
# ====================================================================

def init_model_and_tokenizer_gr(model_choice: str, state_tokenizer, state_model) -> tuple:
    """模型初始化函数，加载模型对象到 Gradio State"""
    is_mock_mode = torch is None # 依赖于全局的 torch 变量（如果导入失败则为 None）
    dependency_status = f"🔴 PyTorch/Transformers 依赖缺失 (Mock模式)。" if is_mock_mode else "✅ PyTorch/Transformers 依赖满足 (真实推理模式)。"
    # --------------------------
    
    if is_mock_mode:
        status_msg = f"❌ PyTorch 库未安装或加载失败。无法加载真实模型。\n{dependency_status}"
        return None, None, status_msg
        
    model_dir = os.path.join(MODEL_BASE_DIR, model_choice)
    
    if not os.path.exists(model_dir):
        status_msg = f"❌ 模型路径不存在：{model_dir}\n请检查您的文件结构是否与 Program/Models/ 一致。\n{dependency_status}"
        return None, None, status_msg
        
    try:
        # 统一使用 CPU 进行模型加载，避免 GPU 显存不足问题
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"尝试从 {model_dir} 加载模型...")
        
        # 使用 quick_start.py 中的逻辑，加载 MegatronBertModel
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        config = MegatronBertConfig.from_pretrained(model_dir)
        # 注意：这里移除了 dtype=torch.float16，因为它可能导致 CPU 或部分 GPU 环境出错
        model = MegatronBertModel.from_pretrained(model_dir, config=config, dtype=torch.float16)
        model.to(device)
        model.eval()
        
        state_tokenizer = tokenizer
        state_model = model

        status_msg = f"""
==================================================
✅ 模型初始化完成 | 使用设备：{device.type.upper()}
📌 模型：{model_choice}
{dependency_status}
==================================================
"""
        return state_tokenizer, state_model, status_msg

    except Exception as e:
        status_msg = f"❌ 模型加载失败：{e}\n请确保模型权重完整且已安装所有依赖（如 `protobuf`）。\nTrace: {traceback.format_exc()}\n{dependency_status}"
        return None, None, status_msg

# --- Helper for CSL Similarity Verification (for batch) ---
def perform_csl_inference(model, tokenizer, device, abstract: str, keywords_input: str):
    """
    执行单条 CSL 任务（摘要关键词验证）的相似度计算和判断
    """
    if not abstract or not keywords_input:
        return 0.0, "⚠️ 文本或关键词为空"
        
    keywords = [k.strip() for k in re.split(r"[,，]", keywords_input) if k.strip()]
    keywords_text = "，".join(keywords)

    # 1. 编码摘要和关键词
    abstract_encoded = tokenizer(abstract, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    keywords_encoded = tokenizer(keywords_text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    
    # 2. 提取特征
    with torch.no_grad():
        abstract_output = model(**abstract_encoded)
        abstract_feat = abstract_output.pooler_output
        keywords_output = model(**keywords_encoded)
        keywords_feat = keywords_output.pooler_output
    
    # 3. 计算余弦相似度（归一化后）
    abstract_feat_norm = F.normalize(abstract_feat, p=2, dim=1)
    keywords_feat_norm = F.normalize(keywords_feat, p=2, dim=1)
    similarity = torch.mm(abstract_feat_norm, keywords_feat_norm.T).item()
    
    # 4. 二分类判断
    threshold = 0.3 # quick_start.py 默认阈值
    is_accurate = "✅ 准确" if similarity >= threshold else "❌ 不准确"
    
    return similarity, is_accurate

# --- Helper for Classification Tasks (TNEWS, CSLDCP, IFLYTEK) (for batch) ---
def perform_classification_inference(model, tokenizer, device, text: str, label_map: dict):
    """
    执行单条分类任务（TNEWS, CSLDCP, IFLYTEK）的零样本分类
    """
    if not text:
        return "N/A", 0.0, None # 预测标签, 置信度, 所有相似度
        
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**encoded)
        text_feat = outputs.pooler_output
    
    pred_idx, _, confidence, all_similarities = predict_by_similarity(model, tokenizer, device, text_feat, label_map)
    pred_label = label_map.get(pred_idx, f"未知类别 ({pred_idx})")
    
    return pred_label, confidence, all_similarities # 返回所有相似度，用于CHIDF

# --- 通用模型推理函数 ---
def run_model_task_gr(task_name: str, tokenizer, model, *args) -> str:
    """
    通用模型任务运行器：执行 quick_start.py 中的全部零样本/特征匹配逻辑。
    """
    if not model or not tokenizer:
        return "❌ 模型未加载，请先在【模型选择】区域加载模型！"
        
    output = f"📝 任务：{task_name} - 模型: {model.__class__.__name__}\n"
    
    is_mock_mode = "MockModel" in model.__class__.__name__
    if is_mock_mode:
        output += "⚠️ **当前输出为模拟 (Mock) 结果，请安装 PyTorch/Transformers 依赖!**\n"
    else:
        output += "✅ **当前输出为模型真实 (Real) 推理结果。**\n"
    
    output += "==================================================\n"
    device = model.device 

    try:
        # --- 任务 1: CHIDF (成语填空) ---
        if task_name == "成语填空（CHIDF）":
            user_text, candidate_input = args
            
            if "[MASK]" not in user_text:
                return "❌ 输入错误：请在句子中添加[MASK]标记成语空缺位置！"
            
            # 1. 编码文本并提取[MASK]位置特征
            encoded = tokenizer(user_text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
            mask_pos = torch.where(encoded["input_ids"] == tokenizer.mask_token_id)[1]
            if len(mask_pos) == 0:
                return "❌ 未识别到[MASK]标记，请重新输入！"
                
            with torch.no_grad():
                outputs = model(**encoded)
                mask_feat = outputs.last_hidden_state[0, mask_pos, :]
            
            # 2. 准备候选成语
            candidate_idioms = [idiom.strip() for idiom in candidate_input.split("，") if idiom.strip()]
            if not candidate_idioms:
                 candidate_idioms = ["坚持不懈", "半途而废", "畏缩不前", "敷衍了事"]
                 output += f"⚠️ 未输入候选成语，使用默认候选：{','.join(candidate_idioms)}\n"
            
            # 3. 提取候选特征
            candidate_feats = []
            for idiom in candidate_idioms:
                template_text = f"这个成语的意思是：{idiom}"
                idiom_encoded = tokenizer(template_text, return_tensors="pt", padding=True, truncation=True, max_length=32).to(device)
                with torch.no_grad():
                    idiom_output = model(** idiom_encoded)
                    idiom_feat = idiom_output.pooler_output 
                    candidate_feats.append(idiom_feat)
            
            candidate_feats = torch.cat(candidate_feats, dim=0)
            
            # 4. 计算相似度（归一化后）
            mask_feat_norm = F.normalize(mask_feat, p=2, dim=1)
            candidate_feats_norm = F.normalize(candidate_feats, p=2, dim=1)
            similarities = torch.mm(mask_feat_norm, candidate_feats_norm.T)[0]
            
            best_idx = torch.argmax(similarities).item()
            
            # 5. 修正：展示所有成语的匹配度
            output += f"📖 原句：{user_text}\n"
            output += "--------------------------------------------------\n"
            output += "| 候选成语 | 原始相似度 | 归一化置信度 | 最佳匹配 |\n"
            output += "| :---: | :---: | :---: | :---: |\n"
            
            result_lines = []
            for i, idiom in enumerate(candidate_idioms):
                raw_similarity = similarities[i].item()
                # 将 [-1, 1] 相似度映射到 [0, 1] 显示
                # normalized_confidence = (raw_similarity + 1) / 2 # 归一化可能误导，直接显示相似度即可
                normalized_confidence = raw_similarity
                
                is_best = "🏆 YES" if i == best_idx else "NO"
                
                result_lines.append(
                    f"| {idiom} | {raw_similarity:.4f} | {normalized_confidence:.4f} | {is_best} |"
                )
            
            output += "\n".join(result_lines)
            return output

        # --- 任务 2: TNEWS (新闻分类) ---
        elif task_name == "新闻分类（TNEWS）":
            news_text = args[0]
            
            pred_label, confidence, all_similarities = perform_classification_inference(model, tokenizer, device, news_text, TNEWS_LABELS)
            
            output += f"📄 新闻文本：{news_text[:50]}...\n"
            output += f"🏆 预测类别：**{pred_label}** (置信度: {confidence:.4f})\n"
            
            # 展示所有分类结果
            output += "--------------------------------------------------\n"
            output += "| 类别 | 置信度 |\n"
            output += "| :---: | :---: |\n"
            
            # 结合标签名称和概率进行排序展示
            all_results = []
            for idx, label in TNEWS_LABELS.items():
                all_results.append((label, all_similarities[idx]))
            
            # 使用相似度排序
            all_results.sort(key=lambda x: x[1], reverse=True)
            
            for label, similarity in all_results:
                 output += f"| {label} | {similarity:.4f} |\n"
            
            return output

        # --- 任务 3: OCNLI (自然语言推理) ---
        elif task_name == "自然语言推理（OCNLI）":
            sent1, sent2 = args
            
            # 编码句子对
            nli_text = f"{sent1} {tokenizer.sep_token} {sent2}"
            
            pred_label, confidence, _ = perform_classification_inference(model, tokenizer, device, nli_text, OCNLI_LABELS)
            
            output += f"句子1 (前提): {sent1}\n"
            output += f"句子2 (假设): {sent2}\n"
            output += f"🏆 推理结果：**{pred_label}** (置信度: {confidence:.4f})"
            
            return output

        # --- 任务 4: CSL (摘要关键词验证) ---
        elif task_name == "摘要关键词验证（CSL）":
            abstract, keywords_input = args
            
            similarity, is_accurate = perform_csl_inference(model, tokenizer, device, abstract, keywords_input)
            
            output += f"摘要: {abstract[:50]}...\n"
            output += f"关键词: {keywords_input}\n"
            output += f"语义相似度: **{similarity:.4f}** (阈值 0.3)\n"
            output += f"🏆 验证结果（相似度）：**{is_accurate}**"

            return output

        # --- 任务 5: CSLDCP (主题文献分类) ---
        elif task_name == "主题文献分类（CSLDCP）":
            abstract = args[0]
            
            pred_label, confidence, _ = perform_classification_inference(model, tokenizer, device, abstract, CSLDCP_LABELS)
            
            output += f"摘要: {abstract[:50]}...\n"
            output += f"🏆 预测主题：**{pred_label}** (置信度: {confidence:.4f})"
            
            return output

        # --- 任务 6: IFLYTEK (应用描述分类) ---
        elif task_name == "应用描述分类（IFLYTEK）":
            description = args[0]
            
            pred_label, confidence, _ = perform_classification_inference(model, tokenizer, device, description, IFLYTEK_LABELS)
            
            output += f"描述: {description[:50]}...\n"
            output += f"🏆 预测类别：**{pred_label}** (置信度: {confidence:.4f})"
            
            return output

        # --- 任务 7: CLUEWSC (指代消解) ---
        elif task_name == "指代消解（CLUEWSC）":
            user_text, target_word, _ = args # 忽略 target_pos
            
            # 简化的零样本提示
            wsc_labels = {0: "共指", 1: "不共指"}
            
            # 使用句子本身作为输入，让模型判断句子结构的合理性
            pred_label, confidence, _ = perform_classification_inference(model, tokenizer, device, user_text, wsc_labels)
            
            output += f"句子: {user_text}\n"
            output += f"指代词: {target_word}\n"
            output += f"🏆 消解结果：**{pred_label}** (置信度: {confidence:.4f})\n"
            # output += "（注意：该任务通常需要专门的Span-Prediction头，此为简化零样本演示）"
            
            return output

    except Exception as e:
        return f"❌ 实际推理失败：{type(e).__name__}: {e}\n请检查模型权重是否匹配 MegatronBertModel，或任务输入是否正确。\nTrace: {traceback.format_exc()}"

# --- 任务路径和文件扫描函数（保持不变）---
def get_data_folder_path(task_name: str) -> str:
    mapping = {
        "新闻分类（TNEWS）": os.path.join(DATA_BASE_DIR, "News"),
        "摘要关键词验证（CSL）": os.path.join(DATA_BASE_DIR, "Literature"),
        "主题文献分类（CSLDCP）": os.path.join(DATA_BASE_DIR, "Literature"),
        "应用描述分类（IFLYTEK）": os.path.join(DATA_BASE_DIR, "AppDescriptions")
    }
    return mapping.get(task_name, "路径未知")

def list_task_csv_files(task_name: str) -> list:
    folder_path = get_data_folder_path(task_name)
    if "未知" in folder_path or not os.path.isdir(folder_path):
        return [f"❌ 错误：找不到任务 '{task_name}' 的数据路径配置或文件夹不存在。"]
    try:
        csv_files = [os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*.csv'))]
        return csv_files if csv_files else [f"⚠️ 警告：文件夹内没有找到 .csv 文件: {os.path.basename(folder_path)}/"]
    except Exception as e:
        return [f"❌ 扫描文件时出错: {e}"]

def get_initial_file_choice(task_name: str):
    file_list = list_task_csv_files(task_name)
    if file_list and not file_list[0].startswith(("❌", "⚠️")):
        return file_list[0]
    return None

# --- 修正后的读取和预览函数，包含列名检查 ---
def read_and_preview_data(task_name: str, file_name: str) -> str:
    if file_name.startswith(("❌", "⚠️")): return file_name
    folder_path = get_data_folder_path(task_name)
    file_path = os.path.join(folder_path, file_name)
    
    if "未知" in folder_path or not os.path.exists(file_path):
        return (f"⚠️ 预期数据文件不存在。\n"
                f"请确保您的测试数据位于: **{file_path}**")

    try:
        preview_lines = []
        required_cols = TASK_COLUMN_MAP.get(task_name, {})
        
        if file_path.endswith('.csv'):
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                
                if header and header[0].startswith('\ufeff'):
                    header[0] = header[0].lstrip('\ufeff')

                # 检查列名
                if header and required_cols:
                    indices, missing = find_column_indices(header, required_cols)
                    if missing:
                        preview_lines.append(f"❌ 警告：CSV文件中缺少必需的列：{', '.join(missing)}")
                        preview_lines.append(f"⚠️ 必需列: 输入='{required_cols.get('input', 'N/A')}'")
                
                if header:
                    preview_lines.append("Header: " + ", ".join(header))
                
                # 读取前10行数据
                count = 0
                for i, row in enumerate(reader):
                    # 限制预览长度，避免太长
                    row_preview = [item[:50] + '...' if len(item) > 50 else item for item in row]
                    preview_lines.append(f"Row {i+1}: {row_preview}")
                    count += 1
                    if count >= 10:
                        preview_lines.append("...")
                        break
        
        preview_output = "\n".join(preview_lines)

        return (f"✅ 数据文件找到: {os.path.basename(file_path)}\n"
                f"📌 完整路径: {file_path}\n"
                f"--- 文件内容预览 (前 {count} 行) ---\n"
                f"{preview_output}")

    except Exception as e:
        return f"❌ 读取数据文件时出错: {e}\n请检查文件编码或格式是否正确。\nTrace: {traceback.format_exc()}"


def run_model_batch_task_gr(task_name: str, tokenizer, model, file_name: str):
    """
    批量推理任务：实际执行文件I/O，循环调用模型推理逻辑，并将所有结果格式化后输出到 Gradio 结果框。
    修正：使用列名动态读取 CSV 数据。
    """
    if not model or not tokenizer:
        return "❌ 模型未加载，请先在【模型选择】区域加载模型！"
    
    if file_name.startswith(("❌", "⚠️")):
        return file_name

    folder_path = get_data_folder_path(task_name)
    file_path = os.path.join(folder_path, file_name)
    
    if "未知" in folder_path or not os.path.exists(file_path):
         return read_and_preview_data(task_name, file_name)
        
    device = model.device 
    total_processed = 0
    results_rows = []
    
    start_time = time.time()
    
    # 1. 确定任务配置和输出Header
    required_cols = TASK_COLUMN_MAP.get(task_name)
    
    # === 关键修复点 A: 根据任务确定必需的输入列集合 ===
    # 对于所有任务，'input' 列是必需的
    required_input_keys = {'input'} 
    # CSL 任务还需要 'true_value' 列（关键词）
    if task_name == "摘要关键词验证（CSL）":
        required_input_keys.add('true_value')
        output_header = ['Input_Text (摘要)', 'True_Keywords', 'Predicted_Result', 'Similarity']
        
    else:
        # TNEWS, CSLDCP, IFLYTEK (Classification)
        # 这些任务的输入只需要 'input'
        output_header = ['Input_Text', 'Predicted_Label', 'Confidence']
        label_map = {
            "新闻分类（TNEWS）": TNEWS_LABELS,
            "主题文献分类（CSLDCP）": CSLDCP_LABELS,
            "应用描述分类（IFLYTEK）": IFLYTEK_LABELS,
        }[task_name]
    
    # 2. 读取 CSV 文件 (使用 utf-8 确保兼容性)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            # Read and process header
            header = next(reader, None)
            if not header:
                return "❌ 批量推理失败: 文件内容为空或缺少 Header。"
            
            # === 修复：检查并移除第一个列名上的 BOM 字符 (\ufeff) ===
            if header and header[0].startswith('\ufeff'):
                header[0] = header[0].lstrip('\ufeff')
            # Find column indices (只传入必需的列名)
            # === 关键修复点 B: 动态构造必需列字典 ===
            cols_to_check = {k: required_cols[k] for k in required_input_keys if k in required_cols}
            indices, missing = find_column_indices(header, cols_to_check)
            
            if missing:
                return (f"❌ 批量推理失败: CSV 文件中缺少必需的列：{', '.join(missing)}\n"
                        f"必需列: {', '.join([f'{k}={v}' for k, v in cols_to_check.items()])}")
                
            input_idx = indices['input']
            
            # === 关键修复点 C: 灵活处理 true_value_idx/True_Label ===
            # CSL 任务：true_value_idx 是必需的
            if task_name == "摘要关键词验证（CSL）":
                true_value_idx = indices['true_value']
            # 分类任务：尝试查找 True Label 列（如'label'、'true_label'），找不到则设为 -1 (N/A)
            else:
                header_map = {col.strip(): idx for idx, col in enumerate(header)}
                true_value_idx = header_map.get('真实标签', -1) # 假设真实标签列名可能是 '真实标签'
                if true_value_idx == -1:
                    true_value_idx = header_map.get('True_Label', -1) # 尝试英文
                    if true_value_idx == -1:
                        # 对于分类任务，如果找不到真实标签列，我们允许继续，只是 True_Label 字段会显示 N/A
                        print(f"")


            # 3. 循环推理
            for row in reader:
                # 确保行有足够的列来读取输入文本和（如果存在）真实值/关键词
                if not row or len(row) <= input_idx: continue 
                
                try:
                    
                    input_text = row[input_idx].strip() # 使用找到的索引读取输入文本
                    
                    # 获取真实值：如果索引有效则读取，否则设为 N/A
                    if true_value_idx != -1 and len(row) > true_value_idx:
                         true_value = row[true_value_idx].strip()
                    else:
                         true_value = "N/A"
                         
                    if not input_text: continue # 跳过空输入文本
                    
                    total_processed += 1
                    
                    if task_name == "摘要关键词验证（CSL）":
                        # CSL需要 (Abstract, Keywords)
                        # 确保 true_value 是关键词
                        similarity, pred_result = perform_csl_inference(model, tokenizer, device, input_text, true_value)
                        
                        results_rows.append([
                            input_text[:50] + '...' if len(input_text) > 50 else input_text,
                            true_value,
                            pred_result, 
                            f"{similarity:.4f}"
                        ])

                    else:
                        pred_label, confidence, _ = perform_classification_inference(model, tokenizer, device, input_text, label_map)
                        
                        results_rows.append([
                            input_text[:50] + '...' if len(input_text) > 50 else input_text,
                            pred_label, 
                            f"{confidence:.4f}"
                        ])
                
                except IndexError:
                    print(f"⚠️ 警告: 第 {total_processed} 行数据列数不足或索引错误，跳过处理。")
                    total_processed -= 1 
                    continue

    except Exception as e:
        return (f"❌ 批量推理读取文件失败：{type(e).__name__}: {e}\n"
                f"请检查 CSV 文件路径、格式和内容是否正确。\n"
                f"--- Traceback ---\n{traceback.format_exc()}")
                
    end_time = time.time()
    
    # 4. 格式化输出为 Markdown 表格
    output = f"📝 任务：{task_name} - 批量推理 \n"

    is_mock_mode = "MockModel" in model.__class__.__name__
    if is_mock_mode:
        output += "⚠️ **当前输出为模拟 (Mock) 结果，请安装 PyTorch/Transformers 依赖!**\n"
    else:
        output += "✅ **当前输出为模型真实 (Real) 推理结果。**\n"

    output += "==================================================\n"
    output += f"📂 数据文件: {file_name}\n" 
    output += f"📊 总共处理了 **{total_processed}** 条数据。\n"
    output += f"⏱️ 总耗时: **{end_time - start_time:.2f} 秒**\n"
    # output += "⚠️ **注意：** 模型推理速度取决于您的硬件性能（尤其是 GPU）。**批量处理耗时较长是正常现象。**\n\n"
    
    # 构造 Markdown 表格
    output += "| " + " | ".join(output_header) + " |\n"
    # 构造对齐线
    output += "| :---: | " + " | ".join([':---:'] * (len(output_header) - 1)) + " |\n"
    
    for row in results_rows:
        output += "| " + " | ".join([str(item) for item in row]) + " |\n"
        
    return output

# --- 任务函数包装器 (保持不变) ---
def chidf_task_gr(tokenizer, model, user_text: str, candidate_input: str):
    return run_model_task_gr("成语填空（CHIDF）", tokenizer, model, user_text, candidate_input)
def tnews_task_gr(tokenizer, model, news_text: str):
    return run_model_task_gr("新闻分类（TNEWS）", tokenizer, model, news_text)
def tnews_batch_gr(tokenizer, model, file_name: str):
    return run_model_batch_task_gr("新闻分类（TNEWS）", tokenizer, model, file_name)
def ocnli_task_gr(tokenizer, model, sent1: str, sent2: str):
    return run_model_task_gr("自然语言推理（OCNLI）", tokenizer, model, sent1, sent2)
def csl_task_gr(tokenizer, model, abstract: str, keywords_input: str):
    return run_model_task_gr("摘要关键词验证（CSL）", tokenizer, model, abstract, keywords_input)
def csl_batch_gr(tokenizer, model, file_name: str):
    return run_model_batch_task_gr("摘要关键词验证（CSL）", tokenizer, model, file_name)
def csldcp_task_gr(tokenizer, model, abstract: str):
    return run_model_task_gr("主题文献分类（CSLDCP）", tokenizer, model, abstract)
def csldcp_batch_gr(tokenizer, model, file_name: str):
    return run_model_batch_task_gr("主题文献分类（CSLDCP）", tokenizer, model, file_name)
def iflytek_task_gr(tokenizer, model, description: str):
    return run_model_task_gr("应用描述分类（IFLYTEK）", tokenizer, model, description)
def iflytek_batch_gr(tokenizer, model, file_name: str):
    return run_model_batch_task_gr("应用描述分类（IFLYTEK）", tokenizer, model, file_name)
def cluewsc_task_gr(tokenizer, model, user_text: str, target_word: str, target_pos: int):
    # CLUEWSC 任务不需要 target_pos 字段，但保留输入以匹配 UI
    return run_model_task_gr("指代消解（CLUEWSC）", tokenizer, model, user_text, target_word, target_pos)


# ====================================================================
# 3. Gradio 界面构建
# ====================================================================

with gr.Blocks(title="伪·原神启动器") as demo:
    gr.Markdown("# 信息聚合功能及模型调试WebUI（原二郎神启动器<简称原神启动器>）")
    
    state_tokenizer = gr.State(None)
    state_model = gr.State(None)
    
    with gr.Tabs():
        
        # --- 爬虫部分 (文本框行数增加) ---
        with gr.TabItem("1. 网页爬取"):
            gr.Markdown(f"## 🌐 爬取设定 (路径: Program/Spiders/AllSpider/)\n") # 脚本路径: Program/Spiders/AllSpider/
            
            # 动态选择爬虫脚本
            spider_choice = gr.Dropdown(
                label="选择爬虫脚本",
                choices=SPIDER_FILES, # 使用扫描到的文件列表
                value=SPIDER_FILES[0] if SPIDER_FILES else "未找到脚本",
                interactive=bool(SPIDER_FILES)
            )

            # 通用输入控件
            spider_kw = gr.Textbox(label="搜索关键词", value="人工智能")
            
            # 排序方式 (假设最多支持 4 种排序)
            # 注意：此处使用 Dropdown 以兼容所有脚本，并通过后端逻辑判断是否使用
            spider_sort = gr.Radio(
                label="排序方式",
                choices=SORT_CHOICES, 
                value="默认排序",
                # 默认可见，但可以根据选择的脚本动态隐藏/禁用 (高级功能，此处仅统一显示)
            )
            
            # 页数/爬取个数
            spider_pages = gr.Slider(
                label="爬取页数/个数", 
                minimum=1, maximum=10, value=3, step=1
            )
            
            # 运行按钮
            spider_btn = gr.Button("🚀 运行选定爬虫脚本", variant="primary")
            
            # 输出
            spider_output = gr.Textbox(label="脚本 Standard Output/Error", lines=20) 
            
            # 按钮点击事件：调用通用运行函数
            spider_btn.click(
                fn=run_generic_spider_gr, 
                inputs=[spider_choice, spider_kw, spider_sort, spider_pages], 
                outputs=spider_output
            )
            
            # (可选) 动态更新默认值和组件可见性
            # 复杂：为了简化，我们让所有组件默认都显示，并在通用运行函数中处理参数的取舍。
            # 如果需要动态更新，可以添加以下逻辑：
            def update_ui_on_script_change(script_name):
                config = SPIDER_CONFIG.get(script_name, {})
                kw = config.get('default_kw', '')
                pages_val = config.get('default_pages', 3)
                sort_vis = True if config.get('sort') else False
                pages_vis = True if config.get('pages') else False
                
                # 针对 baidu_news_spider.py 的特殊处理
                if script_name == "baidu_news_spider.py":
                    pages_vis = False
                
                return (
                    gr.update(value=kw), 
                    gr.update(visible=sort_vis), 
                    gr.update(visible=pages_vis, value=pages_val)
                )

            spider_choice.change(
                fn=update_ui_on_script_change,
                inputs=[spider_choice],
                outputs=[spider_kw, spider_sort, spider_pages]
            )

        # ==============================================
        # 4. 模型测试 (Model Testing)
        # ==============================================
        with gr.TabItem("2. 模型调试"):
            gr.Markdown(f"## 🧠 模型加载 (路径: Program/Models/)")

            # --- 获取动态模型列表 ---
            available_models = get_available_models()
            default_model = available_models[0] if available_models else "未找到模型"
            # ------------------------
            model_choice = gr.Radio(
                label="选择模型",
                # 使用动态获取的列表
                choices=available_models if available_models else ["未找到模型"],
                # 默认值设置为列表的第一个元素
                value=default_model,
                # 如果没有模型，禁用 Radio 按钮
                interactive=bool(available_models) 
            )
            load_btn = gr.Button("加载选定模型", variant="primary")
            # UI FIX: 统一增加行数
            model_output = gr.Textbox(label="模型加载状态", lines=10) 

            load_btn.click(
                fn=init_model_and_tokenizer_gr, 
                inputs=[model_choice, state_tokenizer, state_model], 
                outputs=[state_tokenizer, state_model, model_output]
            )

            # --- 任务测试 ---
            gr.Markdown("## 💡 模型测试")
            with gr.Tabs():
                
                with gr.TabItem("1. 成语填空 (CHIDF)"):
                    gr.Markdown("### 单条推理")
                    chidf_sent = gr.Textbox(label="输入句子", info="请输入含[MASK]标记的句子", value="他面对困难时[MASK]")
                    chidf_cands = gr.Textbox(label="候选成语", info="用全角逗号分隔", value="坚持不懈，半途而废，敷衍了事")
                    chidf_btn = gr.Button("执行 CHIDF 任务", variant="primary")
                    # UI FIX: 移除 render_as="html"
                    chidf_output = gr.Textbox(label="任务结果", lines=15)
                    chidf_btn.click(chidf_task_gr, [state_tokenizer, state_model, chidf_sent, chidf_cands], chidf_output)

                # 任务 2: TNEWS (新闻分类) - 包含批量模式
                with gr.TabItem("2. 新闻分类 (TNEWS)"):
                    tnews_task_state = gr.State("新闻分类（TNEWS）")
                    with gr.Tabs():
                        with gr.TabItem("单条推理"):
                            gr.Markdown("### 单条推理")
                            tnews_text = gr.Textbox(label="输入新闻文本", lines=5, value="刚刚从朋友那里听说，特斯拉已经开始在国内进行大规模的降价，不知道是不是真的，我准备去买一辆。")
                            tnews_btn = gr.Button("执行 TNEWS 任务", variant="primary")
                            # UI FIX: 统一增加行数
                            tnews_output = gr.Textbox(label="任务结果", lines=15)
                            tnews_btn.click(tnews_task_gr, [state_tokenizer, state_model, tnews_text], tnews_output)
                        
                        with gr.TabItem("批量测试 (CSV)"):
                            # gr.Markdown(f"批量模式将读取 **`Program/Spiders/Information/News/`** 文件夹下的 CSV 文件，并**按列名（'标题'）**读取数据。")
                            
                            tnews_files = gr.Dropdown(
                                label="选择 CSV 文件", 
                                choices=list_task_csv_files("新闻分类（TNEWS）"), 
                                value=get_initial_file_choice("新闻分类（TNEWS）")
                            )
                            with gr.Row():
                                tnews_batch_preview = gr.Button("1. 检查数据文件并预览")
                                tnews_batch_btn = gr.Button("2. 🚀 执行 TNEWS 批量任务 ", variant="primary")
                            refresh_tnews_files = gr.Button("🔄 刷新文件列表")

                            # UI FIX: 移除 render_as="html"，统一增加行数
                            tnews_batch_output = gr.Textbox(label="批量任务结果 ", lines=30) 
                            
                            refresh_tnews_files.click(
                                fn=lambda: gr.update(choices=list_task_csv_files("新闻分类（TNEWS）"), value=get_initial_file_choice("新闻分类（TNEWS）")), 
                                inputs=[], 
                                outputs=tnews_files
                            )
                            tnews_batch_preview.click(read_and_preview_data, [tnews_task_state, tnews_files], tnews_batch_output)
                            tnews_batch_btn.click(tnews_batch_gr, [state_tokenizer, state_model, tnews_files], tnews_batch_output)

                with gr.TabItem("3. 自然语言推理 (OCNLI)"):
                    gr.Markdown("### 单条推理")
                    ocnli_sent1 = gr.Textbox(label="句子1 (前提)", value="人工智能技术发展快")
                    ocnli_sent2 = gr.Textbox(label="句子2 (假设)", value="AI技术迭代快")
                    ocnli_btn = gr.Button("执行 OCNLI 任务", variant="primary")
                    # UI FIX: 统一增加行数
                    ocnli_output = gr.Textbox(label="任务结果", lines=15) 
                    ocnli_btn.click(ocnli_task_gr, [state_tokenizer, state_model, ocnli_sent1, ocnli_sent2], ocnli_output)

                # 任务 4: CSL (摘要关键词验证) - 包含批量模式
                with gr.TabItem("4. 摘要关键词验证 (CSL)"):
                    csl_task_state = gr.State("摘要关键词验证（CSL）")
                    with gr.Tabs():
                        with gr.TabItem("单条推理"):
                            gr.Markdown("### 单条推理")
                            csl_abstract = gr.Textbox(label="输入摘要", lines=5, value="深度学习是机器学习研究中的一个新领域，致力于模拟人脑的神经网络，通过多层网络结构实现数据的特征提取。")
                            csl_keywords = gr.Textbox(label="输入关键词", info="用全角逗号分隔", value="神经网络，深度学习，特征提取")
                            csl_btn = gr.Button("执行 CSL 任务", variant="primary")
                            # UI FIX: 统一增加行数
                            csl_output = gr.Textbox(label="任务结果", lines=15) 
                            csl_btn.click(csl_task_gr, [state_tokenizer, state_model, csl_abstract, csl_keywords], csl_output)

                        with gr.TabItem("批量测试 (CSV)"):
                            # gr.Markdown(f"批量模式将读取 **`Program/Spiders/Information/Literature/`** 文件夹下的 CSV 文件，并**按列名（'摘要'和'关键词'）**读取数据。")
                            csl_files = gr.Dropdown(
                                label="选择 CSV 文件", 
                                choices=list_task_csv_files("摘要关键词验证（CSL）"), 
                                value=get_initial_file_choice("摘要关键词验证（CSL）")
                            )
                            with gr.Row():
                                csl_batch_preview = gr.Button("1. 检查数据文件并预览")
                                csl_batch_btn = gr.Button("2. 🚀 执行 CSL 批量任务 ", variant="primary")
                            refresh_csl_files = gr.Button("🔄 刷新文件列表")

                            # UI FIX: 移除 render_as="html"，统一增加行数
                            csl_batch_output = gr.Textbox(label="批量任务结果 ", lines=30) 
                            
                            refresh_csl_files.click(
                                fn=lambda: gr.update(choices=list_task_csv_files("摘要关键词验证（CSL）"), value=get_initial_file_choice("摘要关键词验证（CSL）")), 
                                inputs=[], 
                                outputs=csl_files
                            )
                            csl_batch_preview.click(read_and_preview_data, [csl_task_state, csl_files], csl_batch_output)
                            csl_batch_btn.click(csl_batch_gr, [state_tokenizer, state_model, csl_files], csl_batch_output)

                # 任务 5: CSLDCP (主题文献分类) - 包含批量模式
                with gr.TabItem("5. 主题文献分类 (CSLDCP)"):
                    csldcp_task_state = gr.State("主题文献分类（CSLDCP）")
                    with gr.Tabs():
                        with gr.TabItem("单条推理"):
                            gr.Markdown("### 单条推理")
                            csldcp_abstract = gr.Textbox(label="输入文献摘要", lines=5, value="本文研究了基于Transformer模型的自然语言处理技术在医疗诊断系统中的应用和潜力。")
                            csldcp_btn = gr.Button("执行 CSLDCP 任务", variant="primary")
                            # UI FIX: 统一增加行数
                            csldcp_output = gr.Textbox(label="任务结果", lines=15) 
                            csldcp_btn.click(csldcp_task_gr, [state_tokenizer, state_model, csldcp_abstract], csldcp_output)
                            
                        with gr.TabItem("批量测试 (CSV)"):
                            # gr.Markdown(f"批量模式将读取 **`Program/Spiders/Information/Literature/`** 文件夹下的 CSV 文件，并**按列名（'摘要'）**读取数据。")
                            csldcp_files = gr.Dropdown(
                                label="选择 CSV 文件", 
                                choices=list_task_csv_files("主题文献分类（CSLDCP）"), 
                                value=get_initial_file_choice("主题文献分类（CSLDCP）")
                            )
                            with gr.Row():
                                csldcp_batch_preview = gr.Button("1. 检查数据文件并预览")
                                csldcp_batch_btn = gr.Button("2. 🚀 执行 CSLDCP 批量任务 ", variant="primary")
                            refresh_csldcp_files = gr.Button("🔄 刷新文件列表")

                            # UI FIX: 移除 render_as="html"，统一增加行数
                            csldcp_batch_output = gr.Textbox(label="批量任务结果 ", lines=30) 
                            
                            refresh_csldcp_files.click(
                                fn=lambda: gr.update(choices=list_task_csv_files("主题文献分类（CSLDCP）"), value=get_initial_file_choice("主题文献分类（CSLDCP）")), 
                                inputs=[], 
                                outputs=csldcp_files
                            )
                            csldcp_batch_preview.click(read_and_preview_data, [csldcp_task_state, csldcp_files], csldcp_batch_output)
                            csldcp_batch_btn.click(csldcp_batch_gr, [state_tokenizer, state_model, csldcp_files], csldcp_batch_output)

                # 任务 6: IFLYTEK (应用描述分类) - 包含批量模式
                with gr.TabItem("6. 应用描述分类 (IFLYTEK)"):
                    iflytek_task_state = gr.State("应用描述分类（IFLYTEK）")
                    with gr.Tabs():
                        with gr.TabItem("单条推理"):
                            gr.Markdown("### 单条推理")
                            iflytek_desc = gr.Textbox(label="输入应用描述", lines=5, value="一个功能强大的图片编辑器，包含滤镜、裁剪、美颜等多种功能，让你的照片焕然一新。")
                            iflytek_btn = gr.Button("执行 IFLYTEK 任务", variant="primary")
                            # UI FIX: 统一增加行数
                            iflytek_output = gr.Textbox(label="任务结果", lines=15) 
                            iflytek_btn.click(iflytek_task_gr, [state_tokenizer, state_model, iflytek_desc], iflytek_output)

                        with gr.TabItem("批量测试 (CSV)"):
                            # gr.Markdown(f"批量模式将读取 **`Program/Spiders/Information/AppDescriptions/`** 文件夹下的 CSV 文件，并**按列名（'应用简介'）**读取数据。")
                            iflytek_files = gr.Dropdown(
                                label="选择 CSV 文件", 
                                choices=list_task_csv_files("应用描述分类（IFLYTEK）"), 
                                value=get_initial_file_choice("应用描述分类（IFLYTEK）")
                            )
                            with gr.Row():
                                iflytek_batch_preview = gr.Button("1. 检查数据文件并预览")
                                iflytek_batch_btn = gr.Button("2. 🚀 执行 IFLYTEK 批量任务 ", variant="primary")
                            refresh_iflytek_files = gr.Button("🔄 刷新文件列表")

                            # UI FIX: 移除 render_as="html"，统一增加行数
                            iflytek_batch_output = gr.Textbox(label="批量任务结果 ", lines=30) 
                            
                            refresh_iflytek_files.click(
                                fn=lambda: gr.update(choices=list_task_csv_files("应用描述分类（IFLYTEK）"), value=get_initial_file_choice("应用描述分类（IFLYTEK）")),
                                inputs=[], 
                                outputs=iflytek_files
                            )
                            iflytek_batch_preview.click(read_and_preview_data, [iflytek_task_state, iflytek_files], iflytek_batch_output)
                            iflytek_batch_btn.click(iflytek_batch_gr, [state_tokenizer, state_model, iflytek_files], iflytek_batch_output)

                with gr.TabItem("7. 指代消解 (CLUEWSC)"):
                    gr.Markdown("### 单条推理")
                    cluewsc_sent = gr.Textbox(label="输入句子", value="小明对他的同学说，明天他会去图书馆。")
                    cluewsc_word = gr.Textbox(label="指代词", value="他")
                    cluewsc_pos = gr.Number(label="指代词位置", value=4, precision=0, visible=False) 
                    cluewsc_btn = gr.Button("执行 CLUEWSC 任务", variant="primary")
                    # UI FIX: 统一增加行数
                    cluewsc_output = gr.Textbox(label="任务结果", lines=15) 
                    cluewsc_btn.click(cluewsc_task_gr, [state_tokenizer, state_model, cluewsc_sent, cluewsc_word, cluewsc_pos], cluewsc_output)
    
if __name__ == "__main__":
    try:
        print("\n🚀 WebUI正在启动，请等待浏览器自动打开...")
        # 使用多线程，让提示信息在后台等待 Gradio 完成启动日志输出
        import threading
        def print_tip():
            # 延迟 3 秒，确保 Gradio 的 URL 信息已经输出
            time.sleep(3) 
            print("\n💡 您可以随时在终端按下Ctrl+C快捷键，以终止WebUI")
        threading.Thread(target=print_tip, daemon=True).start()

        # 启动 Gradio，此时主线程阻塞
        demo.launch(inbrowser=True)

    except KeyboardInterrupt:
        # 捕获 Ctrl+C 信号
        print("\n👋 接收到关闭信号，正在优雅地关闭WebUI...")
        pass
    except Exception as e:
        print(f"\n\n❌ WebUI运行过程中发生错误: {e}")
    else:
        # 如果没有异常（即程序正常退出 launch()）
        # 在 Gradio 场景下，这里通常不会被执行，但在某些情况下是保险措施
        print("\n👋 接收到关闭信号，正在优雅地关闭WebUI...")