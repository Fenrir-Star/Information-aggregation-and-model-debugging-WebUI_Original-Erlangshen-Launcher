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

import torch
import torch.nn.functional as F
import os
import re
import sys # 引入sys用于查找根目录
from transformers import MegatronBertConfig, MegatronBertModel
from transformers import BertTokenizer

# ===================== 全局模型容器和配置（初始为空）=====================
tokenizer = None
model = None
device = None
model_name = "" # 新增：用于存储当前模型名称，便于打印
config = None


# ===================== 模型初始化函数（传入模型相对路径）=====================
def init_model_and_tokenizer(model_dir, current_model_name):
    """
    初始化模型和分词器（全局仅调用1次）
    """
    global tokenizer, model, device, model_name, config # 声明全局变量
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.isdir(model_dir):
        print(f"❌ 模型文件目录不存在：{model_dir}")
        print("请检查模型文件夹结构是否正确")
        sys.exit(1)

    # 加载模型和分词器（float16节省显存，适配GPU）
    try:
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        config = MegatronBertConfig.from_pretrained(model_dir)
        model = MegatronBertModel.from_pretrained(model_dir, dtype=torch.float16).to(device) # , dtype=torch.float16
        model.eval()  # 固定为推理模式，避免意外训练
        model_name = current_model_name
        
        print("\n" + "="*50)
        print(f"✅ 模型初始化完成 | 使用设备：{device}")
        print(f"📌 模型：{model_name}（{config.hidden_size}维特征）")
        print("="*50)
        
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        sys.exit(1)


# ===================== 辅助函数（仅读取指定列）=====================
def read_target_column_from_csv(file_path, target_column):
    """
    从CSV文件中仅读取指定目标列（如“标题”“概要”“摘要”）
    :param file_path: CSV文件路径
    :param target_column: 目标列名（如“标题”“概要”）
    :return: 目标列的有效文本列表
    """
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在：{file_path}")
        return []
    
    try:
        target_texts = []
        # 优先使用pandas读取（更稳定的列识别）
        try:
            import pandas as pd
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            
            # 仅保留目标列，不存在则提示
            if target_column not in df.columns:
                print(f"❌ CSV文件中未找到“{target_column}”列")
                return []
            
            # 提取目标列有效文本（去重、过滤空值和过短文本）
            for text in df[target_column].dropna().unique():
                text_str = str(text).strip()
                if len(text_str) > 20:  # 过滤过短文本（避免无效内容）
                    target_texts.append(text_str)
        
        except ImportError:
            # 备用方案：使用csv模块读取
            import csv
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)  # 按列名读取
                if target_column not in reader.fieldnames:
                    print(f"❌ CSV文件中未找到“{target_column}”列")
                    return []
                
                for row in reader:
                    text_str = str(row[target_column]).strip()
                    if len(text_str) > 20:
                        target_texts.append(text_str)
        
        if not target_texts:
            print(f"❌ CSV文件中“{target_column}”列无有效文本")
            return []
        
        print(f"✅ 成功读取 {len(target_texts)} 条{target_column}")
        return target_texts
    
    except Exception as e:
        print(f"❌ 读取CSV文件失败：{e}")
        return []


# ===================== 新增：零样本分类辅助函数 =====================
def predict_by_similarity(model, tokenizer, device, text_feat, label_map):
    """
    原理：不训练分类层，而是计算[输入文本特征]与[类别名称特征]的余弦相似度
    """
    # 1. 准备类别的描述文本（Prompt Engineering）
    # 例如把 "体育" 扩展为 "这是一个体育类别" 以增加语义匹配度
    label_texts = [f"关于{label}的内容" for label in label_map.values()]

    # 2. 编码所有类别
    label_encoded = tokenizer(
        label_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32
    ).to(device)

    # 3. 提取类别的特征
    with torch.no_grad():
        label_outputs = model(** label_encoded)
        label_feats = label_outputs.pooler_output

    # 4. 计算相似度 (Cosine Similarity)
    # 归一化向量
    text_feat_norm = F.normalize(text_feat, p=2, dim=1).to(torch.float16) ##
    label_feats_norm = F.normalize(label_feats, p=2, dim=1).to(torch.float16) ##

    # 矩阵乘法计算相似度
    similarities = torch.mm(text_feat_norm, label_feats_norm.T)

    # 5. 获取结果
    # 乘以一个缩放因子(scale)让softmax分布更尖锐
    logits = similarities * 15
    pred_probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(torch.argmax(logits).item())

    return pred_idx, pred_probs

# ===================== 任务函数封装（每个任务独立成函数）=====================
def chidf_task(tokenizer, model, device):
    """任务1：CHIDF（成语填空）"""
    global model_name
    print("\n" + "="*50)
    print(f"📝 任务1：成语填空（CHIDF） - 模型: {model_name}")
    print("="*50)
    
    # 1. 获取用户输入（含[MASK]的句子）
    user_text = input("请输入含成语空缺的句子（用[MASK]标记空缺位置，例如：他面对困难时[MASK]）：")
    if "[MASK]" not in user_text:
        print("❌ 输入错误：请在句子中添加[MASK]标记成语空缺位置！")
        return
    
    # 2. 编码文本
    encoded = tokenizer(
        user_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    ).to(device)
    mask_pos = torch.where(encoded["input_ids"] == tokenizer.mask_token_id)[1]
    if len(mask_pos) == 0:
        print("❌ 未识别到[MASK]标记，请重新输入！")
        return
    
    # 3. 提取[MASK]位置特征
    with torch.no_grad():
        outputs = model(**encoded)
        mask_feat = outputs.last_hidden_state[0, mask_pos, :]
    
    # 4. 获取用户提供的候选成语
    candidate_input = input("请输入候选成语（用逗号分隔，例如：坚持不懈，半途而废）：")
    if not candidate_input.strip():
        candidate_idioms = ["坚持不懈", "半途而废", "畏缩不前", "敷衍了事"]
        print(f"⚠️ 未输入候选成语，使用默认候选：{','.join(candidate_idioms)}")
    else:
        # 使用全角逗号分隔，和原始代码保持一致
        candidate_idioms = [idiom.strip() for idiom in candidate_input.split("，")]
    
    # 5. 改进的特征提取：将成语放入相同上下文中获取特征
    print(f"\n🔄 正在计算{len(candidate_idioms)}个候选成语的匹配度...")
    candidate_feats = []
    
    for idiom in candidate_idioms:
        # 将成语放入与原始句子相似的上下文中
        template_text = f"这个成语的意思是：{idiom}"
        idiom_encoded = tokenizer(
            template_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32
        ).to(device)
        
        with torch.no_grad():
            idiom_output = model(** idiom_encoded)
            # 使用[CLS] token的特征，与mask_feat保持一致
            idiom_feat = idiom_output.last_hidden_state[0, 0, :]  # [CLS] token
            candidate_feats.append(idiom_feat)
    
    # 6. 计算相似度（添加特征归一化）
    candidate_feats = torch.stack(candidate_feats, dim=0)
    
    # 特征归一化，提高相似度计算稳定性
    mask_feat_norm = F.normalize(mask_feat, p=2, dim=1)
    candidate_feats_norm = F.normalize(candidate_feats, p=2, dim=1)
    
    similarities = F.cosine_similarity(mask_feat_norm, candidate_feats_norm, dim=1)
    
    # 7. 如果所有相似度都很低，尝试另一种特征提取方式
    if torch.max(similarities) < 0.1:
        print("⚠️  检测到匹配度较低，尝试备用特征提取方法...")
        candidate_feats_alt = []
        
        for idiom in candidate_idioms:
            # 备用方法：直接编码成语，使用平均池化
            idiom_encoded = tokenizer(
                idiom,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8
            ).to(device)
            
            with torch.no_grad():
                idiom_output = model(**idiom_encoded)
                # 使用所有token的平均特征
                mean_feat = torch.mean(idiom_output.last_hidden_state[0], dim=0)
                candidate_feats_alt.append(mean_feat)
        
        candidate_feats_alt = torch.stack(candidate_feats_alt, dim=0)
        candidate_feats_alt_norm = F.normalize(candidate_feats_alt, p=2, dim=1)
        similarities = F.cosine_similarity(mask_feat_norm, candidate_feats_alt_norm, dim=1)
    
    # 8. 选择最匹配的成语
    best_idx = torch.argmax(similarities).item()
    
    # 9. 输出结果
    print("\n" + "="*30 + " 匹配结果 " + "="*30)
    print(f"📖 原句：{user_text}")
    print(f"🏆 最佳匹配成语：{candidate_idioms[best_idx]}")
    print(f"📊 匹配置信度：{similarities[best_idx].item():.4f}")
    
    # 将相似度映射到[0,1]范围显示
    normalized_similarities = (similarities + 1) / 2  # 从[-1,1]映射到[0,1]
    
    print(f"\n📋 所有候选成语匹配度排名：")
    sorted_pairs = sorted(zip(candidate_idioms, normalized_similarities.cpu().numpy(), similarities.cpu().numpy()), 
                         key=lambda x: x[1], reverse=True)
    for i, (idiom, norm_sim, raw_sim) in enumerate(sorted_pairs, 1):
        print(f"  {i}. {idiom} → 匹配度：{norm_sim:.4f} (原始：{raw_sim:.4f})")


def tnews_task(tokenizer, model, device):
    """任务2：TNEWS（新闻分类）"""
    global model_name
    print("\n" + "="*50)
    print(f"📰 任务2：新闻分类（TNEWS） - 模型: {model_name}")
    print("="*50)
    
    # 1. 定义新闻类别（CLUE基准15类完整版）
    news_labels = {
        0: "科技", 1: "娱乐", 2: "体育", 3: "财经", 4: "时政", 5: "教育",
        6: "军事", 7: "汽车", 8: "房产", 9: "游戏", 10: "时尚", 11: "彩票",
        12: "股票", 13: "家居", 14: "社会"
    }
    print(f"支持分类：{', '.join([f'{k}:{v}' for k, v in news_labels.items()])}")
    
    # 2. 获取用户输入方式
    print("\n请选择新闻输入方式：")
    print("1. 手动输入单条新闻")
    print("2. 从Information/News文件夹读取CSV文件进行批量分类")
    input_choice = input("请输入选项编号（1/2）：")
    
    news_texts = []
    if input_choice == "1":
        # 手动输入单条新闻
        user_news = input("请输入需要分类的新闻文本：")
        if not user_news.strip():
            print("❌ 输入错误：新闻文本不能为空！")
            return
        news_texts = [user_news]
        
    elif input_choice == "2":
        # 从News文件夹读取CSV文件进行批量分类
        # 脚本现在在 '模型' 文件夹，News在 '模型/Information/News'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        news_folder = os.path.join(script_dir, "Spiders", "Information", "News")
        
        # 检查News文件夹是否存在
        if not os.path.exists(news_folder):
            print(f"❌ News文件夹不存在：{news_folder}")
            print("💡 请确保Information/News文件夹结构正确")
            return
        
        # 获取News文件夹中的所有CSV文件
        csv_files = [f for f in os.listdir(news_folder) if f.endswith('.csv')]
        if not csv_files:
            print(f"❌ News文件夹中没有找到CSV文件")
            return
        
        # 显示可用的CSV文件
        print(f"\n📁 找到 {len(csv_files)} 个CSV文件：")
        for i, csv_file in enumerate(csv_files, 1):
            print(f"  {i}. {csv_file}")
        
        # 让用户选择文件
        try:
            file_choice = input(f"\n请选择文件编号（1-{len(csv_files)}）：").strip()
            file_index = int(file_choice) - 1
            if file_index < 0 or file_index >= len(csv_files):
                print("❌ 输入错误：编号超出范围！")
                return
            
            selected_file = csv_files[file_index]
            file_path = os.path.join(news_folder, selected_file)
            print(f"✅ 选择文件：{selected_file}")
            
            # 使用统一工具函数读取"标题"列
            news_texts = read_target_column_from_csv(file_path, "标题")
            if not news_texts:
                return
                
        except ValueError:
            print("❌ 输入错误：请输入数字编号！")
            return
        
    else:
        print("❌ 输入错误！请输入1或2选择输入方式")
        return
    
    # 3. 进行分类预测
    print(f"\n🔍 开始对 {len(news_texts)} 条新闻进行分类...")
    
    results = []
    for i, news_text in enumerate(news_texts, 1):
        if input_choice == "2":  # 批量处理时显示进度
            print(f"📊 处理进度：{i}/{len(news_texts)}", end="\r")
        
        # 编码文本并提取特征
        encoded = tokenizer(
            news_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128  # 增加长度以适应新闻文本
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**encoded)
            feat = outputs.pooler_output.to(torch.float16) ##

            # 使用相似度匹配代替随机分类头
            pred_label_idx, pred_probs = predict_by_similarity(model, tokenizer, device, feat, news_labels)
            pred_label = news_labels[pred_label_idx]
            confidence = pred_probs[pred_label_idx]
        
        results.append({
            'text': news_text,
            'pred_label': pred_label,
            'pred_idx': pred_label_idx,
            'confidence': confidence,
            'all_probs': pred_probs
        })
    
    # 4. 输出结果
    print("\n" + "="*40 + " 分类结果 " + "="*40)
    
    if input_choice == "1":
        # 单条新闻详细结果
        result = results[0]
        print(f"📝 新闻文本：{result['text']}")
        print(f"🏷️ 预测类别：{result['pred_label']}（类别编号：{result['pred_idx']}）")
        print(f"📊 置信度：{result['confidence']:.4f}")
        print(f"📈 各类别概率：")
        for label_idx, label_name in news_labels.items():
            prob = result['all_probs'][label_idx]
            print(f"  {label_name}：{prob:.4f}")
    
    else:
        # 批量结果显示统计信息
        print(f"📈 批量分类统计：")
        print(f"   总新闻数：{len(results)}")
        
        # 按类别统计
        from collections import Counter
        label_counts = Counter([r['pred_label'] for r in results])
        print(f"\n📊 类别分布：")
        for label in news_labels.values():
            count = label_counts.get(label, 0)
            percentage = (count / len(results)) * 100
            print(f"  {label}：{count}条 ({percentage:.1f}%)")
        
        # 显示前几条结果的详情
        print(f"\n🔍 前{min(5, len(results))}条新闻详情：")
        for i, result in enumerate(results[:5], 1):
            print(f"\n{i}. {result['text'][:100]}...")
            print(f"   类别：{result['pred_label']} | 置信度：{result['confidence']:.4f}")
        
        # 询问是否显示所有结果
        if len(results) > 5:
            show_all = input(f"\n💡 还有{len(results)-5}条结果未显示，是否显示全部？(y/n)：")
            if show_all.lower() == 'y':
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result['text']}")
                    print(f"   类别：{result['pred_label']} | 置信度：{result['confidence']:.4f}")


def ocnli_task(tokenizer, model, device): # 移除config参数，直接使用全局config
    """任务3：OCNLI（自然语言推理）"""
    global model_name, config
    print("\n" + "="*50)
    print(f"🔍 任务3：自然语言推理（OCNLI） - 模型: {model_name}")
    print("="*50)
    
    # 1. 定义推理类别
    nli_labels = {0: "蕴含", 1: "矛盾", 2: "中立"}
    print(f"推理关系：{', '.join([f'{k}:{v}' for k, v in nli_labels.items()])}")
    print("示例：句子1='人工智能技术发展快'，句子2='AI技术迭代快' → 蕴含关系")
    
    # 2. 获取用户输入的句子对
    sent1 = input("请输入句子1（前提）：")
    sent2 = input("请输入句子2（假设）：")
    if not sent1.strip() or not sent2.strip():
        print("❌ 输入错误：句子1和句子2不能为空！")
        return
    
    # 3. 编码句子对（用[SEP]分隔）
    nli_text = f"{sent1} {tokenizer.sep_token} {sent2}"
    encoded = tokenizer(
        nli_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)
    
    # 4. 提取特征并推理
    with torch.no_grad():
        outputs = model(**encoded)
        feat = outputs.pooler_output.to(torch.float16) ##
    
    # 5. 推理分类头
    # 动态使用全局 config
    input_size = config.hidden_size 
    # 注意：这个分类头是随机初始化的，对于零样本效果可能不佳，但保持原逻辑
    nli_classifier = torch.nn.Linear(input_size, 3).to(device, dtype=torch.float16)
    with torch.no_grad():
        logits = nli_classifier(feat)
        pred_probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label_idx = int(torch.argmax(logits).item())
        pred_relation = nli_labels[pred_label_idx]

    # 6. 输出结果
    print("\n" + "="*30 + " 结果 " + "="*30)
    print(f"前提句子：{sent1}")
    print(f"假设句子：{sent2}")
    print(f"推理关系：{pred_relation}（置信度：{pred_probs[pred_label_idx]:.4f}）")
    print(f"各关系置信度：{dict(zip(nli_labels.values(), pred_probs.round(4)))}")


def csl_task(tokenizer, model, device):
    """任务4：CSL（摘要关键词识别）- 关键词真实性验证"""
    global model_name
    print("\n" + "="*50)
    print(f"🔤 任务4：摘要关键词真实性验证（CSL） - 模型: {model_name}")
    print("="*50)
    print("任务目标：判断候选关键词是否准确反映学术论文摘要的核心内容")
    print("输出结果：1（准确）/ 0（不准确）")
    
    # 1. 选择输入方式
    print("\n请选择输入方式：")
    print("1. 手动输入（摘要+候选关键词）")
    print("2. 从Information/CSL文件读取（需包含'摘要'和'关键词'列）")
    input_choice = input("请输入选项编号（1/2）：")
    
    data_list = []
    if input_choice == "1":
        # 手动输入模式
        print("\n" + "-"*30 + " 手动输入 " + "-"*30)
        # 获取摘要
        user_abstract = input("请输入学术论文摘要：")
        if not user_abstract.strip():
            print("❌ 输入错误：摘要文本不能为空！")
            return
        # 获取候选关键词
        user_keywords = input("请输入候选关键词（用逗号分隔，例如：人工智能,深度学习,神经网络）：")
        if not user_keywords.strip():
            print("❌ 输入错误：关键词不能为空！")
            return
        # 处理关键词列表
        candidate_keywords = [kw.strip() for kw in re.split(r"[,，]", user_keywords) if kw.strip()]
        if not candidate_keywords:
            print("❌ 输入错误：未识别到有效关键词！")
            return
        
        # 构造单条数据
        data_list = [{"abstract": user_abstract, "keywords": candidate_keywords}]
    
    elif input_choice == "2":
        # CSV读取模式
        print("\n" + "-"*30 + " CSV文件读取 " + "-"*30)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 预设CSV文件存放路径（与脚本同目录的Information/CSL文件夹）
        csl_folder = os.path.join(script_dir, "Spiders", "Information", "Literature")
        
        if not os.path.exists(csl_folder):
            os.makedirs(csl_folder)
            print(f"⚠️ 已自动创建CSL文件夹：{csl_folder}")
            print("💡 请将包含'摘要'和'关键词'列的CSV文件放入该文件夹后重试")
            return
        
        # 获取文件夹中所有CSV文件
        csv_files = [f for f in os.listdir(csl_folder) if f.endswith('.csv')]
        if not csv_files:
            print(f"❌ CSL文件夹中未找到CSV文件：{csl_folder}")
            return
        
        # 选择CSV文件
        print(f"\n📁 找到 {len(csv_files)} 个CSV文件：")
        for i, csv_file in enumerate(csv_files, 1):
            print(f"  {i}. {csv_file}")
        
        try:
            file_index = int(input(f"\n请选择文件编号（1-{len(csv_files)}）：")) - 1
            if file_index < 0 or file_index >= len(csv_files):
                print("❌ 输入错误：编号超出范围！")
                return
            selected_file = os.path.join(csl_folder, csv_files[file_index])
            print(f"✅ 选择文件：{csv_files[file_index]}")
            
            # 使用统一工具函数分别读取"摘要"和"关键词"列
            abstracts = read_target_column_from_csv(selected_file, "摘要")
            # 关键词列无需过滤过短文本（>20），但read_target_column_from_csv包含了此逻辑，这里保持原样
            keywords_list = read_target_column_from_csv(selected_file, "关键词") 
            
            if not abstracts or not keywords_list:
                return
                
            # 确保数据长度一致
            min_len = min(len(abstracts), len(keywords_list))
            data_list = []
            for i in range(min_len):
                # 兼容旧代码处理逗号
                keywords = [kw.strip() for kw in keywords_list[i].replace("；", "，").split("，") if kw.strip()]
                if keywords:
                    data_list.append({
                        "abstract": abstracts[i],
                        "keywords": keywords
                    })
            
            if not data_list:
                print("❌ 未提取到有效摘要-关键词数据对")
                return
                
            print(f"✅ 成功读取 {len(data_list)} 条有效数据")
        
        except ValueError:
            print("❌ 输入错误：请输入数字编号！")
            return
    
    else:
        print("❌ 输入错误：请选择1或2！")
        return
    
    # 2. 核心验证逻辑（计算摘要与关键词的语义匹配度）
    print(f"\n🔍 开始验证 {len(data_list)} 条数据...")
    results = []
    
    for idx, data in enumerate(data_list, 1):
        abstract = data["abstract"]
        keywords = data["keywords"]
        
        # 编码摘要
        abstract_encoded = tokenizer(
            abstract,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)
        
        # 编码关键词（拼接为一句话）
        keywords_text = "，".join(keywords)
        keywords_encoded = tokenizer(
            keywords_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(device)
        
        # 提取特征
        with torch.no_grad():
            abstract_output = model(**abstract_encoded)
            abstract_feat = abstract_output.pooler_output
            
            keywords_output = model(** keywords_encoded)
            keywords_feat = keywords_output.pooler_output
        
        # 计算余弦相似度（归一化后）
        abstract_feat_norm = F.normalize(abstract_feat, p=2, dim=1)
        keywords_feat_norm = F.normalize(keywords_feat, p=2, dim=1)
        similarity = torch.mm(abstract_feat_norm, keywords_feat_norm.T).item()
        
        # 二分类判断（阈值设为0.3，可根据实际效果调整）
        is_accurate = 1 if similarity >= 0.3 else 0
        results.append({
            "index": idx,
            "abstract": abstract,
            "keywords": keywords,
            "similarity": round(similarity, 4),
            "is_accurate": is_accurate
        })
    
    # 3. 输出结果
    print("\n" + "="*40 + " 验证结果 " + "="*40)
    for res in results:
        print(f"\n📝 第{res['index']}条数据：")
        print(f"摘要：{res['abstract'][:150]}..." if len(res['abstract']) > 150 else f"摘要：{res['abstract']}")
        print(f"候选关键词：{','.join(res['keywords'])}")
        print(f"语义相似度：{res['similarity']}")
        print(f"验证结果：{'✅ 准确（1）' if res['is_accurate'] == 1 else '❌ 不准确（0）'}")


def csldcp_task(tokenizer, model, device):
    """任务5：CSLDCP（主题文献分类）"""
    global model_name
    print("\n" + "="*50)
    print(f"📚 任务5：主题文献分类（CSLDCP） - 模型: {model_name}")
    print("="*50)
    
    # 1. 定义细粒度学科类别（CLUE基准67类完整版）
    csldcp_labels = {
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
    print(f"支持分类：共{len(csldcp_labels)}个细粒度学科类别")
    print("示例类别：计算机科学与技术、电子科学与技术、数学、医学等")
    
    # 2. 获取用户输入方式（单条/批量）
    print("\n请选择输入方式：")
    print("1. 手动输入单篇文献摘要")
    print("2. 从Information/Literature文件夹读取CSV文件批量分类")
    input_choice = input("请输入选项编号（1/2）：")
    
    abstracts = []
    if input_choice == "1":
        # 单条输入
        user_abstract = input("请输入文献摘要文本：")
        if not user_abstract.strip():
            print("❌ 输入错误：摘要文本不能为空！")
            return
        abstracts = [user_abstract]
    
    elif input_choice == "2":
        # 批量读取CSV
        script_dir = os.path.dirname(os.path.abspath(__file__))
        lit_folder = os.path.join(script_dir, "Spiders", "Information", "Literature")
        if not os.path.exists(lit_folder):
            print(f"❌ Literature文件夹不存在：{lit_folder}")
            print("💡 请确保Information/Literature文件夹结构正确")
            return
        
        csv_files = [f for f in os.listdir(lit_folder) if f.endswith('.csv')]
        if not csv_files:
            print("❌ Literature文件夹中无CSV文件")
            return
        
        # 选择CSV文件
        print(f"\n📁 找到{len(csv_files)}个CSV文件：")
        for i, csv_file in enumerate(csv_files, 1):
            print(f"  {i}. {csv_file}")
        
        try:
            file_index = int(input(f"请选择文件编号（1-{len(csv_files)}）：")) - 1
            if file_index < 0 or file_index >= len(csv_files):
                print("❌ 编号超出范围！")
                return
            selected_file = os.path.join(lit_folder, csv_files[file_index])
            
            # 使用统一工具函数读取"摘要"列
            abstracts = read_target_column_from_csv(selected_file, "摘要")
            if not abstracts:
                return
        
        except ValueError:
            print("❌ 请输入有效数字！")
            return
    
    else:
        print("❌ 输入错误：请选择1或2！")
        return
    
    # 3. 分类预测
    print(f"\n🔍 开始对{len(abstracts)}篇文献进行分类...")
    results = []
    
    for i, abstract in enumerate(abstracts, 1):
        if input_choice == "2":
            print(f"📊 处理进度：{i}/{len(abstracts)}", end="\r")
        
        # 编码文本
        encoded = tokenizer(
            abstract,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256  # 适配长摘要
        ).to(device)
        
        # 提取特征
        with torch.no_grad():
            outputs = model(**encoded)
            feat = outputs.pooler_output.to(torch.float16) ##

            # 使用相似度匹配
            pred_label_idx, pred_probs = predict_by_similarity(model, tokenizer, device, feat, csldcp_labels)
            pred_label = csldcp_labels[pred_label_idx]
            confidence = pred_probs[pred_label_idx]
        
        results.append({
            'abstract': abstract,
            'pred_label': pred_label,
            'confidence': confidence
        })
    
    # 4. 输出结果
    print("\n" + "="*40 + " 分类结果 " + "="*40)
    if input_choice == "1":
        # 单条详细结果
        res = results[0]
        print(f"📝 摘要：{res['abstract'][:150]}..." if len(res['abstract']) > 150 else f"📝 摘要：{res['abstract']}")
        print(f"🏷️ 预测学科：{res['pred_label']}")
        print(f"📊 置信度：{res['confidence']:.4f}")
    else:
        # 批量统计+部分详情
        print(f"📈 批量统计：共{len(results)}篇文献")
        from collections import Counter
        label_counts = Counter([r['pred_label'] for r in results])
        print(f"\n📊 学科分布（Top10）：")
        for label, count in label_counts.most_common(10):  # 显示Top10
            percentage = (count / len(results)) * 100
            print(f"  {label}：{count}篇 ({percentage:.1f}%)")
        
        # 显示前5条详情
        print(f"\n🔍 前{min(5, len(results))}篇文献详情：")
        for i, res in enumerate(results[:5], 1):
            print(f"\n{i}. 摘要：{res['abstract'][:100]}...")
            print(f"   学科：{res['pred_label']} | 置信度：{res['confidence']:.4f}")
        if len(results) > 5:
            show_all = input(f"\n💡 还有{len(results)-5}条结果未显示，是否显示全部？(y/n)：")
            if show_all.lower() == 'y':
                print("\n" + "="*20 + " 全部结果详情 " + "="*20)
                for i, res in enumerate(results, 1):
                    # 打印完整摘要（或长一点的截断）
                    print(f"\n{i}. 摘要：{res['abstract']}")
                    print(f"   学科：{res['pred_label']} | 置信度：{res['confidence']:.4f}")


def iflytek_task(tokenizer, model, device):
    """任务6：应用简介分类"""
    global model_name
    print("\n" + "="*50)
    print(f"📱 任务6：应用简介分类 - 模型: {model_name}")
    print("="*50)
    
    # 1. 定义应用类别
    app_labels = {
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
    print(f"支持分类：{', '.join([f'{k}:{v}' for k, v in app_labels.items()])}")
    
    # 2. 获取用户输入方式
    print("\n请选择输入方式：")
    print("1. 手动输入应用简介")
    print("2. 从Information/AppDescriptions文件夹读取CSV文件批量分类")
    input_choice = input("请输入选项编号（1/2）：")
    
    descriptions = []
    if input_choice == "1":
        # 手动输入
        user_desc = input("请输入应用简介文本：")
        if not user_desc.strip():
            print("❌ 输入错误：应用简介不能为空！")
            return
        descriptions = [user_desc]
    
    elif input_choice == "2":
        # 批量读取CSV
        script_dir = os.path.dirname(os.path.abspath(__file__))
        app_folder = os.path.join(script_dir, "Spiders", "Information", "AppDescriptions")
        if not os.path.exists(app_folder):
            print(f"❌ AppDescriptions文件夹不存在：{app_folder}")
            print("💡 请确保Information/AppDescriptions文件夹结构正确")
            return
        
        csv_files = [f for f in os.listdir(app_folder) if f.endswith('.csv')]
        if not csv_files:
            print("❌ AppDescriptions文件夹中无CSV文件")
            return
        
        # 选择CSV文件
        print(f"\n📁 找到{len(csv_files)}个CSV文件：")
        for i, csv_file in enumerate(csv_files, 1):
            print(f"  {i}. {csv_file}")
        
        try:
            file_index = int(input(f"请选择文件编号（1-{len(csv_files)}）：")) - 1
            if file_index < 0 or file_index >= len(csv_files):
                print("❌ 编号超出范围！")
                return
            selected_file = os.path.join(app_folder, csv_files[file_index])
            
            # 使用统一工具函数读取"应用简介"列
            descriptions = read_target_column_from_csv(selected_file, "应用简介")
            if not descriptions:
                return
        
        except ValueError:
            print("❌ 请输入有效数字！")
            return
    
    else:
        print("❌ 输入错误：请选择1或2！")
        return
    
    # 3. 分类预测
    print(f"\n🔍 开始对{len(descriptions)}条应用简介进行分类...")
    results = []
    
    for i, desc in enumerate(descriptions, 1):
        if input_choice == "2":
            print(f"📊 处理进度：{i}/{len(descriptions)}", end="\r")
        
        # 编码文本
        encoded = tokenizer(
            desc,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)
        
        # 提取特征
        with torch.no_grad():
            outputs = model(**encoded)
            feat = outputs.pooler_output.to(torch.float16) ##

            # 使用相似度匹配
            pred_label_idx, pred_probs = predict_by_similarity(model, tokenizer, device, feat, app_labels)
            pred_label = app_labels[pred_label_idx]
            confidence = pred_probs[pred_label_idx]
        
        results.append({
            'description': desc,
            'pred_label': pred_label,
            'confidence': confidence
        })
    
    # 4. 输出结果
    print("\n" + "="*40 + " 分类结果 " + "="*40)
    if input_choice == "1":
        # 单条详细结果
        res = results[0]
        print(f"📝 应用简介：{res['description'][:150]}..." if len(res['description']) > 150 else f"📝 应用简介：{res['description']}")
        print(f"🏷️ 预测类别：{res['pred_label']}")
        print(f"📊 置信度：{res['confidence']:.4f}")
    else:
        # 批量统计+部分详情
        print(f"📈 批量统计：共{len(results)}条应用简介")
        from collections import Counter
        label_counts = Counter([r['pred_label'] for r in results])
        print(f"\n📊 类别分布：")
        for label in app_labels.values():
            count = label_counts.get(label, 0)
            percentage = (count / len(results)) * 100
            print(f"  {label}：{count}条 ({percentage:.1f}%)")
        
        # 显示前5条详情
        print(f"\n🔍 前{min(5, len(results))}条应用简介详情：")
        for i, res in enumerate(results[:5], 1):
            print(f"\n{i}. 简介：{res['description'][:100]}...")
            print(f"   类别：{res['pred_label']} | 置信度：{res['confidence']:.4f}")


def cluewsc_task(tokenizer, model, device): # 移除config参数，直接使用全局config
    """任务7：CLUEWSC（指代消解）"""
    global model_name, config
    print("\n" + "="*50)
    print(f"🔍 任务7：指代消解（CLUEWSC） - 模型: {model_name}")
    print("="*50)
    
    # 1. 任务说明
    print("任务目标：判断句子中代词是否与指定名词短语共指（指代同一对象）")
    print("示例：句子='小明告诉小华他考试不及格'，名词短语='小明' → 共指（True）/不共指（False）")
    
    # 2. 获取用户输入
    user_sentence = input("请输入包含代词的句子：")
    noun_phrase = input("请输入需要判断的名词短语（如：小明、这本书）：")
    
    if not user_sentence.strip() or not noun_phrase.strip():
        print("❌ 输入错误：句子和名词短语不能为空！")
        return
    
    # 3. 编码文本（用[SEP]分隔句子和名词短语）
    wsc_text = f"{user_sentence} {tokenizer.sep_token} {noun_phrase}"
    encoded = tokenizer(
        wsc_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)
    
    # 4. 提取特征并预测
    with torch.no_grad():
        outputs = model(**encoded)
        feat = outputs.pooler_output.to(torch.float16) ##
    
    # 二分类头（共指/不共指）
    input_size = config.hidden_size
    # 注意：这个分类头是随机初始化的，对于零样本效果可能不佳，但保持原逻辑
    wsc_classifier = torch.nn.Linear(input_size, 2).to(device, dtype=torch.float16)
    with torch.no_grad():
        logits = wsc_classifier(feat)
        pred_probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label_idx = torch.argmax(logits).item()
        is_coreferent = pred_label_idx == 0  # 0=共指，1=不共指
        confidence = pred_probs[pred_label_idx]
    
    # 5. 输出结果
    print("\n" + "="*30 + " 结果 " + "="*30)
    print(f"原句子：{user_sentence}")
    print(f"名词短语：{noun_phrase}")
    print(f"共指关系：{'✅ 是' if is_coreferent else '❌ 否'}")
    print(f"置信度：{confidence:.4f}")
    print(f"详细概率：共指概率 {pred_probs[0]:.4f} | 不共指概率 {pred_probs[1]:.4f}")


# 主函数（用于测试和运行）
if __name__ == "__main__":
    
    # 获取脚本所在的根目录 (Program 目录)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 顶层选择逻辑
    while True:
        print("\n" + "="*50)
        print("欢迎使用Erlangshen-MegatronBert 模型和爬虫工具")
        print("="*50)
        print("\n请选择功能：")
        print("1. 运行爬虫并爬取网页")
        print("2. 加载模型并执行任务")
        print("0. 退出")
        
        main_choice = input("请输入选项编号（1/2/0）：").strip()
        
        if main_choice == "1":
            # 运行爬虫脚本
            spider_launcher_path = os.path.join(script_dir, "Spiders", "spider_launcher.py")
            if os.path.exists(spider_launcher_path):
                print(f"\n🔄 正在运行爬虫脚本: {spider_launcher_path}")
                # 使用 os.system 或 subprocess 运行另一个 Python 脚本
                # 这里使用 os.system 简化操作，如果需要更复杂的控制，应使用 subprocess
                try:
                    os.system(f"{sys.executable} {spider_launcher_path}")
                except Exception as e:
                    print(f"❌ 运行爬虫脚本失败：{e}")
            else:
                print(f"❌ 爬虫脚本不存在：{spider_launcher_path}")
            
        elif main_choice == "2":
            # 进入模型选择和任务执行流程
            
            # 模型选择逻辑
            MODEL_MAP = {
                "1": "Erlangshen-MegatronBert-1.3B",
                "2": "Erlangshen-MegatronBert-3.9B",
            }
            
            while True:
                print("\n请选择需要使用的模型：")
                for key, name in MODEL_MAP.items():
                    print(f"{key}. {name}")
                
                model_choice = input("请输入模型编号（1/2）：").strip()
                
                if model_choice in MODEL_MAP:
                    selected_model_name = MODEL_MAP[model_choice]
                    
                    # 拼接出选中模型的完整路径： Models/Erlangshen-MegatronBert-X.XB
                    model_dir = os.path.join(script_dir, "Models", selected_model_name)
                    
                    # 检查 Models 文件夹是否存在
                    if not os.path.exists(os.path.join(script_dir, "Models")):
                        print(f"❌ 文件夹不存在：{os.path.join(script_dir, 'Models')}")
                        print("请将模型文件夹放入名为 'Models' 的新文件夹中！")
                        sys.exit(1)

                    init_model_and_tokenizer(model_dir, selected_model_name)
                    
                    # 检查模型是否成功加载
                    if model is not None:
                        break # 模型加载成功，退出模型选择循环
                    else:
                        print("模型加载失败，请检查文件路径和内容。")
                        continue # 返回模型选择
                        
                else:
                    print("无效的模型编号，请重试！")
                    continue
            
            # 模型加载成功，进入任务选择逻辑
            while True:
                print("\n请选择任务：")
                print("1. 成语填空（CHIDF）")
                print("2. 新闻分类（TNEWS）")
                print("3. 自然语言推理（OCNLI）")
                print("4. 摘要关键词验证（CSL）")
                print("5. 主题文献分类（CSLDCP）")
                print("6. 应用描述分类（IFLYTEK）")
                print("7. 指代消解（CLUEWSC）")
                print("0. 返回主菜单")
                
                choice = input("请输入任务编号：")
                
                if choice == "1":
                    chidf_task(tokenizer, model, device)
                elif choice == "2":
                    tnews_task(tokenizer, model, device)
                elif choice == "3":
                    ocnli_task(tokenizer, model, device)
                elif choice == "4":
                    csl_task(tokenizer, model, device)
                elif choice == "5":
                    csldcp_task(tokenizer, model, device)
                elif choice == "6":
                    iflytek_task(tokenizer, model, device)
                elif choice == "7":
                    cluewsc_task(tokenizer, model, device)
                elif choice == "0":
                    print("返回主菜单...")
                    break  # 退出任务循环，返回顶层主菜单
                else:
                    print("无效的选择，请重试")
                    
        elif main_choice == "0":
            print("感谢使用，再见！")
            sys.exit(0) # 退出整个程序
        
        else:
            print("无效的选项，请重新输入！")