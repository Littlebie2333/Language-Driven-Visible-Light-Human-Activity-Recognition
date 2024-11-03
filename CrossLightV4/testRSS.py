import os
import re
import logging
from RSSNet import TCNencoder
from GPT2 import TextGenerator
import torch
from dataset import Load_data
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载器
student_root = r'G:\CrosslightData705\test'
data_loader = Load_data(student_root, batch_size=1, device=device, shuffle=False)

# 模型权重文件夹
weights_folder = r'C:\Users\Lenovo\Desktop\111'

# 日志文件存储文件夹
log_folder = 'log_results'
os.makedirs(log_folder, exist_ok=True)

# 汇总文件
summary_file = 'summary_metrics.txt'

# 记录所有日志文件中的准确率和分数
metrics_records = []

# 创建ROUGE评估器
rouge = Rouge()
# 初始化最高得分和对应的权重文件
highest_accuracy = 0.0
highest_bleu1 = 0.0
highest_bleu2 = 0.0
highest_bleu3 = 0.0
highest_rouge1_f = 0.0
highest_rouge2_f = 0.0
highest_rouge3_f = 0.0
highest_rouge1_p = 0.0
highest_rouge2_p = 0.0
highest_rouge3_p = 0.0
highest_rouge1_r = 0.0
highest_rouge2_r = 0.0
highest_rouge3_r = 0.0
best_weight_file = None
best_bleu1_weight_file = None
best_bleu2_weight_file = None
best_bleu3_weight_file = None
best_rouge1_weight_file = None
best_rouge2_weight_file = None
best_rouge3_weight_file = None

# 遍历文件夹中的所有权重文件
for weight_file in os.listdir(weights_folder):
    if weight_file.endswith('.pt'):
        weight_path = os.path.join(weights_folder, weight_file)

        # 创建独立的日志记录器
        logger = logging.getLogger(weight_file)
        logger.setLevel(logging.INFO)

        # 创建文件处理器
        log_filename = os.path.join(log_folder, f'{weight_file}_test_results.txt')
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 创建格式化器并添加到处理器
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 将处理器添加到日志记录器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.info(f"Processing weight file: {weight_file}")

        # 加载模型
        RSS_net = TCNencoder().to(device)
        GPT2 = TextGenerator().to(device)
        RSS_net.load_state_dict(torch.load(weight_path))

        try:
            total_correct = 0
            total_samples = 0
            total_bleu1 = 0.0
            total_bleu2 = 0.0
            total_bleu3 = 0.0
            total_rouge1_f = 0.0
            total_rouge2_f = 0.0
            total_rouge3_f = 0.0
            total_rouge1_p = 0.0
            total_rouge2_p = 0.0
            total_rouge3_p = 0.0
            total_rouge1_r = 0.0
            total_rouge2_r = 0.0
            total_rouge3_r = 0.0
            num_batches = len(data_loader)

            for batch_idx, (rss_batch, labels_text, _) in enumerate(data_loader):
                logger.info(f"----------------Batch {batch_idx + 1}-------------------")
                logger.info(f"label_text: {labels_text}")

                # 处理真实标签
                labels = [GPT2.tokenizer(label_text + GPT2.eos_token, return_tensors="pt").input_ids.squeeze(0) for
                          label_text in labels_text]
                labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True).to(device).long()
                logger.info(f"label_id: {labels}")

                # 提取特征
                rss_feature = RSS_net(rss_batch.to(device))
                attention_mask = (rss_feature.sum(dim=-1) != 0).long().to(device)

                # 使用训练好的模型生成输出
                with torch.no_grad():
                    output = GPT2.forward(inputs_embeds=rss_feature, attention_mask=attention_mask)
                    generated_labels = output.logits.argmax(dim=-1)  # 获取每个位置的最可能的标签

                # 找到第一个 50256 的位置并截取
                if (generated_labels == 50256).any():
                    first_50256_pos = (generated_labels == 50256).nonzero(as_tuple=True)[1][0].item()
                    generated_labels = generated_labels[:, :first_50256_pos + 1]
                else:
                    first_50256_pos = generated_labels.size(1)

                logger.info(f"first_50256_pos: {first_50256_pos}")

                # 将截取后的 token ID 转换为文本
                generated_text = GPT2.tokenizer.decode(generated_labels[0], skip_special_tokens=True)
                logger.info(f"Generated IDs: {generated_labels}")
                logger.info(f"Generated text: {generated_text}")

                # 检查生成文本和真实标签是否为空
                if not generated_text or not labels_text[0]:
                    logger.warning(
                        f"Skipping ROUGE calculation due to empty text: Generated text: '{generated_text}', Reference text: '{labels_text[0]}'")
                    # 设置ROUGE分数为0
                    total_rouge1_f += 0.0
                    total_rouge2_f += 0.0
                    total_rouge3_f += 0.0
                    total_rouge1_p += 0.0
                    total_rouge2_p += 0.0
                    total_rouge3_p += 0.0
                    total_rouge1_r += 0.0
                    total_rouge2_r += 0.0
                    total_rouge3_r += 0.0
                    continue  # 跳过ROUGE计算并继续下一个批次

                # 计算准确率，确保输入长度一致
                true_tokens = labels_text[0].split()
                pred_tokens = generated_text.split()
                correct = int(true_tokens == pred_tokens)  # 完全匹配为 1，不匹配为 0
                total_correct += correct
                total_samples += 1
                logger.info(f"Accuracy: {correct}")

                # 计算 BLEU 分数
                reference = [true_tokens]
                hypothesis = pred_tokens
                bleu1 = sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0))
                bleu2 = sentence_bleu(reference, hypothesis, weights=(0.5, 0.5, 0, 0))
                bleu3 = sentence_bleu(reference, hypothesis, weights=(0.33, 0.33, 0.33, 0))
                total_bleu1 += bleu1
                total_bleu2 += bleu2
                total_bleu3 += bleu3
                logger.info(f"BLEU-1: {bleu1:.4f}")
                logger.info(f"BLEU-2: {bleu2:.4f}")
                logger.info(f"BLEU-3: {bleu3:.4f}")

                # 计算 ROUGE 分数
                rouge_scores = rouge.get_scores(generated_text, labels_text[0])
                total_rouge1_f += rouge_scores[0]['rouge-1']['f']
                total_rouge2_f += rouge_scores[0]['rouge-2']['f']
                total_rouge3_f += rouge_scores[0]['rouge-l']['f']
                total_rouge1_p += rouge_scores[0]['rouge-1']['p']
                total_rouge2_p += rouge_scores[0]['rouge-2']['p']
                total_rouge3_p += rouge_scores[0]['rouge-l']['p']
                total_rouge1_r += rouge_scores[0]['rouge-1']['r']
                total_rouge2_r += rouge_scores[0]['rouge-2']['r']
                total_rouge3_r += rouge_scores[0]['rouge-l']['r']
                logger.info(f"ROUGE-1 F1: {rouge_scores[0]['rouge-1']['f']:.4f}")
                logger.info(f"ROUGE-1 Precision: {rouge_scores[0]['rouge-1']['p']:.4f}")
                logger.info(f"ROUGE-1 Recall: {rouge_scores[0]['rouge-1']['r']:.4f}")
                logger.info(f"ROUGE-2 F1: {rouge_scores[0]['rouge-2']['f']:.4f}")
                logger.info(f"ROUGE-2 Precision: {rouge_scores[0]['rouge-2']['p']:.4f}")
                logger.info(f"ROUGE-2 Recall: {rouge_scores[0]['rouge-2']['r']:.4f}")
                logger.info(f"ROUGE-L F1: {rouge_scores[0]['rouge-l']['f']:.4f}")
                logger.info(f"ROUGE-L Precision: {rouge_scores[0]['rouge-l']['p']:.4f}")
                logger.info(f"ROUGE-L Recall: {rouge_scores[0]['rouge-l']['r']:.4f}")

            # 计算总体准确率和平均分
            overall_accuracy = total_correct / total_samples
            avg_bleu1 = total_bleu1 / total_samples
            avg_bleu2 = total_bleu2 / total_samples
            avg_bleu3 = total_bleu3 / total_samples
            avg_rouge1_f = total_rouge1_f / total_samples
            avg_rouge2_f = total_rouge2_f / total_samples
            avg_rouge3_f = total_rouge3_f / total_samples
            avg_rouge1_p = total_rouge1_p / total_samples
            avg_rouge2_p = total_rouge2_p / total_samples
            avg_rouge3_p = total_rouge3_p / total_samples
            avg_rouge1_r = total_rouge1_r / total_samples
            avg_rouge2_r = total_rouge2_r / total_samples
            avg_rouge3_r = total_rouge3_r / total_samples

            metrics_info = (f"Overall Accuracy for {weight_file}: {overall_accuracy:.4f}\n"
                            f"Average BLEU-1: {avg_bleu1:.4f}\n"
                            f"Average BLEU-2: {avg_bleu2:.4f}\n"
                            f"Average BLEU-3: {avg_bleu3:.4f}\n"
                            f"Average ROUGE-1 F1: {avg_rouge1_f:.4f}\n"
                            f"Average ROUGE-1 Precision: {avg_rouge1_p:.4f}\n"
                            f"Average ROUGE-1 Recall: {avg_rouge1_r:.4f}\n"
                            f"Average ROUGE-2 F1: {avg_rouge2_f:.4f}\n"
                            f"Average ROUGE-2 Precision: {avg_rouge2_p:.4f}\n"
                            f"Average ROUGE-2 Recall: {avg_rouge2_r:.4f}\n"
                            f"Average ROUGE-L F1: {avg_rouge3_f:.4f}\n"
                            f"Average ROUGE-L Precision: {avg_rouge3_p:.4f}\n"
                            f"Average ROUGE-L Recall: {avg_rouge3_r:.4f}")
            logger.info(metrics_info)
            logger.info(f"correct::total_sample: {total_correct}::{total_samples}")

            # 记录准确率和分数信息
            metrics_records.append(metrics_info)

            # 更新最高得分和对应的权重文件
            if overall_accuracy > highest_accuracy:
                highest_accuracy = overall_accuracy
                best_weight_file = weight_file
            if avg_bleu1 > highest_bleu1:
                highest_bleu1 = avg_bleu1
                best_bleu1_weight_file = weight_file
            if avg_bleu2 > highest_bleu2:
                highest_bleu2 = avg_bleu2
                best_bleu2_weight_file = weight_file
            if avg_bleu3 > highest_bleu3:
                highest_bleu3 = avg_bleu3
                best_bleu3_weight_file = weight_file
            if avg_rouge1_f > highest_rouge1_f:
                highest_rouge1_f = avg_rouge1_f
                best_rouge1_weight_file = weight_file
            if avg_rouge2_f > highest_rouge2_f:
                highest_rouge2_f = avg_rouge2_f
                best_rouge2_weight_file = weight_file
            if avg_rouge3_f > highest_rouge3_f:
                highest_rouge3_f = avg_rouge3_f
                best_rouge3_weight_file = weight_file
            if avg_rouge1_p > highest_rouge1_p:
                highest_rouge1_p = avg_rouge1_p
                best_rouge1_p_weight_file = weight_file
            if avg_rouge2_p > highest_rouge2_p:
                highest_rouge2_p = avg_rouge2_p
                best_rouge2_p_weight_file = weight_file
            if avg_rouge3_p > highest_rouge3_p:
                highest_rouge3_p = avg_rouge3_p
                best_rouge3_p_weight_file = weight_file
            if avg_rouge1_r > highest_rouge1_r:
                highest_rouge1_r = avg_rouge1_r
                best_rouge1_r_weight_file = weight_file
            if avg_rouge2_r > highest_rouge2_r:
                highest_rouge2_r = avg_rouge2_r
                best_rouge2_r_weight_file = weight_file
            if avg_rouge3_r > highest_rouge3_r:
                highest_rouge3_r = avg_rouge3_r
                best_rouge3_r_weight_file = weight_file

        except FileNotFoundError as e:
            logger.error(f"Error: {e}")

        # 移除处理器
        logger.removeHandler(file_handler)
        logger.removeHandler(console_handler)

# 将所有分数信息写入汇总文件
with open(summary_file, 'w') as f:
    for record in metrics_records:
        f.write(record + '\n')

    # 写入最高分数的信息
    f.write(f'\nHighest Overall Accuracy: {highest_accuracy:.4f} for {best_weight_file}\n')
    f.write(f'Highest BLEU-1: {highest_bleu1:.4f} for {best_bleu1_weight_file}\n')
    f.write(f'Highest BLEU-2: {highest_bleu2:.4f} for {best_bleu2_weight_file}\n')
    f.write(f'Highest BLEU-3: {highest_bleu3:.4f} for {best_bleu3_weight_file}\n')
    f.write(f'Highest ROUGE-1 F1: {highest_rouge1_f:.4f} for {best_rouge1_weight_file}\n')
    f.write(f'Highest ROUGE-2 F1: {highest_rouge2_f:.4f} for {best_rouge2_weight_file}\n')
    f.write(f'Highest ROUGE-L F1: {highest_rouge3_f:.4f} for {best_rouge3_weight_file}\n')
    f.write(f'Highest ROUGE-1 Precision: {highest_rouge1_p:.4f} for {best_rouge1_weight_file}\n')
    f.write(f'Highest ROUGE-2 Precision: {highest_rouge2_p:.4f} for {best_rouge2_weight_file}\n')
    f.write(f'Highest ROUGE-L Precision: {highest_rouge3_p:.4f} for {best_rouge3_weight_file}\n')
    f.write(f'Highest ROUGE-1 Recall: {highest_rouge1_r:.4f} for {best_rouge1_weight_file}\n')
    f.write(f'Highest ROUGE-2 Recall: {highest_rouge2_r:.4f} for {best_rouge2_weight_file}\n')
    f.write(f'Highest ROUGE-L Recall: {highest_rouge3_r:.4f} for {best_rouge3_weight_file}\n')


