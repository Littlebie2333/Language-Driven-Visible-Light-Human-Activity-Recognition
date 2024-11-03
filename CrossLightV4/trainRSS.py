import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import Load_data
from RSSNet import TCNencoder
from GPT2 import TextGenerator
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

student_root = r'G:\CrosslightData705\person2\TokenData3s'
data_loader = Load_data(student_root, batch_size=24, device=device, shuffle=False)

# 模型保存路径
save_folder = 'output'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

RSS_net = TCNencoder().to(device)
GPT2 = TextGenerator()

optimizer_RSS = optim.Adam(RSS_net.parameters(), lr=0.001, weight_decay=1e-5)
scheduler_RSS = optim.lr_scheduler.StepLR(optimizer_RSS, step_size=20, gamma=0.8)

custom_pad_token_id = 50257  # 50257是GPT-2词汇表中不存在的ID


# 截断到第一个 <EOS> 标记的函数
def truncate_at_eos(pred_ids, eos_token_id):
    truncated_ids = []
    for ids in pred_ids:
        if eos_token_id in ids:
            truncated_ids.append(ids[:ids.index(eos_token_id) + 1])
        else:
            truncated_ids.append(ids)
    return truncated_ids


print("---------------START-------------")
best_loss = float('inf')

# 训练循环300
for epoch in range(300):
    start_time = time.time()  # 记录当前epoch开始的时间
    total_match_loss = 0.0
    for batch_idx, (data, labels_text, _) in enumerate(data_loader):
        data = data.to(device).float()

        # prompts = GPT2.tokenizer(labels_text, return_tensors="pt").input_ids.to(device)
        # prompts_embeds = GPT2.model.transformer.wte(prompts).to(device)
        # 将 labels_text 转换为 token IDs

        labels = [GPT2.tokenizer(label_text + GPT2.eos_token, return_tensors="pt").input_ids.squeeze(0) for label_text
                  in labels_text]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True).to(device).long()
        # 创建 attention mask
        # attention_mask = (labels != GPT2.tokenizer.pad_token_id).long()

        rss_feature = RSS_net(data).float()
        # rss_feature = RSS_net(data).to(dtype=torch.float16)

        attention_mask = (rss_feature.sum(dim=-1) != 0).long().to(device)
        # rss_prompts = torch.cat((prompts_embeds, rss_feature), dim=1)

        # 获取特征提取器的输出序列长度
        output_seq_len = rss_feature.size(1)

        # 确保 labels 和 attention_mask 的长度与 rss_feature 的序列长度一致
        if labels.size(1) != output_seq_len:
            labels = torch.nn.functional.pad(labels, (0, output_seq_len - labels.size(1)), "constant",
                                             custom_pad_token_id)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, output_seq_len - attention_mask.size(1)),
                                                     "constant", 0)

        # 确保 GPT-2 模型的参数不被更新
        for param in GPT2.parameters():
            param.requires_grad = False

        # 进行前向传播，计算 logits
        output = GPT2.forward(inputs_embeds=rss_feature, attention_mask=attention_mask)
        logits = output.logits
        # 调整 logits 和 labels 的形状
        logits = logits.view(-1, logits.size(-1))  # (batch_size * sequence_length, vocab_size)
        labels_padded = labels.view(-1)  # (batch_size * sequence_length)

        # 计算交叉熵损失，忽略填充值
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=custom_pad_token_id)
        match_loss = loss_fn(logits, labels_padded)
        # output = GPT2.forward(inputs_embeds=rss_feature, attention_mask=attention_mask, labels=labels)
        # match_loss = output.loss  # 直接使用 GPT2 的损失

        predicted_ids = torch.argmax(logits, dim=-1).view(-1, output_seq_len)
        # 截断预测的序列
        truncated_preds = truncate_at_eos(predicted_ids.tolist(), GPT2.eos_token_id)


        optimizer_RSS.zero_grad()
        match_loss.backward()
        optimizer_RSS.step()

        total_match_loss += match_loss.item()

    scheduler_RSS.step()  # 在每个 epoch 结束后调用调度器

    end_time = time.time()  # 记录当前epoch结束的时间
    epoch_time = end_time - start_time  # 计算当前epoch所需时间

    # 打印每个epoch的平均损失和所需时间
    print(f"Epoch {epoch + 1}")
    print(f"Total_MatchLoss: {total_match_loss / len(data_loader)}, Time: {epoch_time} seconds")
    print(f"{''}")

    # 保存最优模型
    if total_match_loss / len(data_loader) < best_loss:
        best_loss = total_match_loss / len(data_loader)
        save_path = os.path.join(save_folder, f"gpt2_pd4_light4_best.pt")
        torch.save(RSS_net.state_dict(), save_path)
        print(f'Model saved with best loss {best_loss:.4f}')

    # 每隔20个epoch保存一次模型
    if (epoch + 1) % 20 == 0:
        save_path = os.path.join(save_folder, f"gpt2_pd4_light4_epoch_{epoch + 1}.pt")
        torch.save(RSS_net.state_dict(), save_path)
        print(f'Model saved at epoch {epoch + 1}')

