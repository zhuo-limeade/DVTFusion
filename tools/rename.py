import torch

# 假设你的 .pth 文件名为 'old_weights.pth'
old_weights = torch.load('./tools/resnet15_121_pretrain.pth')

# 创建一个新的权重字典
new_weights = {}

# 遍历原始权重字典
for old_key in old_weights['state_dict']:
    # 在这里定义你的重命名规则
    new_key = old_key[9:]  # 示例：替换层的名字
    if  'head.fc' in old_key:
        print('yes')
        break
    new_weights[new_key] = old_weights['state_dict'][old_key]

# 保存新的权重文件
torch.save(new_weights, './tools/resnet15_121_pretrain_mod.pth')

print("权重重命名并保存成功！")
