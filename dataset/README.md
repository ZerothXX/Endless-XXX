# 本地角色数据布局

此目录只保存本机角色素材和角色卡；Git 忽略其内容。请不要提交未获授权的参考图或生成结果。

```text
dataset/<role>/
├── ckr/                         # 角色外观参考：视图图、脸部特写、角色插画等
├── ckw/                         # 设计语言参考：饰品、纹样、材质或配色证据
└── card/
    ├── current.json             # 当前版本指针
    └── versions/<digest>/
        ├── card.json            # 角色事实、设计约束、参考 profile
        ├── source_assets/       # 指向本机原始素材并记录哈希
        └── derived_assets/      # 自动提取信息与对应 provenance
```

人物自动入口需要 `card.json` 中的 `profiles.human_complete`；物品/动物入口需要 `profiles.material_isolated` 及其配色和设计特征。引用路径必须能在本机读取，内容哈希须与卡片记录一致。素材改变后建立新的卡版本，不要编辑旧版本或挪用过期哈希。

公开工作流仓库未包含视觉分析与服装区域分割 worker，也不携带任何角色卡。请在本地完成卡片准备后再调用主入口。

