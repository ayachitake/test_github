# 运行规则
- 在指定的conda环境中运行
- 指定的conda环境为`translate_akkadian`，若未激活则自动激活，若不存在则自动创建
- 运行notebook时，python内核选择`translate_akkadian`

# 交互规则
- 所有交互均在指定的conda环境中进行
- 使用中文进行交互

# 代码规则
- 使用powershell代码时，`&&`连接符会报错，使用`;`替代
- 使用相对路径，且注意路径分隔符为`/`，而不是`\`
- 编写的临时测试脚本，均放在`./temp_scripts/`目录下