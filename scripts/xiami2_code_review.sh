#!/bin/zsh
# xiami-2 Task #1 代码审查脚本

echo "【xiami-2 代码审查 - Task #1 脚本分析】"
echo ""

# 读取脚本
SCRIPT_PATH="/Users/nanzhang/rocket2/train/collect_grounding_data.py"

echo "=== 第1部分：导入和初始化 ==="
head -50 $SCRIPT_PATH | grep -E "^import|^from|class|def " | head -20
echo ""

echo "=== 第2部分：VPT 模型加载 ==="
grep -A 5 "VPTPolicy" $SCRIPT_PATH | head -10
echo ""

echo "=== 第3部分：事件监控核心逻辑 ==="
grep -B 3 -A 10 "mine_block\|BUFFER_SIZE\|save.*offset" $SCRIPT_PATH | head -40
echo ""

echo "=== 第4部分：主循环结构 ==="
grep -B 2 -A 5 "def main\|for step\|if mine_block" $SCRIPT_PATH | head -50
echo ""

echo "=== 第5部分：数据保存 ==="
grep -B 2 -A 5 "cv2.imwrite\|save\|annotations" $SCRIPT_PATH | head -30
echo ""

echo "【结论】"
echo "xiami-2 正在分析脚本..."
echo "1. VPT 是否真的在执行动作?"
echo "2. mine_block 事件计数是否工作?"
echo "3. 为什么没有 PNG 被保存?"
