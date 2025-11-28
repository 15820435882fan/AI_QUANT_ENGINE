#!/bin/bash
echo "🚀 开始上传代码到GitHub..."
git add .
git commit -m "自动提交: $(date '+%Y-%m-%d %H:%M:%S')"
git push origin main
echo "✅ 代码上传完成!"