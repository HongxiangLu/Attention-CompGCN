#!/bin/bash
set -e

# 压缩包名称
file="result-$(date "+%Y%m%d_%H%M%S").zip"
# 把 result 目录做成 zip 压缩包
zip -q -r "${file}" log checkpoints stdout
# 通过 oss 上传到个人数据中的 backup 文件夹中
oss cp "${file}" oss://backup/
rm "${file}"

# 发送电子邮件（需要安装 mail 命令）
mail -s "CompGCN 运行完成" lhx0157@email.swu.edu.cn

# 传输成功后关机
shutdown