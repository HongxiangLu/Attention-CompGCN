#!/bin/bash
set -e

# 压缩包名称
file="result-$(date "+%Y%m%d_%H%M%S").zip"
# 把 result 目录做成 zip 压缩包
zip -q -r "${file}" log checkpoints stdout
# 通过 oss 上传到个人数据中的 backup 文件夹中
oss cp "${file}" oss://backup/
rm "${file}"
