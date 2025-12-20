#!/bin/bash
set -e

ENV=$1
case "$ENV" in
  "prod")
    TARGET="production"
    # 同时执行 CI 构建和 CD 部署
    TAGS="ci,deploy"
    ;;
  "prod-quick")
    TARGET="production"
    # 跳过构建，仅部署 (假设镜像已存在或只改了配置)
    TAGS="deploy"
    ;;
  *)
    echo "Usage: $0 {prod|prod-quick}"
    exit 1
    ;;
esac

# 必须包含 localhost 以执行 CI 任务
ansible-playbook -i ansible/inventory.yml ansible/playbook.yml \
  --limit "localhost,$TARGET" \
  --tags "$TAGS" \
  -e "target_host=$TARGET" \
  -e "ci_project_name=rvc-web" \
  -e "ci_image_name=app"
