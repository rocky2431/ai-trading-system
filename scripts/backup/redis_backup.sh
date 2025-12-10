#!/bin/bash
#
# Redis 备份脚本
# 支持 RDB 快照和 AOF 备份
#
# 使用方法:
#   ./redis_backup.sh [rdb|aof|both]
#
# 环境变量:
#   REDIS_HOST - Redis 主机 (默认: localhost)
#   REDIS_PORT - Redis 端口 (默认: 6379)
#   REDIS_PASSWORD - Redis 密码 (可选)
#   BACKUP_DIR - 备份目录 (默认: /var/backups/redis)
#   RETENTION_DAYS - 保留天数 (默认: 7)
#   S3_BUCKET - S3 存储桶 (可选)

set -euo pipefail

# ==================== 配置 ====================
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/redis}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
BACKUP_TYPE="${1:-both}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Redis CLI 命令
if [[ -n "$REDIS_PASSWORD" ]]; then
    REDIS_CLI="redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD"
else
    REDIS_CLI="redis-cli -h $REDIS_HOST -p $REDIS_PORT"
fi

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
    exit 1
}

# ==================== 前置检查 ====================
check_prerequisites() {
    log "检查前置条件..."

    # 检查 redis-cli
    if ! command -v redis-cli &> /dev/null; then
        error "redis-cli 未安装"
    fi

    # 检查 Redis 连接
    if ! $REDIS_CLI PING &> /dev/null; then
        error "无法连接到 Redis"
    fi

    # 创建备份目录
    mkdir -p "$BACKUP_DIR"/{rdb,aof}

    log "前置检查完成"
}

# ==================== RDB 备份 ====================
rdb_backup() {
    log "开始 RDB 备份..."

    # 触发 BGSAVE
    $REDIS_CLI BGSAVE

    # 等待 BGSAVE 完成
    log "等待 BGSAVE 完成..."
    while [[ $($REDIS_CLI LASTSAVE) == $($REDIS_CLI LASTSAVE) ]]; do
        sleep 1
    done

    # 获取 RDB 文件路径
    REDIS_DIR=$($REDIS_CLI CONFIG GET dir | tail -1)
    REDIS_DBFILENAME=$($REDIS_CLI CONFIG GET dbfilename | tail -1)
    RDB_FILE="$REDIS_DIR/$REDIS_DBFILENAME"

    # 复制 RDB 文件
    BACKUP_FILE="$BACKUP_DIR/rdb/redis_${TIMESTAMP}.rdb"
    cp "$RDB_FILE" "$BACKUP_FILE"

    # 压缩
    gzip "$BACKUP_FILE"
    BACKUP_FILE="${BACKUP_FILE}.gz"

    # 记录元数据
    cat > "$BACKUP_DIR/rdb/redis_${TIMESTAMP}.meta" << EOF
{
    "type": "rdb",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "file": "$BACKUP_FILE",
    "size": $(stat -f%z "$BACKUP_FILE" 2>/dev/null || stat -c%s "$BACKUP_FILE"),
    "checksum": "$(sha256sum "$BACKUP_FILE" | cut -d' ' -f1)"
}
EOF

    log "RDB 备份完成: $BACKUP_FILE"
    echo "$BACKUP_FILE"
}

# ==================== AOF 备份 ====================
aof_backup() {
    log "开始 AOF 备份..."

    # 触发 AOF 重写 (如果启用)
    AOF_ENABLED=$($REDIS_CLI CONFIG GET appendonly | tail -1)

    if [[ "$AOF_ENABLED" != "yes" ]]; then
        log "AOF 未启用，跳过 AOF 备份"
        return
    fi

    # 触发 BGREWRITEAOF
    $REDIS_CLI BGREWRITEAOF

    # 等待完成
    log "等待 AOF 重写完成..."
    while [[ $($REDIS_CLI INFO persistence | grep aof_rewrite_in_progress | cut -d: -f2 | tr -d '\r') == "1" ]]; do
        sleep 1
    done

    # 获取 AOF 文件路径
    REDIS_DIR=$($REDIS_CLI CONFIG GET dir | tail -1)
    AOF_FILENAME=$($REDIS_CLI CONFIG GET appendfilename | tail -1)
    AOF_FILE="$REDIS_DIR/$AOF_FILENAME"

    # 复制 AOF 文件
    BACKUP_FILE="$BACKUP_DIR/aof/redis_${TIMESTAMP}.aof"
    cp "$AOF_FILE" "$BACKUP_FILE"

    # 压缩
    gzip "$BACKUP_FILE"
    BACKUP_FILE="${BACKUP_FILE}.gz"

    # 记录元数据
    cat > "$BACKUP_DIR/aof/redis_${TIMESTAMP}.meta" << EOF
{
    "type": "aof",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "file": "$BACKUP_FILE",
    "size": $(stat -f%z "$BACKUP_FILE" 2>/dev/null || stat -c%s "$BACKUP_FILE"),
    "checksum": "$(sha256sum "$BACKUP_FILE" | cut -d' ' -f1)"
}
EOF

    log "AOF 备份完成: $BACKUP_FILE"
    echo "$BACKUP_FILE"
}

# ==================== 上传到 S3 ====================
upload_to_s3() {
    local file="$1"

    if [[ -n "${S3_BUCKET:-}" ]]; then
        log "上传到 S3: $S3_BUCKET"
        aws s3 cp "$file" "s3://$S3_BUCKET/redis/$(basename "$file")"
        log "S3 上传完成"
    fi
}

# ==================== 清理旧备份 ====================
cleanup_old_backups() {
    log "清理 $RETENTION_DAYS 天前的备份..."

    find "$BACKUP_DIR/rdb" -name "*.gz" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR/rdb" -name "*.meta" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR/aof" -name "*.gz" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR/aof" -name "*.meta" -mtime +$RETENTION_DAYS -delete

    log "清理完成"
}

# ==================== 发送通知 ====================
send_notification() {
    local status="$1"
    local message="$2"

    if [[ -n "${TELEGRAM_BOT_TOKEN:-}" && -n "${TELEGRAM_CHAT_ID:-}" ]]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d chat_id="${TELEGRAM_CHAT_ID}" \
            -d text="🔴 Redis 备份 $status: $message" \
            -d parse_mode="HTML" > /dev/null || true
    fi
}

# ==================== 主函数 ====================
main() {
    log "========== Redis 备份开始 =========="
    log "备份类型: $BACKUP_TYPE"

    check_prerequisites

    case "$BACKUP_TYPE" in
        rdb)
            backup_file=$(rdb_backup)
            upload_to_s3 "$backup_file"
            ;;
        aof)
            backup_file=$(aof_backup)
            [[ -n "$backup_file" ]] && upload_to_s3 "$backup_file"
            ;;
        both)
            rdb_file=$(rdb_backup)
            upload_to_s3 "$rdb_file"

            aof_file=$(aof_backup)
            [[ -n "$aof_file" ]] && upload_to_s3 "$aof_file"
            ;;
        *)
            error "未知的备份类型: $BACKUP_TYPE (支持: rdb, aof, both)"
            ;;
    esac

    cleanup_old_backups

    send_notification "成功" "备份类型: $BACKUP_TYPE"

    log "========== Redis 备份完成 =========="
}

# 执行
main
