#!/bin/bash
#
# TimescaleDB 备份脚本
# 支持增量备份 (hourly) 和全量备份 (daily)
#
# 使用方法:
#   ./timescaledb_backup.sh [incremental|full]
#
# 环境变量:
#   PGHOST - PostgreSQL 主机 (默认: localhost)
#   PGPORT - PostgreSQL 端口 (默认: 5432)
#   PGUSER - PostgreSQL 用户 (默认: iqfmp)
#   PGPASSWORD - PostgreSQL 密码
#   PGDATABASE - 数据库名称 (默认: iqfmp)
#   BACKUP_DIR - 备份目录 (默认: /var/backups/timescaledb)
#   RETENTION_DAYS - 保留天数 (默认: 7)
#   S3_BUCKET - S3 存储桶 (可选)

set -euo pipefail

# ==================== 配置 ====================
PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-iqfmp}"
PGDATABASE="${PGDATABASE:-iqfmp}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/timescaledb}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
BACKUP_TYPE="${1:-incremental}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATE=$(date +%Y%m%d)

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

    # 检查 pg_dump
    if ! command -v pg_dump &> /dev/null; then
        error "pg_dump 未安装"
    fi

    # 检查数据库连接
    if ! PGPASSWORD="${PGPASSWORD}" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -c "SELECT 1" &> /dev/null; then
        error "无法连接到数据库"
    fi

    # 创建备份目录
    mkdir -p "$BACKUP_DIR"/{full,incremental,wal}

    log "前置检查完成"
}

# ==================== 全量备份 ====================
full_backup() {
    log "开始全量备份..."

    BACKUP_FILE="$BACKUP_DIR/full/iqfmp_full_${TIMESTAMP}.sql.gz"

    # 使用 pg_dump 进行全量备份
    PGPASSWORD="${PGPASSWORD}" pg_dump \
        -h "$PGHOST" \
        -p "$PGPORT" \
        -U "$PGUSER" \
        -d "$PGDATABASE" \
        -F c \
        -f "$BACKUP_DIR/full/iqfmp_full_${TIMESTAMP}.dump"

    # 压缩备份
    gzip -c "$BACKUP_DIR/full/iqfmp_full_${TIMESTAMP}.dump" > "$BACKUP_FILE"
    rm -f "$BACKUP_DIR/full/iqfmp_full_${TIMESTAMP}.dump"

    # 记录备份元数据
    cat > "$BACKUP_DIR/full/iqfmp_full_${TIMESTAMP}.meta" << EOF
{
    "type": "full",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "database": "$PGDATABASE",
    "file": "$BACKUP_FILE",
    "size": $(stat -f%z "$BACKUP_FILE" 2>/dev/null || stat -c%s "$BACKUP_FILE"),
    "checksum": "$(sha256sum "$BACKUP_FILE" | cut -d' ' -f1)"
}
EOF

    log "全量备份完成: $BACKUP_FILE"
    echo "$BACKUP_FILE"
}

# ==================== 增量备份 ====================
incremental_backup() {
    log "开始增量备份..."

    BACKUP_FILE="$BACKUP_DIR/incremental/iqfmp_incr_${TIMESTAMP}.sql.gz"

    # 获取上次备份时间
    LAST_BACKUP_TIME=$(cat "$BACKUP_DIR/.last_backup_time" 2>/dev/null || echo "1970-01-01 00:00:00")

    # 备份自上次以来变更的数据
    PGPASSWORD="${PGPASSWORD}" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" << EOF | gzip > "$BACKUP_FILE"
-- 增量备份: 自 $LAST_BACKUP_TIME 以来的变更
-- 生成时间: $(date -u +%Y-%m-%dT%H:%M:%SZ)

-- 备份因子表变更
COPY (
    SELECT * FROM factors
    WHERE updated_at > '$LAST_BACKUP_TIME'
) TO STDOUT WITH CSV HEADER;

-- 备份策略表变更
COPY (
    SELECT * FROM strategies
    WHERE updated_at > '$LAST_BACKUP_TIME'
) TO STDOUT WITH CSV HEADER;

-- 备份回测结果变更
COPY (
    SELECT * FROM backtest_results
    WHERE created_at > '$LAST_BACKUP_TIME'
) TO STDOUT WITH CSV HEADER;

-- 备份交易记录变更
COPY (
    SELECT * FROM trades
    WHERE created_at > '$LAST_BACKUP_TIME'
) TO STDOUT WITH CSV HEADER;

-- 备份研究账本变更
COPY (
    SELECT * FROM research_ledger
    WHERE created_at > '$LAST_BACKUP_TIME'
) TO STDOUT WITH CSV HEADER;
EOF

    # 更新最后备份时间
    date -u +"%Y-%m-%d %H:%M:%S" > "$BACKUP_DIR/.last_backup_time"

    # 记录备份元数据
    cat > "$BACKUP_DIR/incremental/iqfmp_incr_${TIMESTAMP}.meta" << EOF
{
    "type": "incremental",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "database": "$PGDATABASE",
    "file": "$BACKUP_FILE",
    "size": $(stat -f%z "$BACKUP_FILE" 2>/dev/null || stat -c%s "$BACKUP_FILE"),
    "since": "$LAST_BACKUP_TIME",
    "checksum": "$(sha256sum "$BACKUP_FILE" | cut -d' ' -f1)"
}
EOF

    log "增量备份完成: $BACKUP_FILE"
    echo "$BACKUP_FILE"
}

# ==================== 上传到 S3 ====================
upload_to_s3() {
    local file="$1"

    if [[ -n "${S3_BUCKET:-}" ]]; then
        log "上传到 S3: $S3_BUCKET"
        aws s3 cp "$file" "s3://$S3_BUCKET/timescaledb/$(basename "$file")"
        aws s3 cp "${file%.gz}.meta" "s3://$S3_BUCKET/timescaledb/$(basename "${file%.gz}.meta")" 2>/dev/null || true
        log "S3 上传完成"
    fi
}

# ==================== 清理旧备份 ====================
cleanup_old_backups() {
    log "清理 $RETENTION_DAYS 天前的备份..."

    # 清理本地备份
    find "$BACKUP_DIR/full" -name "*.gz" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR/full" -name "*.meta" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR/incremental" -name "*.gz" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR/incremental" -name "*.meta" -mtime +$RETENTION_DAYS -delete

    # 清理 S3 备份 (如果配置了)
    if [[ -n "${S3_BUCKET:-}" ]]; then
        aws s3 ls "s3://$S3_BUCKET/timescaledb/" | while read -r line; do
            file_date=$(echo "$line" | awk '{print $1}')
            file_name=$(echo "$line" | awk '{print $4}')
            if [[ $(date -d "$file_date" +%s) -lt $(date -d "-$RETENTION_DAYS days" +%s) ]]; then
                aws s3 rm "s3://$S3_BUCKET/timescaledb/$file_name"
            fi
        done 2>/dev/null || true
    fi

    log "清理完成"
}

# ==================== 发送通知 ====================
send_notification() {
    local status="$1"
    local message="$2"

    if [[ -n "${TELEGRAM_BOT_TOKEN:-}" && -n "${TELEGRAM_CHAT_ID:-}" ]]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d chat_id="${TELEGRAM_CHAT_ID}" \
            -d text="🗄️ TimescaleDB 备份 $status: $message" \
            -d parse_mode="HTML" > /dev/null || true
    fi
}

# ==================== 主函数 ====================
main() {
    log "========== TimescaleDB 备份开始 =========="
    log "备份类型: $BACKUP_TYPE"

    check_prerequisites

    case "$BACKUP_TYPE" in
        full)
            backup_file=$(full_backup)
            ;;
        incremental)
            backup_file=$(incremental_backup)
            ;;
        *)
            error "未知的备份类型: $BACKUP_TYPE (支持: full, incremental)"
            ;;
    esac

    upload_to_s3 "$backup_file"
    cleanup_old_backups

    send_notification "成功" "备份文件: $(basename "$backup_file")"

    log "========== TimescaleDB 备份完成 =========="
}

# 执行
main
