#!/bin/bash
#
# TimescaleDB 恢复脚本
# 从备份文件恢复数据库
#
# 使用方法:
#   ./timescaledb_restore.sh <backup_file> [--verify-only]
#
# 参数:
#   backup_file - 备份文件路径 (.sql.gz 或 .dump)
#   --verify-only - 仅验证备份文件，不执行恢复
#
# 环境变量:
#   PGHOST - PostgreSQL 主机 (默认: localhost)
#   PGPORT - PostgreSQL 端口 (默认: 5432)
#   PGUSER - PostgreSQL 用户 (默认: iqfmp)
#   PGPASSWORD - PostgreSQL 密码
#   PGDATABASE - 数据库名称 (默认: iqfmp)

set -euo pipefail

# ==================== 配置 ====================
PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-iqfmp}"
PGDATABASE="${PGDATABASE:-iqfmp}"
BACKUP_FILE="${1:-}"
VERIFY_ONLY="${2:-}"
TEMP_DIR="/tmp/iqfmp_restore_$$"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
    cleanup
    exit 1
}

cleanup() {
    rm -rf "$TEMP_DIR" 2>/dev/null || true
}

trap cleanup EXIT

# ==================== 使用说明 ====================
usage() {
    cat << EOF
使用方法: $0 <backup_file> [--verify-only]

参数:
    backup_file     备份文件路径 (.sql.gz, .dump, 或从 S3)
    --verify-only   仅验证备份文件，不执行恢复

示例:
    $0 /var/backups/timescaledb/full/iqfmp_full_20240101_120000.sql.gz
    $0 s3://my-bucket/timescaledb/iqfmp_full_20240101.sql.gz
    $0 /var/backups/timescaledb/full/iqfmp_full_20240101.dump --verify-only

环境变量:
    PGHOST      PostgreSQL 主机 (默认: localhost)
    PGPORT      PostgreSQL 端口 (默认: 5432)
    PGUSER      PostgreSQL 用户 (默认: iqfmp)
    PGPASSWORD  PostgreSQL 密码
    PGDATABASE  数据库名称 (默认: iqfmp)
EOF
    exit 1
}

# ==================== 前置检查 ====================
check_prerequisites() {
    log "检查前置条件..."

    if [[ -z "$BACKUP_FILE" ]]; then
        usage
    fi

    # 检查必要工具
    for cmd in pg_restore psql gunzip; do
        if ! command -v "$cmd" &> /dev/null; then
            error "$cmd 未安装"
        fi
    done

    # 创建临时目录
    mkdir -p "$TEMP_DIR"

    log "前置检查完成"
}

# ==================== 下载备份文件 ====================
download_backup() {
    log "准备备份文件..."

    if [[ "$BACKUP_FILE" == s3://* ]]; then
        log "从 S3 下载: $BACKUP_FILE"
        LOCAL_BACKUP="$TEMP_DIR/$(basename "$BACKUP_FILE")"
        aws s3 cp "$BACKUP_FILE" "$LOCAL_BACKUP"
    else
        if [[ ! -f "$BACKUP_FILE" ]]; then
            error "备份文件不存在: $BACKUP_FILE"
        fi
        LOCAL_BACKUP="$BACKUP_FILE"
    fi

    log "备份文件: $LOCAL_BACKUP"
}

# ==================== 验证备份 ====================
verify_backup() {
    log "验证备份文件..."

    # 检查文件大小
    FILE_SIZE=$(stat -f%z "$LOCAL_BACKUP" 2>/dev/null || stat -c%s "$LOCAL_BACKUP")
    if [[ "$FILE_SIZE" -lt 1000 ]]; then
        error "备份文件太小，可能已损坏"
    fi

    # 检查 checksum (如果元数据文件存在)
    META_FILE="${LOCAL_BACKUP%.gz}.meta"
    if [[ -f "$META_FILE" ]]; then
        EXPECTED_CHECKSUM=$(grep -o '"checksum": "[^"]*"' "$META_FILE" | cut -d'"' -f4)
        ACTUAL_CHECKSUM=$(sha256sum "$LOCAL_BACKUP" | cut -d' ' -f1)

        if [[ "$EXPECTED_CHECKSUM" != "$ACTUAL_CHECKSUM" ]]; then
            error "校验和不匹配! 期望: $EXPECTED_CHECKSUM, 实际: $ACTUAL_CHECKSUM"
        fi
        log "校验和验证通过"
    fi

    # 尝试解压测试
    if [[ "$LOCAL_BACKUP" == *.gz ]]; then
        if ! gunzip -t "$LOCAL_BACKUP" 2>/dev/null; then
            error "gzip 文件损坏"
        fi
        log "gzip 完整性验证通过"
    fi

    log "备份验证完成"

    if [[ "$VERIFY_ONLY" == "--verify-only" ]]; then
        log "仅验证模式，退出"
        exit 0
    fi
}

# ==================== 恢复前确认 ====================
confirm_restore() {
    log "⚠️  警告: 此操作将覆盖现有数据库!"
    log "目标数据库: $PGHOST:$PGPORT/$PGDATABASE"
    log "备份文件: $LOCAL_BACKUP"

    read -p "确认恢复? (输入 'yes' 继续): " confirmation
    if [[ "$confirmation" != "yes" ]]; then
        log "用户取消恢复"
        exit 0
    fi
}

# ==================== 停止服务 ====================
stop_services() {
    log "停止相关服务..."

    # 停止应用连接 (可选)
    # docker-compose stop backend celery-worker celery-beat || true

    # 终止活跃连接
    PGPASSWORD="${PGPASSWORD}" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d postgres << EOF || true
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = '$PGDATABASE' AND pid <> pg_backend_pid();
EOF

    log "服务已停止"
}

# ==================== 执行恢复 ====================
restore_database() {
    log "开始恢复数据库..."

    # 解压文件 (如果需要)
    if [[ "$LOCAL_BACKUP" == *.gz ]]; then
        RESTORE_FILE="$TEMP_DIR/$(basename "${LOCAL_BACKUP%.gz}")"
        gunzip -c "$LOCAL_BACKUP" > "$RESTORE_FILE"
    else
        RESTORE_FILE="$LOCAL_BACKUP"
    fi

    # 根据文件类型选择恢复方式
    if [[ "$RESTORE_FILE" == *.dump ]]; then
        log "使用 pg_restore 恢复..."

        # 删除并重建数据库
        PGPASSWORD="${PGPASSWORD}" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d postgres << EOF
DROP DATABASE IF EXISTS ${PGDATABASE}_backup;
ALTER DATABASE $PGDATABASE RENAME TO ${PGDATABASE}_backup;
CREATE DATABASE $PGDATABASE OWNER $PGUSER;
EOF

        # 恢复数据
        PGPASSWORD="${PGPASSWORD}" pg_restore \
            -h "$PGHOST" \
            -p "$PGPORT" \
            -U "$PGUSER" \
            -d "$PGDATABASE" \
            --no-owner \
            --no-privileges \
            "$RESTORE_FILE"

    elif [[ "$RESTORE_FILE" == *.sql ]]; then
        log "使用 psql 恢复..."

        # 备份当前数据库
        PGPASSWORD="${PGPASSWORD}" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d postgres << EOF
DROP DATABASE IF EXISTS ${PGDATABASE}_backup;
ALTER DATABASE $PGDATABASE RENAME TO ${PGDATABASE}_backup;
CREATE DATABASE $PGDATABASE OWNER $PGUSER;
EOF

        # 恢复数据
        PGPASSWORD="${PGPASSWORD}" psql \
            -h "$PGHOST" \
            -p "$PGPORT" \
            -U "$PGUSER" \
            -d "$PGDATABASE" \
            -f "$RESTORE_FILE"
    else
        error "不支持的备份文件格式: $RESTORE_FILE"
    fi

    log "数据库恢复完成"
}

# ==================== 验证恢复 ====================
verify_restore() {
    log "验证恢复结果..."

    # 检查表是否存在
    TABLE_COUNT=$(PGPASSWORD="${PGPASSWORD}" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -t -c "
        SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';
    " | tr -d ' ')

    log "恢复的表数量: $TABLE_COUNT"

    if [[ "$TABLE_COUNT" -lt 1 ]]; then
        error "恢复失败: 未找到任何表"
    fi

    # 检查关键表的数据
    for table in factors strategies backtest_results; do
        COUNT=$(PGPASSWORD="${PGPASSWORD}" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -t -c "
            SELECT COUNT(*) FROM $table;
        " 2>/dev/null | tr -d ' ' || echo "0")
        log "表 $table 记录数: $COUNT"
    done

    log "恢复验证完成"
}

# ==================== 清理旧数据库 ====================
cleanup_old_database() {
    log "清理旧数据库备份..."

    read -p "删除旧数据库 ${PGDATABASE}_backup? (y/n): " confirm
    if [[ "$confirm" == "y" ]]; then
        PGPASSWORD="${PGPASSWORD}" psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d postgres -c "
            DROP DATABASE IF EXISTS ${PGDATABASE}_backup;
        "
        log "旧数据库已删除"
    else
        log "保留旧数据库: ${PGDATABASE}_backup"
    fi
}

# ==================== 发送通知 ====================
send_notification() {
    local status="$1"
    local message="$2"

    if [[ -n "${TELEGRAM_BOT_TOKEN:-}" && -n "${TELEGRAM_CHAT_ID:-}" ]]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d chat_id="${TELEGRAM_CHAT_ID}" \
            -d text="🔄 TimescaleDB 恢复 $status: $message" \
            -d parse_mode="HTML" > /dev/null || true
    fi
}

# ==================== 主函数 ====================
main() {
    log "========== TimescaleDB 恢复开始 =========="

    check_prerequisites
    download_backup
    verify_backup
    confirm_restore
    stop_services
    restore_database
    verify_restore
    cleanup_old_database

    send_notification "成功" "数据库已恢复"

    log "========== TimescaleDB 恢复完成 =========="
}

# 执行
main
