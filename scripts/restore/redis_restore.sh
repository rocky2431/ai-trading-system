#!/bin/bash
#
# Redis 恢复脚本
# 从 RDB 或 AOF 备份恢复数据
#
# 使用方法:
#   ./redis_restore.sh <backup_file> [--verify-only]
#
# 参数:
#   backup_file - 备份文件路径 (.rdb.gz 或 .aof.gz)
#   --verify-only - 仅验证备份文件，不执行恢复
#
# 环境变量:
#   REDIS_HOST - Redis 主机 (默认: localhost)
#   REDIS_PORT - Redis 端口 (默认: 6379)
#   REDIS_PASSWORD - Redis 密码 (可选)

set -euo pipefail

# ==================== 配置 ====================
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"
BACKUP_FILE="${1:-}"
VERIFY_ONLY="${2:-}"
TEMP_DIR="/tmp/redis_restore_$$"

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
    backup_file     备份文件路径 (.rdb.gz 或 .aof.gz)
    --verify-only   仅验证备份文件，不执行恢复

示例:
    $0 /var/backups/redis/rdb/redis_20240101_120000.rdb.gz
    $0 /var/backups/redis/aof/redis_20240101_120000.aof.gz --verify-only

环境变量:
    REDIS_HOST      Redis 主机 (默认: localhost)
    REDIS_PORT      Redis 端口 (默认: 6379)
    REDIS_PASSWORD  Redis 密码 (可选)
EOF
    exit 1
}

# ==================== 前置检查 ====================
check_prerequisites() {
    log "检查前置条件..."

    if [[ -z "$BACKUP_FILE" ]]; then
        usage
    fi

    # 检查 redis-cli
    if ! command -v redis-cli &> /dev/null; then
        error "redis-cli 未安装"
    fi

    # 检查 Redis 连接
    if ! $REDIS_CLI PING &> /dev/null; then
        error "无法连接到 Redis"
    fi

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
    if [[ "$FILE_SIZE" -lt 100 ]]; then
        error "备份文件太小，可能已损坏"
    fi

    # 尝试解压测试
    if [[ "$LOCAL_BACKUP" == *.gz ]]; then
        if ! gunzip -t "$LOCAL_BACKUP" 2>/dev/null; then
            error "gzip 文件损坏"
        fi
        log "gzip 完整性验证通过"
    fi

    # 检查文件类型
    if [[ "$LOCAL_BACKUP" == *".rdb"* ]]; then
        RESTORE_TYPE="rdb"
    elif [[ "$LOCAL_BACKUP" == *".aof"* ]]; then
        RESTORE_TYPE="aof"
    else
        error "无法识别的备份文件类型"
    fi

    log "备份类型: $RESTORE_TYPE"
    log "备份验证完成"

    if [[ "$VERIFY_ONLY" == "--verify-only" ]]; then
        log "仅验证模式，退出"
        exit 0
    fi
}

# ==================== 恢复前确认 ====================
confirm_restore() {
    log "⚠️  警告: 此操作将覆盖现有 Redis 数据!"
    log "目标 Redis: $REDIS_HOST:$REDIS_PORT"
    log "备份文件: $LOCAL_BACKUP"

    # 显示当前数据库状态
    DBSIZE=$($REDIS_CLI DBSIZE | awk '{print $2}')
    log "当前 Redis 键数量: $DBSIZE"

    read -p "确认恢复? (输入 'yes' 继续): " confirmation
    if [[ "$confirmation" != "yes" ]]; then
        log "用户取消恢复"
        exit 0
    fi
}

# ==================== 执行恢复 ====================
restore_redis() {
    log "开始恢复 Redis..."

    # 获取 Redis 数据目录
    REDIS_DIR=$($REDIS_CLI CONFIG GET dir | tail -1)

    # 解压文件
    if [[ "$LOCAL_BACKUP" == *.gz ]]; then
        RESTORE_FILE="$TEMP_DIR/$(basename "${LOCAL_BACKUP%.gz}")"
        gunzip -c "$LOCAL_BACKUP" > "$RESTORE_FILE"
    else
        RESTORE_FILE="$LOCAL_BACKUP"
    fi

    if [[ "$RESTORE_TYPE" == "rdb" ]]; then
        restore_rdb "$RESTORE_FILE"
    else
        restore_aof "$RESTORE_FILE"
    fi

    log "Redis 恢复完成"
}

# ==================== RDB 恢复 ====================
restore_rdb() {
    local rdb_file="$1"
    log "恢复 RDB 文件..."

    # 获取 RDB 文件名
    REDIS_DBFILENAME=$($REDIS_CLI CONFIG GET dbfilename | tail -1)

    # 停止 Redis 持久化
    $REDIS_CLI CONFIG SET save ""

    # 备份当前 RDB
    if [[ -f "$REDIS_DIR/$REDIS_DBFILENAME" ]]; then
        cp "$REDIS_DIR/$REDIS_DBFILENAME" "$REDIS_DIR/${REDIS_DBFILENAME}.bak"
        log "已备份当前 RDB: ${REDIS_DBFILENAME}.bak"
    fi

    # 清空当前数据
    log "清空当前数据..."
    $REDIS_CLI FLUSHALL

    # 复制新 RDB 文件
    cp "$rdb_file" "$REDIS_DIR/$REDIS_DBFILENAME"

    # 重启 Redis 以加载新数据
    log "重新加载数据..."
    $REDIS_CLI DEBUG RELOAD || {
        log "DEBUG RELOAD 不可用，请手动重启 Redis"
        log "RDB 文件已复制到: $REDIS_DIR/$REDIS_DBFILENAME"
    }

    # 恢复持久化配置
    $REDIS_CLI CONFIG SET save "900 1 300 10 60 10000"
}

# ==================== AOF 恢复 ====================
restore_aof() {
    local aof_file="$1"
    log "恢复 AOF 文件..."

    # 获取 AOF 文件名
    AOF_FILENAME=$($REDIS_CLI CONFIG GET appendfilename | tail -1)

    # 停止 AOF
    $REDIS_CLI CONFIG SET appendonly no

    # 备份当前 AOF
    if [[ -f "$REDIS_DIR/$AOF_FILENAME" ]]; then
        cp "$REDIS_DIR/$AOF_FILENAME" "$REDIS_DIR/${AOF_FILENAME}.bak"
        log "已备份当前 AOF: ${AOF_FILENAME}.bak"
    fi

    # 清空当前数据
    log "清空当前数据..."
    $REDIS_CLI FLUSHALL

    # 复制新 AOF 文件
    cp "$aof_file" "$REDIS_DIR/$AOF_FILENAME"

    # 重新启用 AOF
    $REDIS_CLI CONFIG SET appendonly yes

    # 重新加载 AOF
    log "重新加载 AOF..."
    $REDIS_CLI DEBUG LOADAOF || {
        log "DEBUG LOADAOF 不可用，请手动重启 Redis"
        log "AOF 文件已复制到: $REDIS_DIR/$AOF_FILENAME"
    }
}

# ==================== 验证恢复 ====================
verify_restore() {
    log "验证恢复结果..."

    # 检查键数量
    DBSIZE=$($REDIS_CLI DBSIZE | awk '{print $2}')
    log "恢复后键数量: $DBSIZE"

    if [[ "$DBSIZE" == "0" ]]; then
        log "⚠️  警告: 数据库为空"
    fi

    # 检查内存使用
    MEMORY=$($REDIS_CLI INFO memory | grep used_memory_human | cut -d: -f2 | tr -d '\r')
    log "内存使用: $MEMORY"

    log "恢复验证完成"
}

# ==================== 发送通知 ====================
send_notification() {
    local status="$1"
    local message="$2"

    if [[ -n "${TELEGRAM_BOT_TOKEN:-}" && -n "${TELEGRAM_CHAT_ID:-}" ]]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d chat_id="${TELEGRAM_CHAT_ID}" \
            -d text="🔄 Redis 恢复 $status: $message" \
            -d parse_mode="HTML" > /dev/null || true
    fi
}

# ==================== 主函数 ====================
main() {
    log "========== Redis 恢复开始 =========="

    check_prerequisites
    download_backup
    verify_backup
    confirm_restore
    restore_redis
    verify_restore

    send_notification "成功" "Redis 已恢复"

    log "========== Redis 恢复完成 =========="
}

# 执行
main
