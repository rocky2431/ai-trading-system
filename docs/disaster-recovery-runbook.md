# 灾难恢复演练手册

## 概述

本文档描述 IQFMP 系统的灾难恢复流程，包括数据库备份策略、恢复步骤和定期演练计划。

---

## 1. 备份策略

### 1.1 TimescaleDB 备份

| 备份类型 | 频率 | 保留期 | 存储位置 |
|---------|------|--------|---------|
| 增量备份 | 每小时 | 7 天 | `/var/backups/timescaledb/incremental/` |
| 全量备份 | 每天 02:00 | 30 天 | `/var/backups/timescaledb/full/` |
| S3 同步 | 实时 | 90 天 | `s3://iqfmp-backups/timescaledb/` |

**备份内容**:
- 因子库 (factors)
- 策略配置 (strategies)
- 回测结果 (backtest_results)
- 交易记录 (trades)
- 研究账本 (research_ledger)

### 1.2 Redis 备份

| 备份类型 | 频率 | 保留期 | 存储位置 |
|---------|------|--------|---------|
| RDB 快照 | 每小时 | 7 天 | `/var/backups/redis/rdb/` |
| AOF 文件 | 每 6 小时 | 3 天 | `/var/backups/redis/aof/` |
| S3 同步 | 实时 | 30 天 | `s3://iqfmp-backups/redis/` |

**备份内容**:
- 因子缓存
- 会话数据
- 任务队列状态
- 实时计算中间结果

---

## 2. Cron 调度配置

将以下内容添加到 crontab:

```bash
# 编辑 crontab
crontab -e

# TimescaleDB 备份
0 * * * * /opt/iqfmp/scripts/backup/timescaledb_backup.sh incremental >> /var/log/iqfmp/backup.log 2>&1
0 2 * * * /opt/iqfmp/scripts/backup/timescaledb_backup.sh full >> /var/log/iqfmp/backup.log 2>&1

# Redis 备份
30 * * * * /opt/iqfmp/scripts/backup/redis_backup.sh rdb >> /var/log/iqfmp/backup.log 2>&1
0 */6 * * * /opt/iqfmp/scripts/backup/redis_backup.sh both >> /var/log/iqfmp/backup.log 2>&1

# 清理 30 天前的日志
0 3 * * * find /var/log/iqfmp -name "*.log" -mtime +30 -delete
```

---

## 3. 恢复流程

### 3.1 TimescaleDB 恢复

#### 场景 A: 数据损坏 (部分恢复)

```bash
# 1. 列出可用备份
ls -la /var/backups/timescaledb/full/

# 2. 验证备份完整性
./scripts/restore/timescaledb_restore.sh /var/backups/timescaledb/full/iqfmp_full_YYYYMMDD_HHMMSS.sql.gz --verify-only

# 3. 执行恢复 (会提示确认)
./scripts/restore/timescaledb_restore.sh /var/backups/timescaledb/full/iqfmp_full_YYYYMMDD_HHMMSS.sql.gz
```

#### 场景 B: 服务器故障 (完整恢复)

```bash
# 1. 在新服务器安装 PostgreSQL + TimescaleDB
sudo apt install postgresql-14 timescaledb-2-postgresql-14

# 2. 从 S3 下载备份
./scripts/restore/timescaledb_restore.sh s3://iqfmp-backups/timescaledb/iqfmp_full_YYYYMMDD.sql.gz

# 3. 验证恢复结果
psql -U iqfmp -d iqfmp -c "SELECT COUNT(*) FROM factors;"
```

#### 场景 C: 误删数据 (时间点恢复)

```bash
# 1. 找到误删之前的增量备份
ls -la /var/backups/timescaledb/incremental/ | grep "YYYYMMDD"

# 2. 恢复全量备份
./scripts/restore/timescaledb_restore.sh /var/backups/timescaledb/full/iqfmp_full_YYYYMMDD.sql.gz

# 3. 依次应用增量备份 (按时间顺序)
for backup in /var/backups/timescaledb/incremental/iqfmp_incr_YYYYMMDD_*.sql.gz; do
    psql -U iqfmp -d iqfmp < <(gunzip -c "$backup")
done
```

### 3.2 Redis 恢复

#### 场景 A: Redis 重启后数据丢失

```bash
# 1. 列出可用 RDB 备份
ls -la /var/backups/redis/rdb/

# 2. 恢复 RDB
./scripts/restore/redis_restore.sh /var/backups/redis/rdb/redis_YYYYMMDD_HHMMSS.rdb.gz
```

#### 场景 B: 需要精确恢复 (使用 AOF)

```bash
# AOF 包含所有写操作，可恢复到更精确的时间点
./scripts/restore/redis_restore.sh /var/backups/redis/aof/redis_YYYYMMDD_HHMMSS.aof.gz
```

---

## 4. 演练计划

### 4.1 月度演练 (每月第一个周六)

**目标**: 验证备份完整性和恢复流程

**步骤**:

1. **备份验证** (10 分钟)
   ```bash
   # 验证最近的全量备份
   ./scripts/restore/timescaledb_restore.sh /var/backups/timescaledb/full/latest.sql.gz --verify-only
   ./scripts/restore/redis_restore.sh /var/backups/redis/rdb/latest.rdb.gz --verify-only
   ```

2. **测试环境恢复** (30 分钟)
   ```bash
   # 在测试环境执行完整恢复
   export PGHOST=test-db.iqfmp.local
   export PGDATABASE=iqfmp_test
   ./scripts/restore/timescaledb_restore.sh /var/backups/timescaledb/full/latest.sql.gz
   ```

3. **数据一致性检查** (15 分钟)
   ```bash
   # 比较生产和测试环境的记录数
   psql -h prod-db -U iqfmp -d iqfmp -c "SELECT COUNT(*) FROM factors;"
   psql -h test-db -U iqfmp -d iqfmp_test -c "SELECT COUNT(*) FROM factors;"
   ```

4. **记录结果** (5 分钟)
   - 记录演练时间
   - 记录恢复耗时
   - 记录发现的问题

### 4.2 季度演练 (每季度末)

**目标**: 完整灾难恢复模拟

**步骤**:

1. 模拟主数据库故障
2. 切换到备用服务器
3. 从 S3 恢复最新备份
4. 验证所有服务正常运行
5. 测试应用程序连接

**预期 RTO (恢复时间目标)**: < 1 小时
**预期 RPO (恢复点目标)**: < 1 小时

### 4.3 演练检查清单

- [ ] 备份文件存在且可访问
- [ ] 备份文件校验和匹配
- [ ] gzip 文件可正常解压
- [ ] 恢复脚本执行成功
- [ ] 表结构完整
- [ ] 关键表数据正确
- [ ] 应用程序可正常连接
- [ ] 通知系统已发送告警

---

## 5. 故障响应流程

### 5.1 告警触发

当以下情况发生时，系统会通过 Telegram 发送告警:

- 备份脚本执行失败
- 备份文件大小异常
- S3 上传失败
- 磁盘空间不足 (<10%)

### 5.2 响应步骤

```
1. 收到告警 (Telegram)
   ↓
2. 确认故障范围
   - 检查服务状态: docker-compose ps
   - 检查数据库连接: psql -h localhost -U iqfmp -c "SELECT 1"
   - 检查 Redis: redis-cli ping
   ↓
3. 评估数据丢失程度
   - 检查最近备份时间
   - 确定 RPO 影响
   ↓
4. 执行恢复
   - 选择合适的备份文件
   - 执行恢复脚本
   - 验证数据完整性
   ↓
5. 恢复服务
   - 重启应用服务
   - 验证功能正常
   ↓
6. 事后分析
   - 记录故障原因
   - 更新恢复文档
   - 优化监控告警
```

---

## 6. 联系人

| 角色 | 姓名 | 联系方式 |
|-----|------|---------|
| DBA | TBD | Telegram: @xxx |
| DevOps | TBD | Telegram: @xxx |
| 技术负责人 | TBD | Telegram: @xxx |

---

## 7. 相关命令速查

```bash
# 查看备份状态
ls -la /var/backups/timescaledb/full/
ls -la /var/backups/redis/rdb/

# 手动触发备份
./scripts/backup/timescaledb_backup.sh full
./scripts/backup/redis_backup.sh both

# 验证备份
./scripts/restore/timescaledb_restore.sh <backup_file> --verify-only

# 查看数据库大小
psql -U iqfmp -d iqfmp -c "SELECT pg_size_pretty(pg_database_size('iqfmp'));"

# 查看 Redis 内存使用
redis-cli INFO memory | grep used_memory_human

# 检查最近备份
find /var/backups -name "*.gz" -mmin -60 -ls
```

---

## 更新记录

| 日期 | 版本 | 更新内容 |
|-----|------|---------|
| 2024-12-10 | 1.0 | 初始版本 |
