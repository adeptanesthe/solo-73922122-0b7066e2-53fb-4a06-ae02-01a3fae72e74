# 风机耗电共用模型平台（Fan Power Model Platform）

各厂只上报工况摘要（样本数 + 整数向量和 + 外积和 + 压差零点修正量），平台用
NumPy 岭回归拟合一套共用的"转速/压差 → 每分钟耗电量"模型，按留一厂交叉验证
选 λ，SQLite 多版本保存，支持凭版号原子替换单厂修正量。

## 快速开始

```bash
docker compose up --build      # 服务监听 :8000
./demo.sh                      # 演示：建厂 → 预测 → 选档 → 更正修正量 → 换版复核被拒 → 冲突
```

本地开发（无 Docker）：

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn --factory app.main:create_app --port 8000
.venv/bin/python -m pytest tests/ -q     # 50 个自测
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/versions` | 上报 3–8 家厂的摘要，训练并发布新版本（201） |
| GET | `/versions/current` | 现行版：所选 λ、逐厂误差、修正量、模型 |
| GET | `/versions/{id}` | 历史版本（不可变） |
| POST | `/versions/{id}/correction` | 凭版号替换一厂修正量；成功则修正量+模型+逐厂误差作为新版本一起生效；版号冲突 409 / 重算失败 422，均保留现行版 |
| POST | `/predict` | 按厂号 + 原始转速、压差预测耗电，返回所用版号；重算期间沿用现行版 |
| POST | `/gear/select` | 在原始压差区间、整数转速范围与每分钟耗电上限下，解析求出**最高可用整数档**及版号；无解返回 `available: false` |
| POST | `/gear/review` | 凭原条件 + 选定档位复核：按**现行版**的修正量/摘要/模型判定；不合格 409 并附本次建议 |
| GET | `/health` | 健康检查 |

### 上报格式（`POST /versions`）

```json
{
  "factories": [
    {
      "factory_id": "F1",
      "n": 1440,
      "sums":  [Σ转速, Σ压差, Σ耗电],
      "outer": [[Σs², Σsp, Σsy], [Σsp, Σp², Σpy], [Σsy, Σpy, Σy²]],
      "correction": 0
    }
  ]
}
```

`sums`/`outer` 均为**整数**（任意精度，JSON 大整数无损解析）；`correction`
为加到压差上的整数零点修正量。预测时传原始读数，平台自动加上该厂修正量。

### 选档（`POST /gear/select`）

```json
{
  "factory_id": "F3",
  "pressure_min": 500000, "pressure_max": 500030,
  "speed_min": 999900, "speed_max": 1000200,
  "power_limit": 27007
}
```

`pressure_min`/`pressure_max` 是**原始整数压差端点**（顺序任意），服务端加上
该厂修正量后得到压差区间；`speed_min`/`speed_max` 是整数转速范围，
`power_limit` 是每分钟耗电上限。档位 `s` **可用**当且仅当对修正后区间内
**每一点** `p`：

1. `model.predict(s, p) <= power_limit`；
2. `(s, p)` 到该厂工况均值的马氏距离 `<= 3`；均值与**总体协方差**取自该厂
   的转速和修正后压差。零方差方向按伪逆极限处理：该方向分量必须与均值
   **严格相等**，否则距离为无穷。

返回 `{available, gear, version, correction, power_limit, mahalanobis_limit,
max_power, max_distance}`。`max_power`/`max_distance` 是所选档位在整个区间上
的最坏值。**无解时** `available: false`、`gear: null`（仍返回所用版号）。

求解不逐档试算：耗电对 `p` 线性、马氏距离平方对 `p` 凸（含伪逆），故只需
两个区间端点；固定端点时距离界化为**整数二次不等式**（用 `math.isqrt` 精确
求根），耗电界化为**有理数线性不等式**；两个端点可行转速区间的交仍是一个
区间，其右端就是最高可用档。整个可行集求解为常数时间，百万级、十亿级转速
范围同样即时。距离计算全程 Python 大整数（协方差分子精确），大基值小方差
不会抵消；耗电按存储模型的精确有理数计算。

### 复核（`POST /gear/review`）

在选档请求体上加 `gear` 字段。按**提交时刻的现行版**重放原条件：换版后
（如修正量更新）用新版修正量、新版摘要和新版模型重新判定。
- 合格 → 200，`accepted: true`，`reasons: []`；
- 不合格 → **409**，`accepted: false`，`reasons` 含
  `speed_range`/`power_limit`/`mahalanobis`，并附 `suggestion`
  （现行版下的 `available/gear/max_power/max_distance`）。

选档与复核都在一次读取中取同一现行版的修正量、摘要和模型；重算期间仍用
已提交的现行版。

### 选档与复核请求示例

```bash
# 选档
curl -sS -X POST localhost:8000/gear/select -H 'Content-Type: application/json' \
  --data @demo/gear_select.json

# 换版后复核（版本 2 已把 F3 修正量改为 500）
curl -sS -X POST localhost:8000/gear/review -H 'Content-Type: application/json' \
  --data @demo/gear_review.json
# -> HTTP 409, reasons: ["power_limit"], suggestion.gear = 1000059
```

## 关键设计

- **大基值方差保留**：协方差分子 `N·Σxxᵀ − ΣxΣxᵀ` 全程用 Python 大整数精确
  计算，最后一步才转 float64。读数基值 1e12、方差仅个位数时，朴素
  `E[x²]−E[x]²` 会被灾难性抵消成 0，这里仍精确保留（有专门测试）。
  入库前对所有厂做精确 PSD 校验（全部主子式 ≥ 0，整数判定）。
- **岭回归**：损失 = 均方误差 + λ·‖w‖²，截距免罚。特征按训练厂的 pooled
  均值与**总体标准差**标准化后，闭式解 `(C + λI)w = Cov(z,y)`，
  `b = ȳ`；零方差特征忽略。
- **留一厂验证**：轮流留一厂，只用其余厂的均值/总体标准差标准化并拟合，
  评估留出厂 MSE（由充分统计量精确求出，无需原始行）。在
  {0.01, 0.1, 1} 中选"各留出厂 MSE 最大值"最小的 λ，并列取较小者，
  再用全厂重训。
- **版本原子性**：修正量、模型、逐厂误差在同一 SQLite 事务写入，最后才翻动
  `current_version` 指针；预测只读已提交的现行版，因此重算期间预测仍沿用
  现行版。凭版号更新时若版号已过期 → 409 且现行版不变；重算失败同样保留。
- **并发**：WAL 模式 + 写锁；计算在事务外完成，提交窗口极短。
- **选档解析求解**：两个条件对压差分别线性/凸二次，故只检查区间两端点。
  固定端点时，"整数转速 + 马氏距离² ≤ 9" 化成整数二次不等式，
  `math.isqrt` 精确求根后再处理整数舍入；耗电上限化成有理数线性不等式。
  可行转速集为一个区间，取右端即最高可用档，与转速范围大小无关，无逐档
  试算。零方差方向按伪逆极限：分量必须与均值严格相等。
- **判定版本一致**：每次选档/复核只读一次现行版，修正量、逐厂摘要与模型
  都来自这一版；复核在换版后自动按新版判定。

## 演示（demo.sh）

`scripts/make_demo_data.py` 生成确定性的 4 厂数据（转速基值 1e6、压差基值
5e5 的原始计数；F3 压差零点漂移 +500 但按 0 上报）。`demo.sh` 依次：

1. `POST /versions` 建厂 → F3 留厂误差 33.3 明显偏高，λ=0.01；
2. `POST /predict` 用原始读数预测 F3；
2b. `POST /gear/select` 选定档位 1000071（版号 1，修正量 0）；
3. `POST /versions/1/correction` 把 F3 修正量改为 500 → 版本 2，
   逐厂误差全部降到 ≈1–2，模型与修正量同时生效；
3b. `POST /gear/review` 用原条件复核旧档位 → 新版下超出耗电上限，
   HTTP 409，建议档位 1000059；
4. 再次预测 → 结果改变且 `version=2`；再次选档 → 1000059；
5. 用过期版号重放 → HTTP 409，现行版保持版本 2。

## 剩余事项

- 本环境无 Docker，`docker compose` 未实际构建过；Dockerfile 为标准
  python:3.12-slim 流程，建议在有 Docker 的机器上跑一次 `docker compose up --build`。
- 单进程服务（uvicorn 单 worker）；多副本部署需把 SQLite 换成集中式存储。
- 未做鉴权/审计；生产环境应加认证与操作日志。
- 留一厂验证对工况迥异的厂会外推放大误差（属预期行为）；如需可扩展分组验证。
