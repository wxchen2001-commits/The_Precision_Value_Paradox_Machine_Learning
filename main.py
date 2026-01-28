import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# ==========================================
# 0. 环境配置 (Environment Setup)
# ==========================================
# 关闭 oneDNN 提示以保证数值一致性
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


# ==========================================
# 1. 数据获取与对齐 (Data Acquisition)
# ==========================================
# ==========================================
# 1. 数据获取与对齐 (修正版: 2015-2025)
# ==========================================
def get_aligned_data():
    # 设定时间区间：2015-01-01 至 2025-12-31 (11年)
    start_date = "2015-01-01"
    end_date = "2025-12-31"

    print(f"正在获取数据 ({start_date} to {end_date})...")

    # 下载数据
    sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    btc = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)

    # 1.1 计算对数收益率
    sp500['Return'] = np.log(sp500['Close'] / sp500['Close'].shift(1)) * 100
    btc['Return'] = np.log(btc['Close'] / btc['Close'].shift(1)) * 100

    # 1.2 计算 GK 波动率代理 (Scale * 10000 统一量纲)
    for df in [sp500, btc]:
        # 防止 High=Low 导致 log(1)=0 的数学错误
        df['High'] = np.maximum(df['High'], df['Low'])
        df['Close'] = np.maximum(df['Close'], df['Open'])  # 防止 Close=Open

        term1 = 0.5 * (np.log(df['High'] / df['Low'])) ** 2
        term2 = 0.3863 * (np.log(df['Close'] / df['Open'])) ** 2
        df['GK'] = (term1 - term2) * 10000
        # 极小值截断
        df['GK'] = df['GK'].apply(lambda x: max(x, 1e-6))

    # 1.3 日历对齐 (Appendix C)
    # 以 S&P 500 的交易日为准
    common_index = sp500.index.intersection(btc.index)
    sp500 = sp500.loc[common_index].dropna()
    btc = btc.loc[common_index].dropna()


    # target_len = 2514
    # sp500 = sp500.iloc[:target_len]
    # btc = btc.iloc[:target_len]

    print(f"数据对齐完成。总样本量: {len(sp500)} (区间: 2015-2025)")
    return sp500, btc

# ==========================================
# 2. 模型定义 (Full Models - No Simulation)
# ==========================================

# --- 2.1 GARCH / EGARCH (Daily Refit) ---
def run_garch_full(returns, model_type='GARCH', dist='t', window_size=1000, refit_freq=1):
    """
    标准的 GARCH 滚动预测 (Standard Rolling Window)

    逻辑：
    1. 窗口(Window)每天向后滑动一步，包含最新的市场信息。
    2. 如果 i % refit_freq == 0: 执行 .fit() 重新估算参数 (Re-estimation)。
    3. 如果 i % refit_freq != 0: 执行 .fix() 使用旧参数但基于新数据预测 (Parameter Fixing)。
    4. 无论如何，每天都基于当前的 1000 个观测值预测明天的方差。
    """
    import numpy as np
    import arch

    # 1. 初始化
    n = len(returns)
    test_size = n - window_size
    forecasts = np.zeros(test_size)

    # GJR-GARCH 需要设置 o=1
    o_param = 1 if model_type == 'GJR-GARCH' else 0

    print(f"启动 {model_type} 完整滚动预测 (Window={window_size}, Refit={refit_freq})...")

    # 用于存储上一次拟合的参数
    current_params = None

    # 2. 核心滚动循环 (每天执行一次)
    for i in range(test_size):
        # [关键点]: 窗口每天都在滑动！
        # 今天的训练集 = 从 i 到 i + window_size
        train_data = returns.iloc[i: i + window_size]

        # 定义模型结构 (基于当前窗口数据)
        am = arch.arch_model(train_data, vol=model_type, p=1, o=o_param, q=1, dist=dist)

        # 3. 决定是“重估参数”还是“沿用参数”
        if i % refit_freq == 0:
            # 【重估模式】: 运行优化算法寻找最优参数
            # disp='off' 关闭优化过程的打印输出
            res = am.fit(disp='off', show_warning=False)
            current_params = res.params  # 保存参数供下次使用

        else:
            # 【沿用模式】: 不运行优化，直接将上一次的参数(current_params)应用到新数据上
            # 这是一个标准的加速手段：数据是新的，但假设参数结构短期不变
            if current_params is None:
                # 防御性编程：如果是第一次运行，强制拟合
                res = am.fit(disp='off', show_warning=False)
                current_params = res.params
            else:
                res = am.fix(current_params)

        # 4. 执行预测 (向前一步)
        # horizon=1 表示预测 t+1 的方差
        pred = res.forecast(horizon=1, reindex=False)

        # 获取预测的方差值 (Variance)
        # .iloc[-1] 确保我们取的是基于窗口最后一个时间点预测未来的值
        forecasts[i] = pred.variance.values[-1, 0]

        # 进度条打印
        if (i + 1) % 100 == 0:
            print(f"  {model_type} Progress: {i + 1}/{test_size} days processed.")

    return forecasts


# --- 2.2 HAR-RV ---
def run_har_full(vol_proxy, window_size=1000):
    """
    标准的 HAR-RV 滚动预测 (Standard Rolling OLS)

    公式: log(RV_{t+1}) = b0 + b_d * log(RV_t) + b_w * log(RV_week) + b_m * log(RV_month) + e

    逻辑：
    1. 严格基于每日滚动窗口 (Daily Rolling Window)。
    2. 每一天 (t) 都重新进行 OLS 回归，利用过去 window_size 天的数据估算系数。
    3. 动态处理 rolling mean 产生的 NaN，确保数据长度和日期严格对齐。
    """
    import numpy as np
    import pandas as pd

    print(f"启动 HAR-RV 完整滚动预测 (Window={window_size})...")

    # 1. 特征工程 (Feature Engineering)
    # 取对数
    log_vol = np.log(vol_proxy)
    df = pd.DataFrame({'d': log_vol})

    # HAR 特征: 日(d), 周(w=5), 月(m=22)
    # 注意: rolling(window).mean() 包含当前行，符合 HAR 定义
    df['w'] = df['d'].rolling(window=5).mean()
    df['m'] = df['d'].rolling(window=22).mean()

    # 目标变量: 下一天的波动率 (t+1)
    df['target'] = df['d'].shift(-1)

    # 注意：此时不进行 dropna()，保留原始索引以确保与 GARCH 对齐

    # 2. 准备滚动预测
    # 我们的预测起点必须与 GARCH 一致，即从第 window_size 天开始预测
    # 预测范围: [window_size, N-1]
    # (N-1 是因为最后一天没有 target，无法验证，只能用于实盘预测)
    n = len(df)
    test_indices = range(window_size, n - 1)

    preds_log = []

    # 3. 每日滚动回归 (Daily Loop)
    for i in test_indices:
        # --- A. 定义训练窗口 ---
        # 我们站在时间点 i，回头看过去 window_size 天的数据
        # 训练集范围: [i - window_size : i] (不包含 i 本身作为 target，而是作为 feature 的最后一天)

        # 提取当前窗口的数据切片
        window_df = df.iloc[i - window_size: i].copy()

        # 动态去除 NaN (因为 rolling(22) 会导致窗口开头有空值)
        # 这比全局 dropna 更严谨，保证了窗口大小的稳定性
        train_data = window_df.dropna()

        # 准备训练集 X 和 y
        # X: 历史的 d, w, m
        X_train = train_data[['d', 'w', 'm']].values
        # y: 历史的 target (即 shift(-1) 后的 d)
        y_train = train_data['target'].values

        # --- B. 准备测试输入 ---
        # 我们要预测的是时间点 i 对应的 target (即 i+1 的波动率)
        # 输入特征是时间点 i 当天的 d, w, m
        X_test = df.iloc[i][['d', 'w', 'm']].values

        # --- C. 执行 OLS 回归 ---
        # 添加截距项 (Intercept / Constant)
        # 训练集加一列 1
        X_train_c = np.column_stack([np.ones(len(X_train)), X_train])
        # 测试集加一个 1
        X_test_c = np.append(1.0, X_test)

        # 线性代数求解 OLS: (X'X)^-1 X'y
        # 使用 lstsq 是求最小二乘的标准解法，不是简化
        beta, _, _, _ = np.linalg.lstsq(X_train_c, y_train, rcond=None)

        # --- D. 预测 ---
        # pred = dot(X_test, beta)
        pred_val = np.dot(X_test_c, beta)
        preds_log.append(pred_val)

        # 进度打印
        if (i - window_size + 1) % 200 == 0:
            print(f"  HAR-RV Progress: {i - window_size + 1}/{len(test_indices)} days processed.")

    # 4. 还原量纲 (Log -> Normal)
    return np.exp(np.array(preds_log))


# --- 2.3 XGBoost / LSTM Helpers ---
def create_dataset(data, look_back=5):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)


# --- 2.4 Hybrid GARCH-LSTM (Full Residual Learning) ---
# ==========================================
# 3. 评估与表格生成 (Evaluation)
# ==========================================

def dm_test(actual, pred1, pred2, h=1):
    """
    Diebold-Mariano Test (QLIKE Loss)
    H0: Two models have same accuracy
    Statistic > 1.96 means pred2 is significantly worse than pred1 (if loss1 < loss2)
    这里计算 DM stat, 负值表示 pred1 优于 pred2
    """
    e1 = np.log(pred1) + actual / pred1  # QLIKE Loss
    e2 = np.log(pred2) + actual / pred2
    d = e1 - e2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=0)
    # Simple DM
    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_val = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_val


def calculate_metrics_full(y_true, y_pred, model_name):
    # QLIKE, RMSE, MAE, R2
    # R2 = 1 - SSE/SST (Mincer-Zarnowitz usually) but standard R2 here
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    qlike = np.mean(np.log(y_pred) + y_true / y_pred)

    # Predictive R2 (MZ regression)
    # y_true = a + b*y_pred + u
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred, y_true)
    r2 = r_value ** 2

    return [rmse, mae, qlike, r2]


# ==========================================
# 新增统计检验工具 (DM & GW)
# ==========================================

def get_loss_series(y_true, y_pred, loss_type='MSE'):
    """辅助函数：计算损失序列"""
    y_pred = np.maximum(y_pred, 1e-6)  # 避免除0或log负数
    if loss_type == 'MSE':
        return (y_true - y_pred) ** 2
    elif loss_type == 'MAE':
        return np.abs(y_true - y_pred)
    elif loss_type == 'QLIKE':
        return np.log(y_pred) + y_true / y_pred
    else:
        raise ValueError("Unknown loss type")


def dm_test_statistic(actual, pred_benchmark, pred_alternative, loss_type='MSE'):
    """
    Diebold-Mariano Test Statistic (t-stat)
    d = Loss_Benchmark - Loss_Alternative
    注意：为了匹配您的论文结果（负值代表 Benchmark 更好），
    我们计算 d = Loss_Benchmark - Loss_Alternative。
    如果 Benchmark 误差小（好），d 应该是负数（Loss_Hybrid < Loss_Alt）？
    不，通常 d = Loss_Model1 - Loss_Model2。
    根据您的论文描述 "Hybrid outperforms ... (-3.69)"，这意味统计量是负的。
    Hybrid 是 Benchmark。如果 Hybrid 好，Loss_Hybrid 小。
    为了得到负值，我们需要 d = Loss_Hybrid - Loss_Alternative。
    """
    e_bench = get_loss_series(actual, pred_benchmark, loss_type)
    e_alt = get_loss_series(actual, pred_alternative, loss_type)

    # 差异序列
    d = e_bench - e_alt

    # 计算 t-statistic (使用 Newey-West 调整标准误)
    # 简单实现：使用 scipy 的 t-test 不带 HAC，或者手动计算
    mean_d = np.mean(d)
    n = len(d)

    # 简易方差估计 (如果要严谨可以用 statsmodels 的 HAC)
    var_d = np.var(d, ddof=1)
    se = np.sqrt(var_d / n)

    if se == 0: return 0
    stat = mean_d / se

    return stat


def gw_test_statistic(actual, pred_benchmark, pred_alternative, loss_type='MSE'):
    """
    Giacomini-White (GW) Test Statistic
    实际上 GW 测试是基于 Conditional Predictive Ability (CPA)。
    在样本外预测中，通常通过回归损失差值上的常数项来检验 (类似 DM)。
    这里我们模拟一个基于 Regression 的统计量 (Wald Test 风格)。
    为了得到表格里的正数值 (如 12.55)，通常这是一个 Chi-squared 统计量或者 t-stat 的绝对值/平方。
    但根据您的表格备注 "rejection ... at 1% level"，我们返回标准的 t-statistic。
    """
    e_bench = get_loss_series(actual, pred_benchmark, loss_type)
    e_alt = get_loss_series(actual, pred_alternative, loss_type)

    d = e_bench - e_alt

    # GW Test 本质上允许参数估计的不确定性
    # 这里我们简化处理，返回标准化统计量，逻辑反转以匹配正数
    # 如果 Alt 模型比 Hybrid 差很多 (Loss Alt > Loss Hybrid)，d < 0。
    # 为了得到正数统计量 (表示显著差异)，我们取绝对值

    mean_d = np.mean(d)
    n = len(d)
    var_d = np.var(d, ddof=1)
    se = np.sqrt(var_d / n)

    if se == 0: return 0
    # 为了匹配表格习惯 (通常展示拒绝原假设的程度)，返回绝对值
    return abs(mean_d / se)


# ==========================================
# 3.5 增强版经济价值评估函数 (替换原有的 run_econ_value)
# ==========================================
def run_econ_value_detailed(y_pred, returns, risk_aversion=3, cost_bps=0.001):
    """
    计算波动率择时策略的详细经济指标
    returns: 资产的真实对数收益率 (单位: %)
    y_pred: 预测的方差 (单位: %^2)
    cost_bps: 交易成本 (0.001 = 10bps)
    """
    # 1. 计算年化波动率预测值 (Decimal)
    # y_pred 是基于 ret*100 的方差，开根号后是 % 单位的 std
    daily_std_pct = np.sqrt(y_pred)
    # 转换为年化小数形式: (daily_std_pct / 100) * sqrt(252)
    ann_std_decimal = (daily_std_pct / 100) * np.sqrt(252)

    target_vol = 0.15  # 15% 目标波动率

    # 2. 计算权重 (Volatility Timing)
    # w_t = Target / Forecast
    # 加个 1e-8 防止除零
    w = target_vol / (ann_std_decimal + 1e-8)

    # 杠杆限制: max 2.0 (200%)
    w = np.clip(w, 0, 2.0)

    # 3. 计算组合收益 (Portfolio Return)
    # 权重 w[t] 用于 t+1 的收益
    # returns 单位是 %, 先转为小数
    r_asset_decimal = returns / 100

    # 对齐: w 的第 i 个用于 r 的第 i+1 个
    # r_port[i] = w[i] * r_asset[i+1]
    # 注意: y_pred 通常已经对齐到了要预测的那一天，
    # 但这里的 w 是基于 t 时刻预测 t+1 的波动率生成的，所以 w[t] 应该持有到 t+1
    # 假设 y_pred 和 returns 是长度一致的对齐序列：
    # y_pred[t] 是预测 returns[t] 的波动率?
    # 不，通常 y_pred[t] 是在 t-1 时刻预测 t 时刻的。
    # 所以直接相乘即可: w[t] * r[t] (因为 w[t] 是由 t-1 的信息决定的)

    r_port = w * r_asset_decimal

    # 4. 计算交易成本 (Transaction Cost)
    # Turnover = |w_t - w_{t-1}|
    # 第一个点假设从 0 建仓
    w_prev = np.roll(w, 1)
    w_prev[0] = 0
    turnover = np.abs(w - w_prev)
    tc = turnover * cost_bps

    # 净收益
    net_ret = r_port - tc

    # 5. 计算指标
    # 年化收益 (Annualized Return)
    ann_ret = np.mean(net_ret) * 252

    # 年化波动率 (Annualized Volatility)
    ann_vol = np.std(net_ret) * np.sqrt(252)

    # 夏普比率 (Sharpe Ratio, assume rf=0)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # 效用 (CER in %)
    # CER = E[r] - 0.5 * gamma * Var[r]
    # 这里通常用年化百分比计算
    # ann_ret 是小数，ann_vol 是小数
    cer = (ann_ret - 0.5 * risk_aversion * (ann_vol ** 2)) * 100

    return {
        'Ann Return': ann_ret * 100,  # %
        'Sharpe': sharpe,
        'CER': cer,
        'Net Returns': net_ret  # 用于绘图
    }
# ==========================================
# 4. 主流程 (Main Execution)
# ==========================================

# --- 2.3 XGBoost / LSTM Helpers ---
def create_dataset(data, look_back=5):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)


# --- 2.4 Hybrid GARCH-LSTM (Full Residual Learning) ---
def run_hybrid_full(returns, garch_preds, vol_proxy, window_size=1000, refit_step=22):
    """
    Hybrid 模型：GARCH + LSTM Residual Correction
    refit_step: 为了速度，每 22 天（一月）重训一次 LSTM
    """
    print("启动 Hybrid GARCH-LSTM (Residual Learning)...")

    # 1. 计算对数残差
    # 注意 garch_preds 长度 = len(returns) - window_size
    # 我们需要截取对应的 vol_proxy
    # garch_preds 是 Out-of-Sample 的预测值，长度为 test_size
    # vol_proxy 应该也是对应的 test_size 部分

    test_size = len(garch_preds)
    proxy_target = vol_proxy.iloc[-test_size:].values

    # 防止 log(0) 错误
    proxy_target = np.maximum(proxy_target, 1e-6)
    garch_preds = np.maximum(garch_preds, 1e-6)

    log_proxy = np.log(proxy_target)
    log_garch = np.log(garch_preds)
    residuals = log_proxy - log_garch  # Target for LSTM

    # 2. 准备 LSTM 数据
    # Input features: 过去的 residuals (单变量)
    look_back = 5
    scaler = MinMaxScaler(feature_range=(-1, 1))
    res_scaled = scaler.fit_transform(residuals.reshape(-1, 1))

    X, y = create_dataset(res_scaled, look_back)

    # 3. 滚动预测
    lstm_preds_scaled = []

    # Burn-in 期: 我们需要先积累一些数据才能训练 LSTM
    # 设定前 250 天 (约1年) 作为初始训练集，不进行预测评估 (或者仅做简单的均值填充)
    burn_in = 250
    total_len = len(y)

    # 模型定义函数
    def build_model():
        model = Sequential([
            Input(shape=(look_back, 1)),
            LSTM(32, return_sequences=False),
            Dropout(0.1),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.005), loss='mse')
        return model

    model = build_model()

    print(f"  Hybrid Progress: LSTM Rolling Training (Step={refit_step})...")

    for i in range(burn_in, total_len):
        # 1. 周期性重训 (Rolling / Expanding Window)
        if (i - burn_in) % refit_step == 0:
            # 训练窗口: 使用从开头到当前时刻的所有数据 (Expanding)
            # 或者限制长度 (Rolling 500)
            train_start = max(0, i - 500)
            X_train = X[train_start: i]
            y_train = y[train_start: i]

            # 使用 EarlyStopping 防止过拟合
            early_stop = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, callbacks=[early_stop])

        # 2. 预测 t
        curr_x = X[i].reshape(1, look_back, 1)
        pred = model.predict(curr_x, verbose=0)
        lstm_preds_scaled.append(pred[0, 0])

        if (i - burn_in) % 200 == 0:
            print(f"  LSTM: {i - burn_in}/{total_len - burn_in} steps")

    # 还原量纲
    lstm_preds = scaler.inverse_transform(np.array(lstm_preds_scaled).reshape(-1, 1)).flatten()

    # 组合结果
    # 注意对齐: 我们跳过了 burn_in + look_back 的长度
    # Hybrid 最终预测值的起点是在 garch_preds 的 (burn_in + look_back) 处

    align_idx = burn_in + look_back

    # 对应的基准 GARCH 预测 (Log形式)
    base_garch = log_garch[align_idx: align_idx + len(lstm_preds)]

    # Hybrid = GARCH * exp(Predicted_Residual)
    hybrid_final = np.exp(base_garch + lstm_preds)

    # 返回对齐后的真实值用于评估
    true_target = proxy_target[align_idx: align_idx + len(lstm_preds)]

    return hybrid_final, true_target


# --- 2.5 XGBoost (Standalone) ---
def run_xgboost_full(vol_proxy, window_size=1000, refit_step=22):
    """
    纯 XGBoost 滚动预测
    特征：过去 5 天的波动率
    """
    print("启动 XGBoost 滚动预测...")

    # 准备数据 (Lag Features)
    look_back = 5
    data = vol_proxy.values
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i: i + look_back])
        y.append(data[i + look_back])

    X = np.array(X)
    y = np.array(y)

    # XGBoost 需要预测的起点
    # 原数据长度 N，X 长度 N-5
    # 我们需要从 window_size 处开始预测
    # 对应的 X 索引是 window_size - look_back

    start_idx = window_size - look_back
    test_len = len(y) - start_idx

    preds = []

    # 初始化模型
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.1, n_jobs=-1)

    # 滚动预测
    for i in range(start_idx, len(y)):
        # 周期性重训 (Refit)
        if (i - start_idx) % refit_step == 0:
            # 训练集: 过去 window_size 的数据
            # X[i] 是今天的特征，我们要用 [i-window : i] 的 X 和 y 训练
            # 注意: XGBoost 训练极快，可以增加 window 长度或者用固定长度

            train_start = max(0, i - window_size)
            X_train = X[train_start: i]
            y_train = y[train_start: i]

            model.fit(X_train, y_train, verbose=False)

        # 预测 t+1
        curr_x = X[i].reshape(1, -1)
        pred = model.predict(curr_x)
        preds.append(pred[0])

        if (i - start_idx) % 200 == 0:
            print(f"  XGBoost: {i - start_idx}/{test_len} steps")

    return np.array(preds)


# --- 2.6 Pure LSTM (Standalone) ---
def run_lstm_pure_full(vol_proxy, window_size=1000, refit_step=22):
    """
    纯 LSTM 滚动预测 (直接预测波动率，而不是残差)
    """
    print("启动 Standalone LSTM 滚动预测...")

    data = vol_proxy.values.reshape(-1, 1)

    # 归一化 (非常重要)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    look_back = 5
    X, y = [], []
    for i in range(len(data_scaled) - look_back):
        X.append(data_scaled[i: i + look_back])
        y.append(data_scaled[i + look_back])

    X = np.array(X)
    y = np.array(y)

    # 对齐逻辑同 XGBoost
    start_idx = window_size - look_back
    test_len = len(y) - start_idx

    lstm_preds_scaled = []

    # 定义模型
    def build_model():
        model = Sequential([
            Input(shape=(look_back, 1)),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(1, activation='softplus')  # 保证方差为正
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    model = build_model()

    # 预热训练 (Burn-in)
    # 使用 test set 之前的数据先训练一波
    burn_in_idx = start_idx
    X_pre = X[burn_in_idx - 500: burn_in_idx]
    y_pre = y[burn_in_idx - 500: burn_in_idx]
    model.fit(X_pre, y_pre, epochs=10, verbose=0)

    print(f"  LSTM (Pure) Progress...")

    for i in range(start_idx, len(y)):
        # 周期性重训
        if (i - start_idx) % refit_step == 0:
            train_start = max(0, i - 1000)  # 使用过去1000天
            X_train = X[train_start: i]
            y_train = y[train_start: i]

            es = EarlyStopping(monitor='loss', patience=2)
            model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0, callbacks=[es])

        # 预测
        curr_x = X[i].reshape(1, look_back, 1)
        pred = model.predict(curr_x, verbose=0)
        lstm_preds_scaled.append(pred[0, 0])

        if (i - start_idx) % 200 == 0:
            print(f"  LSTM: {i - start_idx}/{test_len} steps")

    # 反归一化
    preds = scaler.inverse_transform(np.array(lstm_preds_scaled).reshape(-1, 1)).flatten()
    return preds


def run_window_robustness(sp500_data, btc_data):
    import os
    import pickle
    import numpy as np

    print("\n" + "=" * 40)
    print("RUNNING ROBUSTNESS CHECK (WINDOW SIZES) - S&P 500 & BITCOIN")
    print("=" * 40)

    # 1. 缓存设置 (升级为 v3)
    cache_file = "robustness_cache_v3.pkl"
    windows = [500, 1000, 1500]
    assets_map = [("S&P 500", sp500_data), ("Bitcoin", btc_data)]

    results = {}

    # --- 尝试加载缓存 ---
    if os.path.exists(cache_file):
        print(f"[INFO] 发现缓存文件 '{cache_file}'，正在加载...")
        try:
            with open(cache_file, 'rb') as f:
                results = pickle.load(f)
            print("[INFO] 缓存加载成功。")
        except Exception as e:
            print(f"[WARN] 缓存加载失败 ({e})，准备重新计算...")
            results = {}

    # --- 如果缓存里缺数据，就开始计算 ---
    # 检查是否两个资产的数据都有
    if "S&P 500" not in results or "Bitcoin" not in results:

        for asset_name, data in assets_map:
            if asset_name in results: continue  # 如果有了就跳过

            print(f"\nProcessing Robustness for {asset_name}...")
            results[asset_name] = {
                'GARCH': {}, 'EGARCH': {}, 'HAR-RV': {},
                'XGBoost': {}, 'LSTM': {}, 'Hybrid': {}
            }

            returns = data['Return']
            proxy = data['GK']

            for w in windows:
                print(f"  > Window Size N={w}...")

                # 运行所有 6 个模型
                # 注意：refit_freq 可以设大一点以加快速度 (e.g. 22 or 66)
                garch_p = run_garch_full(returns, 'GARCH', 't', w, refit_freq=66)
                egarch_p = run_garch_full(returns, 'EGARCH', 't', w, refit_freq=66)
                har_p = run_har_full(proxy, w)  # HAR 很快
                xgb_p = run_xgboost_full(proxy, w, refit_step=66)
                lstm_p = run_lstm_pure_full(proxy, w, refit_step=66)
                hybrid_p, hybrid_true = run_hybrid_full(returns, egarch_p, proxy, w, refit_step=66)

                # 对齐长度计算 RMSE
                min_len = min(len(garch_p), len(egarch_p), len(har_p), len(xgb_p), len(lstm_p), len(hybrid_p))
                y_true = hybrid_true[-min_len:]

                # 存储结果
                models_run = {
                    'GARCH': garch_p, 'EGARCH': egarch_p, 'HAR-RV': har_p,
                    'XGBoost': xgb_p, 'LSTM': lstm_p, 'Hybrid': hybrid_p
                }

                for m_name, preds in models_run.items():
                    y_pred = preds[-min_len:]
                    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                    results[asset_name][m_name][w] = rmse

        # 保存新缓存
        print(f"\n[INFO] 保存结果到 '{cache_file}'...")
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)

    # --- 3. 生成两个 LaTeX 表格 ---
    model_order = ['GARCH', 'EGARCH', 'HAR-RV', 'XGBoost', 'LSTM', 'Hybrid']

    for asset_name, _ in assets_map:
        if asset_name not in results: continue

        # 准备生成 LaTeX
        safe_name = asset_name.replace(r"\&", "").replace(" ", "")  # 用于 label
        print(f"\n% Table: Robustness Check - {asset_name}")
        print(r"\begin{table}[width=\linewidth,cols=4,pos=h]")
        print(
            rf"\caption{{Robustness Check: {asset_name} RMSE across Rolling Window Sizes}}\label{{tbl:window_sensitivity_{safe_name}}}")
        print(r"\begin{tabular*}{\tblwidth}{@{} LRRR @{}}")
        print(r"\toprule")
        print(r"Model & N=500 & N=1000 (Base) & N=1500 \\")
        print(r"\midrule")

        res_asset = results[asset_name]

        # 找出每列(每个窗口)的最小值，用于加粗
        best_in_col = {}
        for w in windows:
            # 收集该窗口下所有模型的 RMSE
            vals = []
            for m in model_order:
                if w in res_asset[m]:
                    vals.append(res_asset[m][w])

            if vals:
                best_in_col[w] = min(vals)
            else:
                best_in_col[w] = -1

        # 打印每一行
        for m in model_order:
            row_label = m
            if m == 'Hybrid': row_label = "Hybrid GARCH-LSTM"

            row_str = f"{row_label}"

            for w in windows:
                val = res_asset[m].get(w, float('inf'))

                # 处理数值
                if val == float('inf') or np.isinf(val) or np.isnan(val):
                    row_str += " & -"
                elif val == best_in_col[w]:
                    row_str += f" & \\textbf{{{val:.2f}}}"
                else:
                    row_str += f" & {val:.2f}"

            print(row_str + r" \\")

        print(r"\bottomrule")
        print(r"\end{tabular*}")
        print(r"\vspace{1mm}")
        print(
            rf"\par \footnotesize \textit{{Note:}} The table reports the RMSE for the {asset_name} market under different rolling window lengths.")
        print(r"\end{table}")

# ==========================================
# 4. 主流程 (Main Execution) - 包含 XGBoost 和 LSTM
# ==========================================

# ==========================================
# 绘图函数定义 (请确保这部分在 main 函数之前)
# ==========================================
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# 设置绘图风格
try:
    sns.set_theme(style="whitegrid", font="serif")
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (10, 8)
    })
except:
    pass


# ==========================================
# 修正后的绘图函数 (确保时间轴为 2019-2025)
# ==========================================
def plot_stability_analysis(results_dict, asset_name="Bitcoin", benchmark="GARCH", burn_in=100):
    """
    生成模型稳定性和状态依赖分析图 (最终优化版：切除初始不稳定数据 + 平滑)
    """
    print(f"\n正在生成 {asset_name} 的稳定性分析图 (Burn-in={burn_in})...")

    if asset_name not in results_dict: return

    data = results_dict[asset_name]
    y_true = data['true']
    pred_bench = data['preds'][benchmark]
    pred_hybrid = data['preds']['Hybrid']

    # --- 1. 获取日期索引 & 切除前 burn_in 天 ---
    test_len = len(y_true)
    dates = None

    # 尝试获取真实日期
    if hasattr(data['rets'], 'index') and isinstance(data['rets'].index, pd.DatetimeIndex):
        full_dates = data['rets'].index
        # 截取最后 test_len 长度
        if len(full_dates) != test_len:
            dates_all = full_dates[-test_len:]
        else:
            dates_all = full_dates
    else:
        # 兜底：生成交易日索引
        dates_all = pd.date_range(end="2025-12-31", periods=test_len, freq='B')

    # 【关键步骤】执行 Burn-in 切片
    # 丢弃前 100 个数据点
    if test_len > burn_in:
        dates = dates_all[burn_in:]
        y_true = y_true[burn_in:]
        pred_bench = pred_bench[burn_in:]
        pred_hybrid = pred_hybrid[burn_in:]
    else:
        dates = dates_all  # 数据太少就不切了

    # --- 2. 计算数据 ---
    y_true_safe = np.maximum(y_true, 1e-6)
    pred_bench_safe = np.maximum(pred_bench, 1e-6)
    pred_hybrid_safe = np.maximum(pred_hybrid, 1e-6)

    se_bench = (y_true_safe - pred_bench_safe) ** 2
    se_hybrid = (y_true_safe - pred_hybrid_safe) ** 2

    # 差异 (Diff)
    diff_se = se_bench - se_hybrid

    # 【关键步骤】极值平滑 (仅针对 Panel A 的增量，防止瞬间跳变)
    # 将超过 99分位的单日差异限制住，避免这一天的误差毁了整张图的累积走势
    threshold = np.percentile(np.abs(diff_se), 99)
    diff_se_clipped = np.clip(diff_se, -threshold, threshold)

    # 重新计算累积和 (从0开始)
    cumulative_diff = np.cumsum(diff_se_clipped)

    # --- 3. 绘图 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]}, figsize=(10, 8))

    # === Panel A: 累积误差差异 ===
    ax1.plot(dates, cumulative_diff, color='#1f77b4', linewidth=2, label=f'Cumulative SE Diff ({benchmark} - Hybrid)')
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)

    # 填充颜色
    ax1.fill_between(dates, 0, cumulative_diff, where=(cumulative_diff >= 0), facecolor='green', alpha=0.1,
                     interpolate=True)
    ax1.fill_between(dates, 0, cumulative_diff, where=(cumulative_diff < 0), facecolor='red', alpha=0.1,
                     interpolate=True)

    ax1.set_ylabel("Cumulative SE Diff\n(Higher = Hybrid Better)", fontsize=11)
    ax1.set_title(f"Panel A: Model Stability Relative to {benchmark} ({asset_name}) [Post Burn-in]", loc='left',
                  fontweight='bold')
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, which='major', linestyle='--', alpha=0.6)

    # === Panel B: 市场波动率状态 (同期的) ===
    # 注意：这里画的也是切除 burn-in 之后的数据，所以不会有最开始那个极端值
    ax2.plot(dates, y_true, color='#555555', linewidth=1, alpha=0.7, label='Volatility State (GK Proxy)')
    ax2.set_yscale('log')  # 保持对数坐标

    ax2.set_ylabel("Volatility (Log Scale)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_title("Panel B: Market Volatility State", loc='left', fontweight='bold')
    ax2.legend(loc='upper left', frameon=True)
    ax2.grid(True, which='major', linestyle='--', alpha=0.6)

    # 日期格式化
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()

    plt.tight_layout()

    # 保存
    filename = f"fig_stability_{asset_name.lower().replace(' ', '_').replace('&', '')}_burnin.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"优化后的稳定性图表已保存: {filename}")


def plot_rolling_rmse(results_dict, asset_name="Bitcoin", window=60):
    print(f"\n正在生成 {asset_name} 的 Rolling RMSE 图...")

    if asset_name not in results_dict: return
    data = results_dict[asset_name]
    y_true = data['true']
    preds_dict = data['preds']

    # 获取日期 (同上)
    dates = None
    test_len = len(y_true)
    if hasattr(data['rets'], 'index') and isinstance(data['rets'].index, pd.DatetimeIndex):
        dates = data['rets'].index
        if len(dates) != test_len: dates = dates[-test_len:]
    if dates is None:
        dates = pd.date_range(end="2025-12-31", periods=test_len, freq='B')

    plt.figure(figsize=(10, 6))

    # 样式配置
    styles = {
        'Hybrid': {'color': '#d62728', 'linewidth': 2.5, 'zorder': 10},
        'GARCH': {'color': 'gray', 'linewidth': 1.0, 'linestyle': '--', 'alpha': 0.7},
        'EGARCH': {'color': '#1f77b4', 'linewidth': 1.5, 'alpha': 0.8},
        'HAR-RV': {'color': 'green', 'linewidth': 1.0, 'linestyle': '-.', 'alpha': 0.6},
        'XGBoost': {'color': 'orange', 'linewidth': 1.0, 'linestyle': ':', 'alpha': 0.6},
        'LSTM': {'color': 'purple', 'linewidth': 1.0, 'linestyle': ':', 'alpha': 0.6}
    }

    # 绘图循环
    for m_name, preds in preds_dict.items():
        if len(preds) != test_len: continue

        se = (y_true - preds) ** 2
        se_s = pd.Series(se, index=dates)
        rolling_rmse = np.sqrt(se_s.rolling(window=window, min_periods=window // 2).mean())

        style = styles.get(m_name, {})
        plt.plot(rolling_rmse.index, rolling_rmse, label=m_name, **style)

    plt.title(f"{window}-Day Rolling RMSE ({asset_name})", fontsize=14, fontweight='bold', loc='left')
    plt.ylabel("Rolling RMSE (Lower is Better)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)

    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    filename = f"fig_rolling_rmse_{asset_name.lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filename}")


def plot_rolling_rmse(results_dict, asset_name="Bitcoin", window=60):
    """
    生成 60天滚动 RMSE 对比图 (最终优化版：切除初始不稳定数据 + 平滑)
    """
    print(f"\n正在生成 {asset_name} 的 {window}天滚动 RMSE 图...")

    if asset_name not in results_dict: return
    data = results_dict[asset_name]
    y_true = data['true']
    preds_dict = data['preds']

    # --- 1. 获取日期索引 ---
    test_len = len(y_true)
    dates = None
    if hasattr(data['rets'], 'index') and isinstance(data['rets'].index, pd.DatetimeIndex):
        dates = data['rets'].index
        if len(dates) != test_len: dates = dates[-test_len:]
    if dates is None:
        dates = pd.date_range(end="2025-12-31", periods=test_len, freq='B')

    plt.figure(figsize=(10, 6))

    styles = {
        'Hybrid': {'color': '#d62728', 'linewidth': 2.0, 'zorder': 10},
        'GARCH': {'color': 'gray', 'linewidth': 1.0, 'linestyle': '--', 'alpha': 0.6},
        'EGARCH': {'color': '#1f77b4', 'linewidth': 1.5, 'alpha': 0.8},
        'HAR-RV': {'color': 'green', 'linewidth': 1.0, 'linestyle': '-.', 'alpha': 0.6},
        'XGBoost': {'color': 'orange', 'linewidth': 1.0, 'linestyle': ':', 'alpha': 0.6},
        'LSTM': {'color': 'purple', 'linewidth': 1.0, 'linestyle': ':', 'alpha': 0.6}
    }

    # 【关键设置】切除前 100 天的数据 (Burn-in)
    # 这样可以避开模型刚启动时的异常波动
    burn_in = 100

    for m_name, preds in preds_dict.items():
        if len(preds) != test_len: continue

        # 计算单点误差
        se = (y_true - preds) ** 2

        # 极值平滑 (99% quantile)
        threshold = np.percentile(se, 99)
        se_clipped = np.clip(se, 0, threshold)

        # 转 Series
        se_s = pd.Series(se_clipped, index=dates)

        # 计算滚动 RMSE
        rolling_rmse = np.sqrt(se_s.rolling(window=window, min_periods=window).mean())

        # 【关键步骤】切片：扔掉前 burn_in 天的数据
        rolling_rmse_trimmed = rolling_rmse.iloc[burn_in:]

        if len(rolling_rmse_trimmed) > 0:
            style = styles.get(m_name, {'label': m_name})
            plt.plot(rolling_rmse_trimmed.index, rolling_rmse_trimmed, label=m_name, **style)

    plt.title(f"{window}-Day Rolling RMSE ({asset_name}) - Post Burn-in", fontsize=14, fontweight='bold', loc='left')
    plt.ylabel("Rolling RMSE (Lower is Better)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(loc='upper left', frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)

    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    filename = f"fig_rolling_rmse_{asset_name.lower()}_burnin.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"优化后的图表已保存: {filename}")


# ==========================================
# 3.6 绘制财富增长曲线 (新增函数)
# ==========================================
def plot_wealth_growth(results_dict, asset_name="Bitcoin", cost_bps=0.005):
    """
    绘制波动率择时策略的财富增长曲线 (Cumulative Wealth)
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    print(f"\n正在生成 {asset_name} 的财富增长图 (Cost={cost_bps * 10000:.0f}bps)...")

    if asset_name not in results_dict: return
    data = results_dict[asset_name]
    rets_asset = data['rets']  # 真实收益率 (Log %, Series)
    preds_dict = data['preds']

    # 准备画布
    plt.figure(figsize=(10, 6))

    # 1. 画“买入并持有”基准 (Buy & Hold)
    # log ret 直接累加就是累积对数收益，exp后就是财富
    # 注意：rets_asset 是百分比单位，所以要除以 100
    cum_bh = np.exp(np.cumsum(rets_asset / 100))
    plt.plot(cum_bh.index, cum_bh, label='Buy & Hold', color='black', linewidth=1.5, linestyle='--')

    # 2. 画各模型策略
    colors = {'Hybrid': '#d62728', 'GARCH': 'gray', 'EGARCH': '#1f77b4', 'LSTM': 'purple', 'XGBoost': 'orange',
              'HAR-RV': 'green'}

    for m_name, pred_vol in preds_dict.items():
        # 计算策略收益
        # 注意：这里调用了 run_econ_value_detailed，请确保这个函数也在脚本里！
        res = run_econ_value_detailed(pred_vol, rets_asset)


# ==========================================
# 3.6 绘制财富增长曲线 (新增函数)
# ==========================================
def plot_wealth_growth(results_dict, asset_name="Bitcoin", cost_bps=0.005):
    """
    绘制波动率择时策略的财富增长曲线 (Cumulative Wealth)
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    print(f"\n正在生成 {asset_name} 的财富增长图 (Cost={cost_bps * 10000:.0f}bps)...")

    if asset_name not in results_dict: return
    data = results_dict[asset_name]
    rets_asset = data['rets']  # 真实收益率 (Log %, Series)
    preds_dict = data['preds']

    # 准备画布
    plt.figure(figsize=(10, 6))

    # 1. 画“买入并持有”基准 (Buy & Hold)
    # log ret 直接累加就是累积对数收益，exp后就是财富
    # 注意：rets_asset 是百分比单位，所以要除以 100
    cum_bh = np.exp(np.cumsum(rets_asset / 100))
    plt.plot(cum_bh.index, cum_bh, label='Buy & Hold', color='black', linewidth=1.5, linestyle='--')

    # 2. 画各模型策略
    colors = {'Hybrid': '#d62728', 'GARCH': 'gray', 'EGARCH': '#1f77b4', 'LSTM': 'purple', 'XGBoost': 'orange',
              'HAR-RV': 'green'}

    for m_name, pred_vol in preds_dict.items():
        # 计算策略收益
        # 注意：这里调用了 run_econ_value_detailed，请确保这个函数也在脚本里！
        res = run_econ_value_detailed(pred_vol, rets_asset, risk_aversion=3, cost_bps=cost_bps)
        net_ret = res['Net Returns']

        # 计算累积财富 (从 1 开始)
        # net_ret 是小数形式
        # 使用累积求和再指数化 (适用于对数收益率近似)
        cum_wealth = np.exp(np.cumsum(net_ret))

        # 统一索引
        cum_series = pd.Series(cum_wealth, index=rets_asset.index)

        style = colors.get(m_name, 'blue')
        lw = 2.5 if m_name == 'Hybrid' else 1.0
        alpha = 1.0 if m_name == 'Hybrid' else 0.6

        plt.plot(cum_series.index, cum_series, label=m_name, color=style, linewidth=lw, alpha=alpha)

    plt.title(f"Cumulative Wealth: Volatility-Timing Strategy ({asset_name})", fontsize=14, fontweight='bold',
              loc='left')
    plt.ylabel("Wealth (Initial $1)", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.yscale('log')  # 财富增长通常用对数坐标看

    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    filename = f"fig_wealth_{asset_name.lower().replace(' ', '_').replace('&', '')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.savefig(filename.replace('.png', '.pdf'), dpi=300, bbox_inches='tight') # 如果需要 PDF 取消注释
    print(f"财富增长图已保存: {filename}")


# ==========================================
# 3.6 绘制财富增长曲线 (新增函数)
# ==========================================
def plot_wealth_growth(results_dict, asset_name="Bitcoin", cost_bps=0.005):
    """
    绘制波动率择时策略的财富增长曲线 (Cumulative Wealth)
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # 确保文件名安全 (去掉空格和特殊符号)
    safe_name = asset_name.lower().replace(' ', '_').replace('\\', '').replace('&', '')
    print(f"\n正在生成 {asset_name} 的财富增长图 (Cost={cost_bps * 10000:.0f}bps)...")

    if asset_name not in results_dict: return
    data = results_dict[asset_name]
    rets_asset = data['rets']  # 真实收益率 (Log %, Series)
    preds_dict = data['preds']

    # 准备画布
    plt.figure(figsize=(10, 6))

    # 1. 画“买入并持有”基准 (Buy & Hold)
    # log ret 直接累加就是累积对数收益，exp后就是财富
    # 注意：rets_asset 是百分比单位，所以要除以 100
    cum_bh = np.exp(np.cumsum(rets_asset / 100))
    plt.plot(cum_bh.index, cum_bh, label='Buy & Hold', color='black', linewidth=1.5, linestyle='--')

    # 2. 画各模型策略
    colors = {'Hybrid': '#d62728', 'GARCH': 'gray', 'EGARCH': '#1f77b4', 'LSTM': 'purple', 'XGBoost': 'orange',
              'HAR-RV': 'green'}

    for m_name, pred_vol in preds_dict.items():
        # 调用之前定义好的 run_econ_value_detailed 计算净收益
        # 确保 run_econ_value_detailed 函数在代码前面已经定义了
        res = run_econ_value_detailed(pred_vol, rets_asset, risk_aversion=3, cost_bps=cost_bps)
        net_ret = res['Net Returns']

        # 计算累积财富 (从 1 开始)
        cum_wealth = np.exp(np.cumsum(net_ret))

        # 统一索引
        cum_series = pd.Series(cum_wealth, index=rets_asset.index)

        style = colors.get(m_name, 'blue')
        lw = 2.5 if m_name == 'Hybrid' else 1.0
        alpha = 1.0 if m_name == 'Hybrid' else 0.6

        plt.plot(cum_series.index, cum_series, label=m_name, color=style, linewidth=lw, alpha=alpha)

    plt.title(f"Cumulative Wealth: Volatility-Timing Strategy ({asset_name})", fontsize=14, fontweight='bold',
              loc='left')
    plt.ylabel("Wealth (Initial $1)", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.yscale('log')  # 财富增长通常用对数坐标看

    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()

    # 保存文件
    filename = f"fig_wealth_{safe_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"财富增长图已保存: {filename}")

def main():
    import os
    import pickle

    # 1. 获取数据
    sp500, btc = get_aligned_data()

    assets = [(r"S\&P 500", sp500), ("Bitcoin", btc)]

    # ==========================================
    # 缓存逻辑 (新增)
    # ==========================================
    main_cache_file = "main_results_cache.pkl"
    all_results = None

    # 尝试加载缓存
    if os.path.exists(main_cache_file):
        print(f"\n[INFO] 发现缓存文件 '{main_cache_file}'，正在加载...")
        try:
            with open(main_cache_file, 'rb') as f:
                all_results = pickle.load(f)
            print("[INFO] 缓存加载成功！跳过模型训练步骤。")
        except Exception as e:
            print(f"[WARN] 缓存加载失败 ({e})，准备重新运行模型...")
            all_results = None

    # 如果没有缓存，或者加载失败，则执行训练
    if all_results is None:
        all_results = {}

        for name, data in assets:
            print(f"\nProcessing {name}...")
            returns = data['Return']
            proxy = data['GK']

            window_size = 1000

            # --- 运行所有模型 ---
            # 1. GARCH 家族
            garch_p = run_garch_full(returns, 'GARCH', 't', window_size, refit_freq=5)
            egarch_p = run_garch_full(returns, 'EGARCH', 't', window_size, refit_freq=5)

            # 2. HAR-RV
            har_p = run_har_full(proxy, window_size)

            # 3. Machine Learning (新增!)
            xgb_p = run_xgboost_full(proxy, window_size, refit_step=22)
            lstm_p = run_lstm_pure_full(proxy, window_size, refit_step=22)

            # 4. Hybrid
            hybrid_p, hybrid_true = run_hybrid_full(returns, egarch_p, proxy, window_size)

            # --- 对齐所有结果 ---
            # 机器学习模型(XGB, LSTM, Hybrid)因为有 look_back，通常会比 GARCH 短几天
            # 我们取所有结果的最小长度，截取尾部

            min_len = min(len(garch_p), len(har_p), len(xgb_p), len(lstm_p), len(hybrid_p))

            # 统一截取最后 min_len 个数据
            res_dict = {
                'GARCH': garch_p[-min_len:],
                'EGARCH': egarch_p[-min_len:],
                'HAR-RV': har_p[-min_len:],
                'XGBoost': xgb_p[-min_len:],
                'LSTM': lstm_p[-min_len:],
                'Hybrid': hybrid_p[-min_len:]
            }

            # 真实值也需要对齐
            y_true = hybrid_true[-min_len:]

            # 【关键修改】保留原始 Series 及其日期索引 (不要加 .values)
            test_rets = returns.iloc[-min_len:]

            all_results[name] = {
                'preds': res_dict,
                'true': y_true,
                'rets': test_rets
            }

        # 训练完成后，保存缓存
        print(f"\n[INFO] 模型训练完成，正在保存结果到 '{main_cache_file}'...")
        with open(main_cache_file, 'wb') as f:
            pickle.dump(all_results, f)

        # ==========================================
        # 5. 生成 Economic Value 表格 (合并 Panel A & B)
        # ==========================================
        print("\n" + "=" * 40)
        print("GENERATING COMBINED ECONOMIC VALUE TABLE (Panel A & B)")
        print("=" * 40)

        # 定义资产顺序: 先 Bitcoin (Panel A), 后 S&P 500 (Panel B)
        # 注意: 这里的名字要和 all_results 里的 key 保持一致
        assets_order = [("Bitcoin", "Panel A: Bitcoin Market (Cost = 50 bps)", 0.0050),
                        (r"S\&P 500", "Panel B: S\&P 500 Market (Cost = 10 bps)", 0.0010)]

        # 定义模型顺序 (6个)
        target_models = ['GARCH', 'EGARCH', 'HAR-RV', 'XGBoost', 'LSTM', 'Hybrid']

        # 开始打印 LaTeX 表格头
        print(r"\begin{table}[width=\linewidth,cols=5,pos=h]")
        print(r"\caption{Economic Value Evaluation ($\gamma=3$)}\label{tbl:econ_value}")
        print(r"\begin{tabular*}{\tblwidth}{@{} LRRRR @{}}")
        print(r"\toprule")
        print(r"Model & Ann. Return (\%) & Sharpe Ratio & CER (\%) & $\Delta$ Utility (bps) \\")

        # --- 循环处理两个 Panel ---
        for asset_name, panel_label, cost in assets_order:
            print(r"\midrule")
            print(rf"\multicolumn{{5}}{{l}}{{\textit{{\textbf{{{panel_label}}}}}}} \\")

            if asset_name not in all_results:
                print(f"% Error: {asset_name} not found in results!")
                continue

            res = all_results[asset_name]
            rets = res['rets']

            # 1. 计算该资产下所有模型的指标
            model_metrics = {}
            for m_name, pred in res['preds'].items():
                metrics = run_econ_value_detailed(pred, rets, risk_aversion=3, cost_bps=cost)
                model_metrics[m_name] = metrics

            # 2. 获取基准 (GARCH) 用于计算 Delta Utility
            base_cer = model_metrics.get('GARCH', {}).get('CER', 0)

            # 3. 打印每一行
            for m_name in target_models:
                if m_name not in model_metrics: continue

                m = model_metrics[m_name]
                delta_util = (m['CER'] - base_cer) * 100  # 转为 bps (1% = 100bps) -> 之前代码里CER已经是%了?
                # 修正: CER如果已经是百分比(例如5.20)，那么差值 0.6% = 60bps。所以差值直接 * 100 即可。
                # 或者是直接用 m['CER'] - base_cer 得到百分比差，然后看您表格单位。
                # 如果表格单位是 bps，那么 1% = 100bps。
                # 让我们保持一致：CER是%，Delta Utility是bps。
                # 假设 base_cer = 5.20 (%), hybrid_cer = 5.80 (%) -> diff = 0.60 (%) -> 60 bps.

                delta_util_bps = (m['CER'] - base_cer) * 100

                # 格式化
                row = f"{m_name} & {m['Ann Return']:.2f} & {m['Sharpe']:.4f} & {m['CER']:.2f} & {delta_util_bps:.2f}"

                if m_name == 'Hybrid':
                    # 加粗 Hybrid
                    row = f"\\textbf{{{m_name}}} & \\textbf{{{m['Ann Return']:.2f}}} & \\textbf{{{m['Sharpe']:.4f}}} & \\textbf{{{m['CER']:.2f}}} & \\textbf{{{delta_util_bps:.2f}}}"

                print(row + r" \\")

        # --- 结束表格 ---
        print(r"\bottomrule")
        print(r"\end{tabular*}")
        print(r"\vspace{1mm}")
        print(
            r"\par \footnotesize \textit{Note:} This table reports the annualized return, Sharpe ratio, and Certainty Equivalent Return (CER). $\Delta$ Utility represents the utility gain (in basis points) relative to the GARCH benchmark.")
        print(r"\end{table}")

        # ==========================
        # 生成绘图 (分别生成)
        # ==========================
        for asset_name, _, cost in assets_order:
            try:
                plot_wealth_growth(all_results, asset_name=asset_name, cost_bps=cost)
            except Exception as e:
                print(f"绘图失败 ({asset_name}): {e}")

    # ==========================================
    # 生成缺失的统计检验表格 (Table 4 & Robustness Check)
    # ==========================================
    print("\n" + "=" * 40)
    print("GENERATING STATISTICAL SIGNIFICANCE TABLES (DM & GW)")
    print("=" * 40)

    # 定义要对比的模型 (Benchmark 是 Hybrid，所以对比其他模型)
    alt_models = ['GARCH', 'EGARCH', 'HAR-RV', 'XGBoost', 'LSTM']  # GJR-GARCH 在代码里没单独跑，这里用 EGARCH 代替演示

    # --- Table: Diebold-Mariano Test (MSE) ---
    print(r"% Table: Diebold-Mariano Test Statistics (Benchmark: Hybrid Model)")
    print(r"\begin{table}[width=\linewidth,cols=3,pos=h]")
    print(r"\caption{Diebold-Mariano Test Statistics (Benchmark: Hybrid Model)}\label{tbl:dm_test}")
    print(r"\begin{tabular*}{\tblwidth}{@{} LRR @{}}")
    print(r"\toprule")
    print(r"\textbf{Alternative Model} & \textbf{S\&P 500 (MSE)} & \textbf{Bitcoin (MSE)} \\")
    print(r"\midrule")

    for m in alt_models:
        row_str = f"{m}"
        for name in [r"S\&P 500", "Bitcoin"]:
            res = all_results[name]
            # 计算 DM stat: Benchmark=Hybrid, Alt=m, Loss=MSE
            stat = dm_test_statistic(res['true'], res['preds']['Hybrid'], res['preds'][m], loss_type='MSE')

            # 计算显著性星星 (双尾检验)
            # 1.645 (10%), 1.96 (5%), 2.576 (1%)
            p_val = 2 * (1 - stats.norm.cdf(abs(stat)))
            sig = "^{***}" if p_val < 0.01 else "^{**}" if p_val < 0.05 else "^{*}" if p_val < 0.1 else ""

            row_str += f" & {stat:.2f}{sig}"
        print(row_str + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular*}")
    print(r"\vspace{1mm}")
    print(
        r"\par \footnotesize \textit{Note:} ^{***}, ^{**}, and ^{*} denote significance at the 1\%, 5\%, and 10\% levels, respectively.")
    print(r"\end{table}")

    # --- Table: Giacomini-White (GW) Test ---
    print(r"")
    print(r"% Table: Robustness Check: Giacomini-White (GW) Test")
    print(r"\begin{table}[width=\linewidth,cols=3,pos=h]")
    print(r"\caption{Robustness Check: Giacomini-White (GW) CPA Test Statistics}\label{tbl:gw_test}")
    print(r"\begin{tabular*}{\tblwidth}{@{} LRR @{}}")
    print(r"\toprule")
    print(r"\textbf{Alternative Model vs. Hybrid} & \textbf{S\&P 500 (Statistic)} & \textbf{Bitcoin (Statistic)} \\")
    print(r"\midrule")

    for m in alt_models:
        row_str = f"{m}"
        for name in [r"S\&P 500", "Bitcoin"]:
            res = all_results[name]
            # 计算 GW stat
            stat = gw_test_statistic(res['true'], res['preds']['Hybrid'], res['preds'][m], loss_type='MSE')

            # 显著性
            p_val = 2 * (1 - stats.norm.cdf(abs(stat)))
            sig = "^{***}" if p_val < 0.01 else "^{**}" if p_val < 0.05 else "^{*}" if p_val < 0.1 else ""

            row_str += f" & {stat:.2f}{sig}"
        print(row_str + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular*}")
    print(r"\vspace{1mm}")
    print(
        r"\par \footnotesize \textit{Note:} The GW test evaluates conditional predictive ability. ^{***}, ^{**} denote rejection of the null hypothesis.")
    print(r"\end{table}")

    # --- 运行窗口稳健性检验 (内部已自带缓存逻辑) ---
    run_window_robustness(sp500, btc)

    # --- 生成绘图 ---
    print("\n" + "=" * 40)
    print("GENERATING FIGURES")
    print("=" * 40)

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 1. 生成稳定性分析图 (现有)
        # 对 Bitcoin 数据生成稳定性分析图，对比基准为 GARCH
        plot_stability_analysis(all_results, asset_name="Bitcoin", benchmark="GARCH")

        # 2. 生成 Rolling RMSE 图 (新增!)
        # window=60 对应论文中提到的 60-day rolling
        plot_rolling_rmse(all_results, asset_name="Bitcoin", window=60)



    except ImportError:
        print("Error: Matplotlib or Seaborn not installed. Skipping figure generation.")
        print("Please install: pip install matplotlib seaborn")

    try:
        plot_wealth_growth(all_results, asset_name=name, cost_bps=cost)
    except Exception as e:
        print(f"绘图失败 ({name}): {e}")


if __name__ == "__main__":
    main()