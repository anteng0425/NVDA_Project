# NVDA預測 Hybrid (ARIMA + LSTM) 與多模型比較 (1)

## 1. 專案背景與需求

本專案以 **NVIDIA (NVDA) 股價資料** (包含 `date`, `adj_close` 等欄位) 作為基礎，運用 **ARIMA 與 LSTM 結合的混合模型 (Hybrid)** 來進行股價預測。

除了 **Hybrid** 模型之外，亦包含下列多種模型以進行比較：

1. **Auto ARIMA + LSTM (Hybrid)**
2. **(1,1,1) ARIMA + LSTM (Hybrid)**
3. **Pure LSTM**
4. **Pure Auto ARIMA**
5. **Pure (1,1,1) ARIMA**
6. **Naive Forecast**

最終將對各個模型在測試集中的 **RMSE**、**MAPE**、**方向準確率 (ACC)**、**R^2** 進行比較與分析。

---

## 2. 資料來源與期間

- 資料檔案路徑：
    
    ```
    C:/Users/wu504/Downloads/NVDA_Project/NVDA_stock_data.csv
    
    ```
    
- 初始的 CSV 檔案中欄位較多，僅需使用 `Date` 與 `Adj Close` 欄位。
- 將資料範圍「**截取至 2023-03-14(不含當日)**」，並於該日之後的資料全部丟棄。

經過前處理後，資料時間跨度約為：

```
2020-01-02  ~  2023-03-13

```

總筆數為 804筆。

---

## 3. 資料前處理

### 3.1 讀取 CSV

```python
df = pd.read_csv("/mnt/data/NVDA_stock_data.csv")

```

由於給定的 CSV 檔案裡，第 1 列實際上為重複標題資料，故我們會在讀取後執行：

```python
df = df.drop(index=0)

```

### 3.2 欄位名稱轉換

- 將欄位命名標準化，如下所示：
    - `Date` → `date`
    - `Adj Close` → `adj_close`
    - 其餘欄位若無需使用則會直接丟棄。

### 3.3 刪除缺失值與型態轉換

- 針對 `date` 欄位，若有缺失或非預期格式則刪除。
- 轉換 `date` 欄位為 `datetime` 型態。
- 將 `adj_close` 轉為 float 型態（若遇到不能轉換的資料則刪除該筆）。

### 3.4 資料過濾至 2023-03-14

```python
df = df[df['date'] < '2023-03-14']

```

並重新排序、重設索引。

### 3.5 最終資料確認

篩選完之後，最終可用的資料筆數約為 804（依實際狀況為準），日期從 `2020-01-02` 到 `2023-03-13`。

---

## 4. 資料切割

### 4.1 切割原則

1. 從完整資料 (截至 `2023/03/13`) 中，先切出 **80%** 為「訓練+驗證 (train_val)」，剩餘 **20%** 為「測試集 (test)」。
2. 在「訓練+驗證集」內再切出 **80%** 作為最終 **訓練集 (train)**，**20%** 作為 **驗證集 (val)**。

### 百分比

- 訓練集約為：**64%**
- 驗證集約為：**16%**
- 測試集約為：**20%**

### 訓練集 (`train_set`)

- **資料筆數**：514 筆
- **日期範圍**：2020-01-02 到 2022-01-13

### 驗證集 (`val_set`)

- **資料筆數**：129 筆
- **日期範圍**：2022-01-14 到 2022-07-21

### 測試集 (`test_set`)

- **資料筆數**：161 筆
- **日期範圍**：2022-07-22 到 2023-03-13

---

### 4.2 程式實作 (概念)

```python
tv_size = int(len(df) * 0.8)
train_val_df = df.iloc[:tv_size]   # 前 80%
test_df = df.iloc[tv_size:]        # 後 20%

t_size = int(len(train_val_df) * 0.8)
train_df = train_val_df.iloc[:t_size]  # (train_val 的) 前 80%
val_df = train_val_df.iloc[t_size:]    # (train_val 的) 後 20%

```

---

## 5. 模型架構與方法

### 5.1.1 滾動式預測 (Rolling Forecast)

根據需求需撰寫一「滾動式預測函數」，對**測試集進行預測**時，必須「**逐日**」滾動預測：

1. 以前一日的移動窗口輸入到模型並預測 **當日** (或隔日) 的股價。
2. 記錄預測值與真實值。
3. 更新模型的滑動窗口，將當日的「真實值」再行加入模型的滑動窗口，以便預測次日。

然後重複進行預測，重點在於每次預測時模型會獲得前一日的真實資訊（或從真實資訊衍生出來的殘差值），以便進行滾動式的預測（換言之模型每次都只有在預測下一日的股價）。

### 5.1.2 整段軌跡預測 (Whole Trajectory Forecast)

根據需求需撰寫一「整段軌跡預測函數」，對**測試集進行預測**時，必須一次性對整段軌跡進行預測：

1. 以驗證集到測試集的第一個資料點前的滑動窗口輸入模型來預測測試集的第一個資料點的數值。
2. 以前一日預測的結果（或從預測結果衍生出來的殘差值）來作為模型更新滑動窗口的輸入。
3. 持續反覆預測直到整段測試集的內容都被迭代預測完成。

此方式與前述「滾動式預測」之最大差異在前開方法於每次預測後將當日真實數值放入模型之輸入之中，而整段預測方法為唯有在預測第一筆資料時模型能獲得真實之資訊，其後模型皆採用預測結果作為新的滑動窗口輸入，不會獲得任何真實之股價資訊。

### 5.2 具體模型

### 5.2.1 **(1,1,1) ARIMA**

- 參數 `(p, d, q) = (1, 1, 1)`。
- 使用 `statsmodels.tsa.arima.model.ARIMA`。
- 訓練 (in-sample) 時直接 `fit()`；無 Early Stopping 概念。

### 5.2.2 **Auto ARIMA**

- 使用 `pmdarima.auto_arima()` 函式，內部以 AIC (或 BIC) 選出最佳 `(p, d, q)`。
- `seasonal=False`、`start_p=0, start_q=0, max_p=5, max_q=5`。

### 5.2.3 **Naive Forecast**

- 直接以「昨日真實價格」作為今日預測值。
- 同樣遵循「逐日滾動」：預測日股價後，再取得真實價格以作為下一日的基準。

### 5.2.4 **Pure LSTM**

1. **資料序列**：只使用 `adj_close`。
2. **滑動窗口 (window)**：20 天 → 代表 LSTM 輸入維度 `(batch, 20, 1)`。
3. **模型結構**：
    - LSTM(32, `return_sequences=True`)
    - LSTM(16, `return_sequences=False`)
    - Dense(16, activation='relu')
    - Dense(1)
4. **優化器**：`Adam`；損失函式：`MSE`。
5. **EarlyStopping**：`monitor='val_loss', patience=30, restore_best_weights=True`。
6. **批次大小 (batch_size)**：128；**epochs**：300 。
7. **滾動預測**時，每天需要把最新的真實股價新增到序列後，再用「最近 20 筆」的真實股價做為輸入，以預測下一天。
8. 於第一與第二層、第三層與輸出之間置入一dropout（20%）。
9. 激活函數使用ReLU。

### 5.2.5 **ARIMA + LSTM (Hybrid)**

1. **ARIMA** → 預測股價（in-sample or 未來一天）。
2. **LSTM** → 預測「殘差」(Residual) = (實際 - ARIMA 預測)。
3. **最終預測** = ARIMA 預測 + LSTM(殘差) 預測。

### 5.2.5.1 訓練階段

- 先在 (train+val) 數據上用 ARIMA 做 in-sample 預測，算出 `residual = actual - arima_pred`。
- 利用該 `residual` 當作新的「時間序列」來訓練 LSTM (同樣使用 window=20)。
- LSTM 架構同前 (兩層 LSTM + Dense)；`EarlyStopping` 同前。

### 5.2.5.2 測試 (滾動預測) 階段

對測試集逐日進行「滾動式預測」與「整段軌跡預測」：

1. 透過前幾日數值輸入（滾動式預測時輸入為真實股價，而整段軌跡預測時除第一次預測使用真實股價外，其餘皆為透過前幾日之預測結果來作為輸入）以ARIMA預測隔日股價 `arima_pred`。
2. 以最近 20 天的「殘差序列」(實際 - ARIMA in-sample) 進行 LSTM 輸入，得到下一日殘差預測 `lstm_res`.
3. **最終預測** = `arima_pred + lstm_res`.
4. 更新滑動視窗的數值，反覆執行預測直到測試集皆預測完成。

---

## 6. 訓練過程紀錄

以下紀錄主要針對 **LSTM 與 Hybrid** 部分，由於 ARIMA 無 Early Stopping，所以僅做一次 `fit()`。

### 6.1 (Train, Val) 資料構建

- 訓練集大小：大約 64% (約 488 筆，以實際執行結果為準)。
- 驗證集大小：大約 16% (約 123 筆)。
- 測試集大小：大約 20% (約 153 筆)。

### 6.2 LSTM 訓練 (Pure LSTM)

1. **資料**：直接使用 `(train_df['adj_close'])` 與 `(val_df['adj_close'])`。
2. **滑動窗口**：20 → 產生 `(N - 20)` 筆輸入的 `(20,1)`。
3. **訓練**：
    - `epochs=300`
    - `batch_size=128`
    - `validation_data=(X_val, y_val)`
    - `EarlyStopping(patience=30)`
4. **損失曲線**：顯示 `loss` 與 `val_loss`，若 `val_loss` 在 30 個 epoch 內未改善則提前停止。
5. **訓練結果**：在最終 epoch（或 early stop）後，取最佳權重作為最終模型。

### 6.2.1 Loss 曲線

(如下範例示意)

```
Epoch 1/300
...
Epoch X/300
Restoring model weights from the end of the best epoch
Epoch X: early stopping

```

顯示 `Train Loss` 與 `Val Loss` 逐 Epoch 變化，若一開始約在 X.XXX，隨著 epoch 進行下降到相對平穩後停止。

### 6.3 ARIMA / Auto ARIMA 殘差 LSTM (Hybrid)

1. **在 (train+val) Fit ARIMA / Auto ARIMA**
    - 以 `ARIMA(1,1,1)`：`model = ARIMA(train_val_series, order=(1,1,1)).fit()`
    - 或 `Auto ARIMA`：`model = auto_arima(train_val_series, ...)`
    - 得到 in-sample 預測值 `arima_pred`。
2. **計算殘差**：`residual = actual - arima_pred`。
3. **殘差序列** → 用上述同樣的滑動窗口 (20) 方法來訓練一個 LSTM。
4. **EarlyStopping**：一樣使用 `patience=30`。
5. **Training loss curve**：同 Pure LSTM，但輸入資料是 `residual`。

---

## 7. 測試 (滾動預測) 與結果

### 7.1 測試步驟

以下以「Pure ARIMA(1,1,1)」為例，其他模型類似：

1. **測試集大小**：161 筆 (實際可能不同)。
2. 每個測試日 `i` (從 `i=0` 到 `i=160`)：
    1. 預測下一筆的股價。
    2. 更新滑動窗口
    3. 重複執行直到預測結束。

### 7.2 評估指標

對於 **預測序列** (長度約 153) 與 **真實序列** (同長度)，計算：

1. **RMSE** = 1n∑i=1n(yi−y^i)2\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
2. **MAPE**(%) = 1n∑i=1n∣yi−y^iyi∣×100%\frac{1}{n}\sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|\times100\%
3. **方向準確率 (ACC)**：計算 `(y_{i} - y_{i-1})` 與 `(\hat{y}_{i} - \hat{y}_{i-1})` 之符號是否相同，以此判斷漲跌方向預測正確與否。
4. **R^2**：`r2_score(y_true, y_pred)`。

---

## 8. 訓練過程與程式碼參考

1. [**主程式碼**](https://chatgpt.com/c/67f58d07-96f4-8011-a99e-f5992bc570e1#)
    - 包含資料前處理、各模型訓練與預測函式模組、滾動式預測實作、最終評估指標計算以及圖表繪製。
2. **依實際需求可拆分以下模組**：
    - `load_and_preprocess_data(csv_path, cutoff_date)`：讀取與清理資料。
    - `split_data(df, train_val_ratio, train_ratio)`：切割資料。
    - **ARIMA**
        - `train_arima_111(series)`, `train_auto_arima(series)`：用於 (train+val)。
        - `rolling_forecast_arima(full_df, model_type)`：逐日滾動預測。
    - **Naive**
        - `naive_forecast(test_df, full_df)`：逐日滾動預測。
    - **Pure LSTM**
        - `create_lstm_model()`：建立雙層 LSTM + Dense 架構。
        - `build_lstm_training_data(series, window)`：滑動窗口產生 X, y。
        - `train_lstm_for_pure(train_df, val_df, window, epochs, batch_size)`：訓練 LSTM (純股價序列)。
        - `rolling_forecast_lstm_pure(full_df, lstm_model, window)`：逐日滾動預測。
    - **Hybrid ARIMA + LSTM**
        - `prepare_residual_data_arima(full_df, model_type)`：以 (train+val) in-sample 預測殘差。
        - `train_lstm_for_residual(train_df, val_df, residuals_col, window, epochs, batch_size)`：訓練殘差 LSTM。
        - `rolling_forecast_hybrid_arima_lstm(full_df, arima_model, lstm_model, model_type, window)`：逐日滾動 (ARIMA預測 + LSTM殘差)。
    - **評估函式**
        - `mean_absolute_percentage_error(y_true, y_pred)`
        - `direction_accuracy(y_true, y_pred)`
        - `evaluate_performance(model_name, preds, actuals)`
3. **注意版本**：
    - `tensorflow` 2.x
    - `pmdarima` ≥ 1.8
    - `statsmodels` ≥ 0.12
    - 若版本過舊，部分函式參數可能略有差異。

---

## 9. 視覺化

1. **Loss曲線**：
    - 於 LSTM 訓練完成後，繪製 `history.history['loss']` 與 `history.history['val_loss']`，以確認收斂情況。
2. **測試集預測 vs. 實際（走勢圖滾動式預測與整段軌跡預測皆要產生）**：
    - 可繪製測試集中預測值與實際值的走勢。
    - 若要多模型同圖比較，注意曲線易重疊，可拆分多張圖或使用不同折線樣式。
3. 產生多個模型在各個指標上的表現的報表。

範例：

```python
plt.figure(figsize=(10,6))
plt.plot(actuals, label='Actual')
plt.plot(preds, label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('Index in Test Set')
plt.ylabel('Price')
plt.legend()
plt.show()

```

---

## 10. 結論

1. **Hybrid 模型** 大多能改善預測精度 (RMSE, MAPE) 與方向準確率 (ACC)，原因在於它結合了 ARIMA 擅長捕捉整體趨勢與 LSTM 擅長捕捉非線性殘差的優點。
2. **Pure LSTM** 在一定程度上也能有不錯的預測效果，特別是當資料量充足、LSTM 能夠捕捉長期時序特徵。
3. **Naive Forecast** 雖然簡易，但對於非常短期的預測有時能提供一個 baseline；本實驗中它通常表現較差。
4. 具體最適參數 (ARIMA p, d, q, LSTM 層數/神經元數目等) 仍可透過更多調參來進一步提升表現。

---

## 11. 未來工作

1. **增量/局部更新**：
    - 減少 ARIMA 在測試階段天天 re-fit 所需的耗時，可考慮 `append()` 或 partial fit 的方式。
2. **調整 LSTM 架構**：
    - 若想要更深層或不同學習率，可再嘗試多層 CNN-LSTM、Bi-LSTM 等進階方法。
3. **特徵擴增**：
    - 本實驗只用 `adj_close`，實務中可加入技術指標(成交量、移動平均、MACD、RSI 等)或其他公司財報、新聞等數據。
4. **加入時間特性**：
    - 可考慮週期性 (如星期幾、月份特徵) 等加以建模。

---

## 12. 參考資源

1. **pmdarima**：[https://github.com/alkaline-ml/pmdarima](https://github.com/alkaline-ml/pmdarima)
2. **statsmodels ARIMA**：[https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
3. **Keras LSTM**：[https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)

---