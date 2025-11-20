import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from snownlp import SnowNLP
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import plotly.graph_objs as go
import os
import time

# 忽略 TensorFlow 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(page_title="台股 AI 預測 (時效加權版)", layout="wide")

# === 🔐 密碼保護 (已修正 Local 端錯誤) ===
def check_password():
    """回傳 True 代表登入成功"""
    if st.session_state.get("password_correct", False):
        return True

    st.header("🔒 請輸入存取密碼")
    st.markdown("內部測試系統，請輸入密碼。")
    password_input = st.text_input("Password", type="password")
    
    if st.button("登入"):
        # --- 防呆機制：自動判斷是本機還是雲端 ---
        try:
            true_password = st.secrets["PASSWORD"] # 嘗試讀取雲端密碼
        except FileNotFoundError:
            true_password = "1234" # 本機測試預設密碼
        # ---------------------------------------

        if password_input == true_password:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("❌ 密碼錯誤")
    return False

if not check_password():
    st.stop()
# ===========================================

st.title("🇹🇼 台股 AI 預測 (時效加權 + 深度分析)")
st.markdown("演算法升級：加入 **時效遞減權重 (Time Decay)**，越接近當下的新聞，對預測分數的影響越大。")

# --- 側邊欄 ---
st.sidebar.header("設定參數")
stock_map = {
    "2330 台積電": "2330.TW",
    "2317 鴻海": "2317.TW",
    "2454 聯發科": "2454.TW",
    "2603 長榮": "2603.TW",
    "3231 緯創": "3231.TW",
    "2382 廣達": "2382.TW",
    "3008 大立光": "3008.TW",
    "自訂輸入": "CUSTOM"
}
selected_label = st.sidebar.selectbox("選擇股票", list(stock_map.keys()))

if selected_label == "自訂輸入":
    stock_ticker = st.sidebar.text_input("請輸入台股代碼 (需加 .TW)", "2330.TW")
    stock_id = stock_ticker.split(".")[0]
    stock_name_for_ptt = stock_id 
else:
    stock_ticker = stock_map[selected_label]
    stock_id = stock_ticker.split(".")[0]
    stock_name_for_ptt = selected_label.split(" ")[1]

look_back = st.sidebar.slider("參考過去幾天", 10, 60, 30)
epochs = st.sidebar.slider("訓練次數", 1, 30, 10)
ensemble_runs = st.sidebar.slider("預測平均次數", 1, 5, 3)

# --- 核心：金融情緒校正 ---
def analyze_financial_sentiment(text):
    s = SnowNLP(text)
    score = s.sentiments
    
    negative_keywords = [
        "暴跌", "重挫", "崩盤", "縮水", "虧損", "新低", "疲弱", "利空", 
        "賣壓", "下修", "不如預期", "衰退", "侵蝕", "保守", "疑慮", "過剩",
        "砍單", "裁員", "違約", "降評", "失望", "重摔"
    ]
    positive_keywords = [
        "暴漲", "大漲", "飆股", "新高", "獲利", "利多", "優於預期", 
        "成長", "噴出", "滿載", "加碼", "看好", "強勁", "回升", "股息",
        "樂觀", "爆發", "完銷"
    ]
    
    adjustment = 0
    for word in negative_keywords:
        if word in text: adjustment -= 0.15
    for word in positive_keywords:
        if word in text: adjustment += 0.1
            
    return max(0.01, min(0.99, score + adjustment))

# --- 核心：加權平均計算 (新功能) ---
def calculate_weighted_score(scores):
    """
    輸入分數列表 [最新, 次新, 舊, ...]
    給予遞減權重，越新的權重越重
    """
    if not scores:
        return 0.5
    
    # 權重係數：0.9 (代表每多舊一篇，重要性剩 90%)
    decay_factor = 0.9
    weights = [decay_factor ** i for i in range(len(scores))]
    
    # 計算加權平均
    weighted_average = np.average(scores, weights=weights)
    return weighted_average

# --- 爬蟲函式 (整合權重計算) ---

def get_yahoo_news_sentiment(stock_id):
    url = f"https://tw.stock.yahoo.com/quote/{stock_id}.TW/news"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = soup.find_all('h3')
        scores, titles_list, seen_titles = [], [], set()
        
        count = 0
        # Yahoo 預設就是按時間排序 (最新在最上面)
        for h in headlines:
            if count >= 6: break
            title = h.get_text().strip()
            if len(title) < 5 or title in seen_titles: continue
            seen_titles.add(title)
            
            final_score = analyze_financial_sentiment(title)
            scores.append(final_score)
            
            # 顯示時加入權重標示，讓使用者知道這篇多重要
            # 權重計算僅供顯示: 0.9^0=100%, 0.9^1=90%...
            weight_display = (0.9 ** count) * 100
            titles_list.append(f"[Yahoo] (權重{weight_display:.0f}%) {final_score:.2f} - {title}")
            
            count += 1
            
        # 使用加權平均
        avg_score = calculate_weighted_score(scores) if scores else 0.5
        return avg_score, titles_list
        
    except Exception as e:
        return 0.5, [f"Yahoo 錯誤: {e}"]

def get_ptt_sentiment(keyword):
    url = f"https://www.ptt.cc/bbs/Stock/search?q={keyword}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    cookies = {'over18': '1'}
    try:
        response = requests.get(url, headers=headers, cookies=cookies, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        titles_tags = soup.find_all('div', class_='title')
        
        scores, titles_list, seen_titles = [], [], set()
        count = 0
        
        for t in titles_tags:
            if count >= 3: break
            a_tag = t.find('a')
            if a_tag:
                title = a_tag.get_text().strip()
                href = a_tag['href']
                if "已被刪除" in title or title in seen_titles: continue
                seen_titles.add(title)
                
                try:
                    post_url = "https://www.ptt.cc" + href
                    post_resp = requests.get(post_url, headers=headers, cookies=cookies, timeout=5)
                    post_soup = BeautifulSoup(post_resp.text, 'html.parser')
                    main_content = post_soup.find(id="main-content")
                    if main_content:
                        for meta in main_content.find_all('div', class_='article-metaline'):
                            meta.decompose()
                        raw_text = main_content.get_text().strip()[:500]
                        full_text = title + " " + raw_text
                    else:
                        full_text = title
                except:
                    full_text = title
                
                final_score = analyze_financial_sentiment(full_text)
                scores.append(final_score)
                
                weight_display = (0.9 ** count) * 100
                titles_list.append(f"[PTT] (權重{weight_display:.0f}%) {final_score:.2f} - {title}")
                
                count += 1
                time.sleep(0.5)
        
        avg_score = calculate_weighted_score(scores) if scores else 0.5
        return avg_score, titles_list
        
    except Exception as e:
        return 0.5, [f"PTT 錯誤: {e}"]

# --- 資料處理與模型 ---
def preprocess_data(df, look_back):
    dataset = df['Close'].values.reshape(-1, 1)
    np.random.seed(42)
    sentiment_history = np.random.uniform(0, 1, size=(len(dataset), 1))
    combined_data = np.hstack((dataset, sentiment_history))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(combined_data)
    return scaled_data, scaler, dataset

def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- 主程式 ---
st.subheader(f"📊 分析標的：{stock_ticker}")

with st.spinner('正在下載所有歷史資料...'):
    df = yf.download(stock_ticker, period="max")

if df is not None and not df.empty:
    if isinstance(df.columns, pd.MultiIndex):
        try: df = df.xs('Close', axis=1, level=0, drop_level=True)
        except: df = df.iloc[:, 3].to_frame()
    if isinstance(df, pd.Series): df = df.to_frame()
    df.columns = ['Close']
    df = df.dropna()

    # 1. 計算時間範圍
    # 我們預設讓圖表顯示「最近 1 年」，這樣才不會一打開就看到 30 年擠在一起
    last_date = df.index[-1]
    first_date = last_date - pd.Timedelta(days=365)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='收盤價'))
    
    fig.update_layout(
        title=f"{stock_ticker} 歷史股價",
        xaxis_title="日期",
        yaxis_title="股價",
        dragmode='pan', # 手機與滑鼠皆為平移模式
        height=500,
        
        # 2. 關鍵修正：強制設定 X 軸範圍 (Range)
        xaxis=dict(
            range=[first_date, last_date], # <--- 這裡鎖定：起點是1年前，終點是最新資料
            rangeslider=dict(visible=True), # 下方保留時間拉桿
            type="date"
        )
    )
    
    # 3. 啟用滾輪縮放 (Scroll Zoom)
    st.plotly_chart(
        fig, 
        use_container_width=True, 
        config={
            'scrollZoom': True,       # 啟用滾輪/雙指縮放
            'displayModeBar': True,   # 顯示工具列
            'displaylogo': False      # 隱藏 logo
        }
    )
    
    training_limit = 1250 
    df_for_training = df.iloc[-training_limit:] if len(df) > training_limit else df
        
    scaled_data, scaler, raw_data = preprocess_data(df_for_training, look_back)
    
    train_size = int(len(scaled_data) * 0.9)
    train_data = scaled_data[0:train_size, :]
    
    x_train, y_train = [], []
    for i in range(look_back, len(train_data)):
        x_train.append(train_data[i-look_back:i, :])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    if st.button(f'🚀 啟動 AI 深度分析 (含時效加權)'):
        st.write("---")
        st.info("正在進行深度輿情分析...")
        yahoo_score, yahoo_titles = get_yahoo_news_sentiment(stock_id)
        ptt_score, ptt_titles = get_ptt_sentiment(stock_name_for_ptt)
        
        if "無結果" in ptt_titles[0]: ptt_score, ptt_titles = get_ptt_sentiment(stock_id)
        
        final_sentiment = (yahoo_score + ptt_score) / 2
        
        col1, col2 = st.columns(2)
        col1.metric("綜合情緒分數 (加權後)", f"{final_sentiment:.2f}")
        with col2.expander("查看新聞權重與分數"):
            st.markdown("權重說明：排序越前 (越新) 的新聞，對分數影響越大。")
            for t in yahoo_titles + ptt_titles: st.write(t)
            
        st.write("---")
        st.subheader(f"🧠 正在訓練 {ensemble_runs} 個 AI 模型...")
        progress_bar = st.progress(0)
        prediction_list = []
        
        last_days = scaled_data[-look_back:].copy()
        last_days[-1, 1] = final_sentiment
        X_input = last_days.reshape(1, look_back, 2)
        
        for i in range(ensemble_runs):
            model = build_model((x_train.shape[1], x_train.shape[2]))
            model.fit(x_train, y_train, batch_size=16, epochs=epochs, verbose=0)
            pred_scaled = model.predict(X_input, verbose=0)
            temp = np.zeros((1, 2))
            temp[0, 0] = pred_scaled[0, 0]
            pred_price = scaler.inverse_transform(temp)[0][0]
            prediction_list.append(pred_price)
            progress_bar.progress((i + 1) / ensemble_runs)
            
        avg_price = np.mean(prediction_list)
        last_close = raw_data[-1][0]
        
        st.subheader("🔮 預測結果")
        r_col1, r_col2 = st.columns(2)
        r_col1.metric("昨日收盤價", f"{last_close:.2f}")
        r_col2.metric("AI 平均預測價", f"{avg_price:.2f}", delta=f"{avg_price - last_close:.2f}")
        st.write(f"個別模型預測值： {[round(p, 1) for p in prediction_list]}")

else:
    st.error("無法取得資料。")