import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import unicodedata
import re
import math
import os
import requests

# -------------------------
# 1. データベース読み込み (CSV外部ファイル化)
# -------------------------
@st.cache_data
def load_database():
    db_list = []
    file_path = 'database.csv'
    if os.path.exists(file_path):
        try:
            try:
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='cp932')
                
            for _, row in df.iterrows():
                db_list.append({
                    '代表名': str(row['代表名']).strip(),
                    '別名リスト': [alias.strip() for alias in str(row['別名リスト']).split(',')],
                    'MW': float(row['MW']),
                    'EC3': float(row['EC3']),
                    'PII': float(row['PII']),
                    '備考': str(row['備考'])
                })
        except Exception as e:
            st.error(f"データベースの読み込み中にエラーが発生しました: {e}")
    return db_list

db_data = load_database()

def normalize_text(text):
    return unicodedata.normalize('NFKC', text).strip().upper()

# -------------------------
# 2. リスク計算＆リスク指数算出関数
# -------------------------
def norm_mw(mw): 
    if mw >= 500:
        return 0.0
    score = 10.0 / (1.0 + math.exp(0.022 * (mw - 260)))
    return max(0.0, min(10.0, round(score, 1)))

def norm_ec3(ec3): 
    return max(0.0, min(10.0, round((35 - ec3) / 3.5, 1)))

def norm_pii(pii): 
    return max(0.0, min(10.0, round(pii * (10/8.0), 1)))

def calc_risk_index(mw, ec3, pii):
    s1 = norm_mw(mw)
    s2 = norm_ec3(ec3)
    s3 = norm_pii(pii)
    risk_idx = ((s1 * s2 + s2 * s3 + s3 * s1) / 300.0) * 100.0
    return round(risk_idx, 1), (s1, s2, s3)

def get_risk_color_and_label(risk_idx):
    if risk_idx < 15.0:
        return "#2ca02c", "低リスク（比較的安全）"
    elif risk_idx < 40.0:
        return "#ff7f0e", "中リスク（要注意）"
    else:
        return "#d62728", "高リスク（厳重警戒）"

def render_risk_bar(risk_idx):
    color, label = get_risk_color_and_label(risk_idx)
    bar_html = f"""
    <div style="margin-top: 10px; margin-bottom: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 10px; border: 1px solid #e0e0e0;">
        <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 10px;">
            <span style="font-size: 20px; font-weight: bold; color: #333;">🔥 製品の総合リスク指数: <span style="color: {color};">{label}</span></span>
            <span style="font-size: 28px; font-weight: bold; color: {color};">{risk_idx} <span style="font-size: 16px;">%</span></span>
        </div>
        <div style="width: 100%; background-color: #e6e6e6; border-radius: 10px; height: 30px; overflow: hidden;">
            <div style="width: {min(risk_idx, 100.0)}%; background-color: {color}; height: 100%; transition: width 0.5s ease-in-out;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 12px; color: #666;">
            <span>0% (安全)</span>
            <span>40% (警告ライン)</span>
            <span>100% (極大リスク)</span>
        </div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

def create_radar_chart(mw, ec3, pii, title="", figsize=(5, 5), font_size=9):
    risk_idx, (s1, s2, s3) = calc_risk_index(mw, ec3, pii)
    risk_scores = [s1, s2, s3]
    labels = ['浸透リスク\n(小MW)', 'アレルギー\n(低EC3)', '一次刺激\n(高PII)']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    
    risk_scores += risk_scores[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], labels, fontsize=font_size, fontweight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "MAX"], color="grey", size=font_size-2)
    plt.ylim(0, 10)
    
    color, _ = get_risk_color_and_label(risk_idx)
        
    ax.plot(angles, risk_scores, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, risk_scores, color=color, alpha=0.4)
    
    if title:
        plt.title(title, size=font_size+2, fontweight='bold', pad=15)
        
    plt.tight_layout()
    return fig, risk_idx, (s1, s2, s3)

# -------------------------
# 3. UIとメインロジック
# -------------------------
st.set_page_config(page_title="ジェルネイル成分リスク評価シミュレーター", layout="wide")

if not db_data:
    st.error("🚨 データベースファイル (`database.csv`) が見つかりません。")
    st.stop()

if 'input_text' not in st.session_state:
    st.session_state['input_text'] = "ウレタンアクリレート, メタクリル酸2-ヒドロキシエチル, 酸化チタン, 赤色202号, TPO, 酢酸エチル"

st.sidebar.title("📚 成分データベース情報")
st.sidebar.info(f"現在 **{len(db_data)}種類** の成分がローカル登録されています。")

search_filter = st.sidebar.text_input("データベース内をキーワード検索", placeholder="例: メタクリル酸, 酢酸, 酸化チタン")
with st.sidebar.expander("登録済み全成分リストを見る", expanded=True):
    for item in db_data:
        if not search_filter or normalize_text(search_filter) in normalize_text(item['代表名']) or any(normalize_text(search_filter) in normalize_text(a) for a in item['別名リスト']):
            st.caption(f"・**{item['代表名']}** ({item['別名リスト'][0]})")

st.title("💅 ジェルネイル成分 リスク評価シミュレーター")
st.markdown("ローカルデータベースを参照し、ジェルネイル成分の毒性リスクを算出します。未登録の成分は自動的に管理者に報告され、順次データベースへ追加されます。")

input_text = st.text_area("▼ ここに成分表をカンマ（,）または読点（、）区切りで入力", 
                          value=st.session_state['input_text'], height=100)

if st.button("リスクを解析する"):
    st.session_state['input_text'] = input_text
    
    split_pattern = r'[,、]\s*(?![^\(]*\))'
    components = [c.strip() for c in re.split(split_pattern, input_text) if c.strip()]
    
    if not components:
        st.warning("成分が入力されていません。")
        st.stop()

    matched_data = []
    unmatched_names = []
    all_analyzed_names = []
    
    for comp in components:
        norm_comp = normalize_text(comp)
        found_item = None
        for item in db_data:
            normalized_aliases = [normalize_text(alias) for alias in item['別名リスト']]
            if norm_comp in normalized_aliases:
                found_item = item
                break
        
        if found_item:
            all_analyzed_names.append(found_item['代表名'])
            if not any(d['表示用名称'] == found_item['代表名'] for d in matched_data):
                matched_data.append({
                    '成分名': found_item['代表名'],
                    '表示用名称': found_item['代表名'],
                    'MW': found_item['MW'],
                    'EC3': found_item['EC3'],
                    'PII': found_item['PII'],
                    '備考': found_item.get('備考', '')
                })
        else:
            unmatched_names.append(comp)
            all_analyzed_names.append(f"【未】{comp}")

    # スプレッドシートへの送信ロジック
    def log_to_spreadsheet(component_list):
        if not component_list:
            return
        current_list_str = ",".join(component_list)
        if st.session_state.get('last_logged_list') == current_list_str:
            return
        url = "https://docs.google.com/forms/d/e/1FAIpQLSdiSHSWGtG2th3k5uvU5lOujfDGjLqLXIUHIt-olNK4RxCJtg/formResponse"
        form_data = {"entry.796504215": current_list_str}
        try:
            requests.post(url, data=form_data)
            st.session_state['last_logged_list'] = current_list_str
        except Exception:
            pass

    log_to_spreadsheet(all_analyzed_names)

    if unmatched_names:
        st.error(f"🚨 **データベース未登録エラー**: 以下の {len(unmatched_names)} 成分がデータベースに見つかりませんでした。")
        st.code("\n".join(unmatched_names), language="text")
        st.caption("※管理者のダッシュボードには報告済みです。今後のデータベース更新で順次追加されます。")

    # -------------------------
    # 4. 評価とチャート描画
    # -------------------------
    if matched_data:
        df_matched = pd.DataFrame(matched_data)
        
        st.divider()
        st.subheader("📊 1. 製品全体の総合評価（ワーストケース・ボトルネック）")
        
        base_components = df_matched.head(5)
        worst_mw = base_components['MW'].min()
        worst_mw_name = base_components.loc[base_components['MW'].idxmin()]['表示用名称']
        
        worst_ec3 = df_matched['EC3'].min()
        worst_ec3_name = df_matched.loc[df_matched['EC3'].idxmin()]['表示用名称']
        
        worst_pii = df_matched['PII'].max()
        worst_pii_name = df_matched.loc[df_matched['PII'].idxmax()]['表示用名称']
        
        total_risk_idx, (ts1, ts2, ts3) = calc_risk_index(worst_mw, worst_ec3, worst_pii)
        
        # 巨大なリスクプログレスバーを描画（一番目立つ位置に配置）
        render_risk_bar(total_risk_idx)
        
        # ワースト成分のサマリー（3列でスッキリ配置）
        col1, col2, col3 = st.columns(3)
        col1.metric("① 浸透リスク (最小MW)", f"{worst_mw} Da", f"原因: {worst_mw_name}", delta_color="inverse")
        col2.metric("② アレルギーリスク (最低EC3)", f"{worst_ec3} %", f"原因: {worst_ec3_name}", delta_color="inverse")
        col3.metric("③ 刺激リスク (最高PII)", f"{worst_pii}", f"原因: {worst_pii_name}", delta_color="inverse")
        
        if total_risk_idx < 25.0 and max(ts1, ts2, ts3) >= 8.0:
            st.warning("⚠️ **単一リスク突出警告**: 総合リスク指数は低めですが、特定のリスク項目が極めて高い水準にあります。皮膚への付着に注意が必要です。")
        
        # レーダーチャートの描画
        fig_total, _, _ = create_radar_chart(worst_mw, worst_ec3, worst_pii, title=f"全体のリスクバランス", figsize=(5.5, 5.5), font_size=10)
        
        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            st.pyplot(fig_total)
            plt.close(fig_total)

        st.divider()
        st.subheader(f"🧪 2. 配合成分ごとの個別リスクプロファイル一覧（全{len(matched_data)}成分）")
        
        cols_per_row = 3
        for idx in range(0, len(matched_data), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if idx + j < len(matched_data):
                    item = matched_data[idx + j]
                    with cols[j]:
                        with st.container(border=True):
                            fig_indiv, indiv_risk_idx, (is1, is2, is3) = create_radar_chart(
                                item['MW'], item['EC3'], item['PII'], figsize=(4, 4), font_size=8
                            )
                            st.markdown(f"**{idx + j + 1}. {item['表示用名称']}**")
                            st.markdown(f"リスク指数: **`{indiv_risk_idx}%`**")
                            
                            if indiv_risk_idx < 25.0 and max(is1, is2, is3) >= 8.0:
                                st.caption("🚨 **単一リスク突出型（要注意）**")
                                
                            st.pyplot(fig_indiv)
                            plt.close(fig_indiv)
                            st.caption(f"MW: **{item['MW']} Da** | EC3: **{item['EC3']}%** | PII: **{item['PII']}**")
                            st.info(f"💡 {item['備考']}")
