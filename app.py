import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import unicodedata
import re
import math
import os
import requests
import html

# -------------------------
# 0. ページ設定とハイエンド向けCSSデザイン
# -------------------------
st.set_page_config(page_title="ジェルネイル成分リスク評価シミュレーター", layout="wide")

st.markdown("""
<style>
    /* 全体の背景色とフォント */
    .stApp {
        background-color: #F8F9FA;
        font-family: 'Helvetica Neue', 'Yu Gothic', 'YuGothic', 'Hiragino Kaku Gothic ProN', sans-serif;
    }
    
    /* タイトルの洗練化 */
    h1 {
        color: #111111;
        font-weight: 300;
        letter-spacing: 1.5px;
        padding-bottom: 5px;
        font-size: 28px !important;
    }
    
    /* 解析ボタン */
    div[data-testid="column"]:nth-of-type(1) .stButton > button {
        background: linear-gradient(to right, #232526, #414345);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 12px 30px;
        font-size: 15px;
        font-weight: 500;
        letter-spacing: 1px;
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        width: 100%;
    }
    div[data-testid="column"]:nth-of-type(1) .stButton > button:hover {
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }

    /* クリアボタン */
    div[data-testid="column"]:nth-of-type(2) .stButton > button {
        background: transparent;
        color: #414345;
        border: 1px solid #C0C0C0;
        border-radius: 30px;
        padding: 12px 30px;
        font-size: 15px;
        font-weight: 500;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        width: 100%;
    }
    div[data-testid="column"]:nth-of-type(2) .stButton > button:hover {
        border-color: #414345;
        background: #F0F0F0;
    }
    
    /* テキスト入力エリア（製品名・成分表） */
    .stTextInput input, .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid #E0E0E0;
        background-color: #FFFFFF;
        padding: 12px 15px;
        font-size: 14px;
        line-height: 1.6;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.01);
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #414345;
        box-shadow: 0 0 0 1px #414345;
    }
    
    /* メトリクス */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.03);
        border: 1px solid rgba(0,0,0,0.02);
        border-left: none;
        text-align: center;
    }
    div[data-testid="stMetricLabel"] {
        justify-content: center;
        color: #777777;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* 個別成分のカードUI化 */
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
        border: 1px solid #F0F0F0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.03);
        background-color: #FFFFFF;
        padding: 10px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.06);
    }

    /* =========================================================
       ★ PDF出力用（印刷時）の最適化CSS（強制改ページ設計）
       ========================================================= */
    @media print {
        @page {
            size: A4 portrait;
            margin: 10mm;
        }
        
        /* 不要UI（入力欄含む）を隠す */
        [data-testid="stSidebar"], 
        header, 
        .stTextInput,
        .stTextArea, 
        .stButton, 
        iframe,
        hr {
            display: none !important;
        }
        
        /* カード6枚（2段）をA4に美しく収めるための倍率調整 */
        .block-container {
            zoom: 0.72 !important; 
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            max-width: 100% !important;
        }
        
        .stApp, .main {
            background-color: #FFFFFF !important;
        }
        
        div[data-testid="stVerticalBlockBorderWrapper"] {
            box-shadow: none !important;
            border: 1px solid #DDDDDD !important;
            margin-bottom: 10px !important;
        }
        
        /* グラフ画像の分離を防ぐ */
        img {
            max-width: 100% !important;
            height: auto !important;
            display: block !important;
            margin: 0 auto !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# -------------------------
# 1. データベース読み込み
# -------------------------
@st.cache_data(ttl=60)
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
                rep_name = str(row['代表名']).strip()
                # カッコ内のカンマでは分割しない
                raw_aliases = [alias.strip() for alias in re.split(r',\s*(?![^\(]*\))', str(row['別名リスト'])) if alias.strip()]
                search_aliases = list(dict.fromkeys([rep_name] + raw_aliases))
                
                # 区分列の読み込み（未記入や列未追加時は '通常' として扱う）
                category = '通常'
                if '区分' in df.columns and pd.notna(row['区分']) and str(row['区分']).strip():
                    category = str(row['区分']).strip()
                
                db_list.append({
                    '代表名': rep_name,
                    '別名リスト': search_aliases,
                    '表示用別名': raw_aliases[0] if raw_aliases else rep_name,
                    'MW': float(row['MW']),
                    'EC3': float(row['EC3']),
                    'PII': float(row['PII']),
                    '備考': str(row['備考']) if pd.notna(row['備考']) else '',
                    '区分': category
                })
        except Exception as e:
            st.error(f"データベースの読み込み中にエラーが発生しました: {e}")
    return db_list

db_data = load_database()

def normalize_text(text):
    return unicodedata.normalize('NFKC', text).strip().upper()


# -------------------------
# 2. リスク計算関数
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
        return "#2ecc71", "SAFE / マイルド設計"
    elif risk_idx < 40.0:
        return "#f39c12", "CAUTION / やや注意"
    else:
        return "#e74c3c", "DANGER / 高リスク警戒"

def render_risk_bar(risk_idx):
    color, label = get_risk_color_and_label(risk_idx)
    bar_html = f"""
    <div style="margin: 10px 0 30px 0; padding: 30px 35px; background: #ffffff; border-radius: 20px; box-shadow: 0 10px 40px rgba(0,0,0,0.04); border: 1px solid #F5F5F5;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span style="font-size: 14px; font-weight: 600; color: #888; letter-spacing: 2px;">TOTAL RISK SCORE</span>
            <span style="font-size: 38px; font-weight: 300; color: {color}; line-height: 1;">{risk_idx}<span style="font-size: 20px; font-weight: 400; color: #BBBBBB; margin-left: 6px;">%</span></span>
        </div>
        <div style="width: 100%; background-color: #F0F2F5; border-radius: 20px; height: 10px; overflow: hidden; position: relative;">
            <div style="width: {min(risk_idx, 100.0)}%; background-color: {color}; height: 100%; border-radius: 20px; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 15px; font-size: 11px; font-weight: 500; letter-spacing: 1px;">
            <span style="color: #AAAAAA;">0% (安全)</span>
            <span style="color: {color}; font-weight: 700; letter-spacing: 1.5px;">{label}</span>
            <span style="color: #AAAAAA;">100% (極大リスク)</span>
        </div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

def create_radar_chart(mw, ec3, pii, title="", figsize=(4.2, 4.2), font_size=9):
    risk_idx, (s1, s2, s3) = calc_risk_index(mw, ec3, pii)
    risk_scores = [s1, s2, s3]
    labels = ['浸透リスク\n(小MW)', 'アレルギー\n(低EC3)', '一次刺激\n(高PII)']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    
    risk_scores += risk_scores[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], labels, fontsize=font_size, fontweight='400', color='#555555')
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "MAX"], color="#CCCCCC", size=font_size-2)
    plt.ylim(0, 10)
    
    color, _ = get_risk_color_and_label(risk_idx)
        
    ax.plot(angles, risk_scores, color=color, linewidth=1.5, linestyle='solid')
    ax.fill(angles, risk_scores, color=color, alpha=0.2)
    
    ax.spines['polar'].set_color('#E0E0E0')
    ax.grid(color='#EAEAEA')
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    if title:
        plt.title(title, size=font_size+2, fontweight='300', pad=15, color='#333333')
        
    plt.tight_layout()
    return fig, risk_idx, (s1, s2, s3)

# ★ PDF出力用ボタンのコンポーネント定義
def render_pdf_export_button():
    components.html(
        """
        <style>
        .print-btn {
            background: linear-gradient(to right, #232526, #414345);
            color: white;
            border: none;
            border-radius: 30px;
            padding: 15px 30px;
            font-size: 16px;
            font-weight: 500;
            letter-spacing: 2px;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s ease;
            font-family: 'Helvetica Neue', sans-serif;
            box-shadow: 0 8px 15px rgba(0,0,0,0.1);
            margin-top: 10px;
        }
        .print-btn:hover {
            box-shadow: 0 10px 20px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }
        </style>
        <button class="print-btn" onclick="window.parent.print()">🖨️ REPORT PDF EXPORT / レポートを出力</button>
        """,
        height=80
    )


# -------------------------
# 3. UIとメインロジック
# -------------------------
if not db_data:
    st.error("🚨 データベースファイル (`database.csv`) が見つかりません。")
    st.stop()

def clear_text():
    st.session_state['product_name'] = ""
    st.session_state['input_area'] = ""

if 'product_name' not in st.session_state:
    st.session_state['product_name'] = ""

if 'input_area' not in st.session_state:
    st.session_state['input_area'] = "ウレタンアクリレート, メタクリル酸2-ヒドロキシエチル, 酸化チタン, 赤色202号, TPO, 酢酸エチル"

st.sidebar.title("📚 データベース情報")
st.sidebar.info(f"現在 **{len(db_data)}種類** の成分が登録されています。")

search_filter = st.sidebar.text_input("データベース内を検索", placeholder="例: メタクリル酸")
with st.sidebar.expander("登録済み全成分リスト", expanded=True):
    for item in db_data:
        if not search_filter or normalize_text(search_filter) in normalize_text(item['代表名']) or any(normalize_text(search_filter) in normalize_text(a) for a in item['別名リスト']):
            cat_tag = " `[開始剤]`" if item['区分'] == '開始剤' else ""
            st.caption(f"・**{item['代表名']}** ({item['表示用別名']}){cat_tag}")

st.markdown("<h1>成分から導き出す、次世代の安全性評価プロファイル</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#777; font-size: 15px; font-weight: 500; margin-bottom: 20px; letter-spacing: 1.5px;'>NAIL GEL INGREDIENT SIMULATOR</p>", unsafe_allow_html=True)

product_name = st.text_input("製品名（任意・レポートおよびログに反映されます）", key="product_name", placeholder="例: KLOS ベースジェル 01")

input_text = st.text_area("成分表（カンマまたは読点区切り）を入力してください", 
                          key="input_area", height=120)

col1, col2 = st.columns([4, 1])
with col1:
    analyze_submitted = st.button("ANALYSIS / 解析を実行")
with col2:
    st.button("CLEAR", on_click=clear_text)

if analyze_submitted:
    split_pattern = r'[,、]\s*(?![^\(]*\))'
    parsed_components = [c.strip() for c in re.split(split_pattern, input_text) if c.strip()]
    
    if not parsed_components:
        st.warning("成分が入力されていません。")
        st.stop()

    matched_data = []
    unmatched_names = []
    all_analyzed_names = []
    
    for comp in parsed_components:
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
                    '備考': found_item.get('備考', ''),
                    '区分': found_item.get('区分', '通常')
                })
        else:
            unmatched_names.append(comp)
            all_analyzed_names.append(f"【未】{comp}")

    def log_to_spreadsheet(component_list, prod_name=""):
        if not component_list:
            return
        clean_prod = prod_name.strip()
        if clean_prod:
            safe_prod = clean_prod.replace(",", " ").replace("、", " ")
            current_list_str = f"【製品: {safe_prod}】, " + ",".join(component_list)
        else:
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

    log_to_spreadsheet(all_analyzed_names, product_name)

    if unmatched_names:
        st.error(f"🚨 **データベース未登録**: 以下の {len(unmatched_names)} 成分がデータベースに見つかりませんでした。")
        st.code("\n".join(unmatched_names), language="text")
        st.caption("※管理者のダッシュボードに報告しました。順次データベースへ追加されます。")

    # -------------------------
    # 4. 評価とチャート描画
    # -------------------------
    if matched_data:
        df_matched = pd.DataFrame(matched_data)
        safe_product_name = html.escape(product_name.strip())
        
        st.divider()
        
        if safe_product_name:
            st.markdown(f"""
            <div style="margin-bottom: 20px; padding: 14px 24px; background-color: #FFFFFF; border-left: 4px solid #232526; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.03);">
                <span style="font-size: 11px; font-weight: 600; color: #888; letter-spacing: 2px; display: block; margin-bottom: 4px;">TARGET PRODUCT / 評価対象製品</span>
                <span style="font-size: 20px; font-weight: 500; color: #111; letter-spacing: 0.5px;">{safe_product_name}</span>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<h3 style='color:#333; font-weight:400; letter-spacing:1px; margin-bottom:15px;'>01. 総合評価プロファイル</h3>", unsafe_allow_html=True)
        
        # ★ 浸透リスク（最小MW）は光重合開始剤を除外して評価（開始剤のみ入力時は全成分対象）
        df_mw_target = df_matched[df_matched['区分'] != '開始剤']
        if df_mw_target.empty:
            df_mw_target = df_matched

        worst_mw = df_mw_target['MW'].min()
        worst_mw_name = df_mw_target.loc[df_mw_target['MW'].idxmin()]['表示用名称']
        
        worst_ec3 = df_matched['EC3'].min()
        worst_ec3_name = df_matched.loc[df_matched['EC3'].idxmin()]['表示用名称']
        
        worst_pii = df_matched['PII'].max()
        worst_pii_name = df_matched.loc[df_matched['PII'].idxmax()]['表示用名称']
        
        total_risk_idx, (ts1, ts2, ts3) = calc_risk_index(worst_mw, worst_ec3, worst_pii)
        
        render_risk_bar(total_risk_idx)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("浸透リスク (最小MW)", f"{worst_mw} Da", f"原因: {worst_mw_name}", delta_color="inverse")
        col2.metric("アレルギー (最低EC3)", f"{worst_ec3} %", f"原因: {worst_ec3_name}", delta_color="inverse")
        col3.metric("刺激リスク (最高PII)", f"{worst_pii}", f"原因: {worst_pii_name}", delta_color="inverse")
        
        st.markdown("<p style='font-size: 11px; color: #888; margin-top: 8px; margin-bottom: 12px;'>※総合評価における浸透リスク（最小MW）は、主成分である樹脂・モノマー等の設計を正確に評価するため、微量配合成分である光重合開始剤を除外して算出しています。</p>", unsafe_allow_html=True)
        
        if total_risk_idx < 25.0 and max(ts1, ts2, ts3) >= 8.0:
            st.warning("⚠️ **単一リスク突出警告**: 総合リスクは低めですが、特定のリスク項目が極めて高い水準にあります。")
        
        fig_total, _, _ = create_radar_chart(worst_mw, worst_ec3, worst_pii, title=f"全体のリスクバランス", figsize=(5.0, 5.0), font_size=10)
        
        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            st.pyplot(fig_total)
            plt.close(fig_total)

        # ★ ここで強制改ページ（1ページ目はここまで。以降は次のページから開始）
        st.markdown('<div style="page-break-before: always; clear: both;"></div>', unsafe_allow_html=True)
        
        st.divider()
        product_sub_label = f"<span style='font-size: 14px; color: #777; font-weight: 400; margin-left: 12px;'>— {safe_product_name}</span>" if safe_product_name else ""
        st.markdown(f"<h3 style='color:#333; font-weight:400; letter-spacing:1px; margin-bottom:15px;'>02. 成分別 リスクプロファイル{product_sub_label}</h3>", unsafe_allow_html=True)
        
        # ★ 1ページに最大6個（2行×3列）ずつ出力し、6個ごとに強制改ページを入れるロジック
        items_per_page = 6
        cols_per_row = 3
        
        for page_start in range(0, len(matched_data), items_per_page):
            if page_start > 0:
                st.markdown('<div style="page-break-before: always; clear: both;"></div>', unsafe_allow_html=True)
            
            page_items = matched_data[page_start:page_start+items_per_page]
            
            for row_start in range(0, len(page_items), cols_per_row):
                cols = st.columns(cols_per_row)
                row_items = page_items[row_start:row_start+cols_per_row]
                
                for j, item in enumerate(row_items):
                    with cols[j]:
                        with st.container(border=True):
                            fig_indiv, indiv_risk_idx, (is1, is2, is3) = create_radar_chart(
                                item['MW'], item['EC3'], item['PII'], figsize=(3.8, 3.8), font_size=8
                            )
                            st.markdown(f"<div style='font-size:14px; font-weight:600; color:#333; text-align:center;'>{item['表示用名称']}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='font-size:12px; color:#888; text-align:center; margin-bottom:10px;'>リスク指数: <span style='color:#333; font-weight:bold;'>{indiv_risk_idx}%</span></div>", unsafe_allow_html=True)
                            
                            if indiv_risk_idx < 25.0 and max(is1, is2, is3) >= 8.0:
                                st.caption("🚨 **要注意成分**")
                                
                            st.pyplot(fig_indiv)
                            plt.close(fig_indiv)
                            st.caption(f"MW: **{item['MW']} Da** | EC3: **{item['EC3']}%** | PII: **{item['PII']}**")
                            if item['備考']:
                                st.markdown(f"<div style='font-size:11px; color:#666; padding:8px; background:#F9F9F9; border-radius:4px;'>{item['備考']}</div>", unsafe_allow_html=True)
        
        # 最後に免責事項を配置
        disclaimer_html = """
        <div style="page-break-inside: avoid; margin-top: 40px; padding: 20px; border: 1px solid #EAEAEA; border-radius: 8px; background-color: #FAFAFC; font-size: 11px; color: #888; line-height: 1.6; text-align: justify;">
            <strong style="color: #555; letter-spacing: 1px;">DISCLAIMER / 免責事項</strong><br>
            本シミュレーターは、各成分の特性データ（分子量・EC3・PII等）および独自のアルゴリズムに基づき、理論的なリスクスコアを算出するツールです。本評価結果は製品の絶対的な安全性を保証・証明するものではなく、アレルギー等の皮膚トラブルが一切発生しないことをお約束するものではありません。実際の製品の安全性は、配合バランス、硬化条件、およびご使用される方の体質・健康状態により異なります。本データはあくまで成分設計および安全性評価のための参考情報としてご活用ください。
        </div>
        """
        st.markdown(disclaimer_html, unsafe_allow_html=True)
        
        render_pdf_export_button()
