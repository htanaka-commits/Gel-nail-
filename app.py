import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import unicodedata
import google.generativeai as genai
import json
from PIL import Image

# -------------------------
# 1. 成分データベース (全55成分・EC3妥当性備考付き)
# -------------------------
db_data = [
    # --- モノマー・オリゴマー・開始剤（40成分） ---
    {
        '代表名': 'メタクリル酸2-ヒドロキシエチル',
        '別名リスト': ['HEMA', 'メタクリル酸2-ヒドロキシエチル', 'メタクリル酸ヒドロキシエチル'],
        'MW': 130, 'EC3': 18.0, 'PII': 2.5,
        '備考': '【妥当性】文献値10〜21%。臨床アレルギー報告は最多だが、純物質のLLNA感作能自体は中等度〜弱（アクリレートより低い）。'
    },
    {
        '代表名': 'ジカルバミン酸ジHEMAトリメチルヘキシル',
        '別名リスト': ['UDMA', 'ジカルバミン酸ジHEMAトリメチルヘキシル', 'ウレタンジメタクリレート', 'ウレタンアクリレート'],
        'MW': 470, 'EC3': 35.0, 'PII': 0.5,
        '備考': '【妥当性】高分子量かつ剛直な構造のため皮膚浸透性が極めて低く、LLNAでは実質非感作（陰性）。'
    },
    {
        '代表名': 'アクリル酸',
        '別名リスト': ['アクリル酸', 'AA'],
        'MW': 72, 'EC3': 1.5, 'PII': 8.0,
        '備考': '【妥当性】EC3約1.3〜1.5%。極小分子で強力な感作性と強酸による腐食性（高PII）を併せ持つ。'
    },
    {
        '代表名': 'アクリル酸イソボルニル',
        '別名リスト': ['IBOA', 'アクリル酸イソボルニル'],
        'MW': 208, 'EC3': 2.25, 'PII': 3.5,
        '備考': '【妥当性】ECHAデータ等でEC3=2.25%（強感作性）。剛直な環状構造を持つが感作能は高い。'
    },
    {
        '代表名': 'ジメタクリル酸テトラエチレングリコール',
        '別名リスト': ['TTEGDMA', 'ジメタクリル酸テトラエチレングリコール', 'テトラエチレングリコールジメタクリレート'],
        'MW': 330, 'EC3': 35.0, 'PII': 0.5,
        '備考': '【妥当性】PEG鎖を持つジメタクリレート。分子量が大きく皮膚浸透性が低いため実質非感作。'
    },
    {
        '代表名': 'トリアクリル酸PPG-3グリセリルエーテル',
        '別名リスト': ['PPG-3-GTA', 'トリアクリル酸PPG-3グリセリルエーテル', 'PPG-3グリセリルトリアクリレート'],
        'MW': 428, 'EC3': 1.75, 'PII': 2.5,
        '備考': '【妥当性】多官能アクリレート特有の強感作性（EC3<2%）。分子量は大きいが架橋能が要因となる。'
    },
    {
        '代表名': 'テトラメルカプトプロピオン酸ペンタエリスリチル',
        '別名リスト': ['PTMP', 'テトラメルカプトプロピオン酸ペンタエリスリチル', 'ペンタエリスリトールテトラキス(3-メルカプトプロピオネート)'],
        'MW': 488, 'EC3': 1.75, 'PII': 2.0,
        '備考': '【妥当性】チオール化合物特有の強い感作性を示す（ECHAデータで強感作性区分）。'
    },
    {
        '代表名': 'メタクリル酸ヒドロキシプロピル',
        '別名リスト': ['HPMA', 'メタクリル酸ヒドロキシプロピル', 'メタクリル酸2-ヒドロキシプロピル'],
        'MW': 144, 'EC3': 30.0, 'PII': 2.0,
        '備考': '【妥当性】HEMAに比べ疎水性が高く、LLNA試験では非常に弱い感作性（ほぼ陰性）を示す。'
    },
    {
        '代表名': 'メタクリル酸メチル',
        '別名リスト': ['MMA', 'メタクリル酸メチル'],
        'MW': 100, 'EC3': 35.0, 'PII': 5.5,
        '備考': '【妥当性】文献値EC3=60〜90%（実質非感作）。揮発性や刺激（PII）は強いがアレルギー感作能は極めて低い。'
    },
    {
        '代表名': 'メタクリル酸エチル',
        '別名リスト': ['EMA', 'メタクリル酸エチル'],
        'MW': 114, 'EC3': 35.0, 'PII': 3.0,
        '備考': '【妥当性】MMA同様、メタクリレート単官能はLLNAで実質非感作（陰性）または極めて微弱。'
    },
    {
        '代表名': 'メタクリル酸テトラヒドロフルフリル',
        '別名リスト': ['THFMA', 'メタクリル酸テトラヒドロフルフリル'],
        'MW': 170, 'EC3': 15.0, 'PII': 2.5,
        '備考': '【妥当性】文献値10〜15%程度。中等度の感作性を持つ密着性向上モノマー。'
    },
    {
        '代表名': 'メタクリル酸イソボルニル',
        '別名リスト': ['IBOMA', 'メタクリル酸イソボルニル'],
        'MW': 222, 'EC3': 28.0, 'PII': 1.5,
        '備考': '【妥当性】アクリレート版（IBOA）と異なり、メタクリレート化により感作性が著しく低下（弱感作性）。'
    },
    {
        '代表名': 'ジメタクリル酸エチレングリコール',
        '別名リスト': ['EGDMA', 'ジメタクリル酸エチレングリコール', 'エチレングリコールジメタクリレート'],
        'MW': 198, 'EC3': 28.0, 'PII': 1.5,
        '備考': '【妥当性】低分子ジメタクリレートだが、感作性は弱い（EC3 25〜30%付近）。'
    },
    {
        '代表名': 'ジメタクリル酸トリエチレングリコール',
        '別名リスト': ['TEGDMA', 'ジメタクリル酸トリエチレングリコール', 'トリエチレングリコールジメタクリレート'],
        'MW': 286, 'EC3': 32.0, 'PII': 1.0,
        '備考': '【妥当性】歯科材料でもおなじみだが、LLNAでは極めて弱い感作性〜陰性。'
    },
    {
        '代表名': 'ジメタクリル酸PEG-4',
        '別名リスト': ['PEG-4 DMA', 'PEG4DMA', 'ジメタクリル酸PEG-4', 'ジメタクリル酸ポリエチレングリコール'],
        'MW': 330, 'EC3': 35.0, 'PII': 0.5,
        '備考': '【妥当性】TTEGDMAと同等。分子量増大により皮膚浸透・感作リスクは実質ない。'
    },
    {
        '代表名': 'ジメタクリル酸1,6-ヘキサンジオール',
        '別名リスト': ['HDDMA', 'ジメタクリル酸1,6-ヘキサンジオール', 'ヘキサンジオールジメタクリレート'],
        'MW': 254, 'EC3': 25.0, 'PII': 1.5,
        '備考': '【妥当性】弱〜中等度の感作性。アクリレート版（HDDA）と比べ劇的に安全性が高い。'
    },
    {
        '代表名': 'ジアクリル酸1,6-ヘキサンジオール',
        '別名リスト': ['HDDA', 'ジアクリル酸1,6-ヘキサンジオール', 'ヘキサンジオールジアクリレート'],
        'MW': 226, 'EC3': 1.6, 'PII': 5.0,
        '備考': '【妥当性】EC3 1.5%前後。典型的な強感作性多官能アクリレートであり要注意成分。'
    },
    {
        '代表名': 'トリメチロールプロパントリアクリレート',
        '別名リスト': ['TMPTA', 'トリメチロールプロパントリアクリレート', 'トリアクリル酸トリメチロールプロパン'],
        'MW': 296, 'EC3': 1.3, 'PII': 4.5,
        '備考': '【妥当性】EC3 1.3%。3官能アクリレートのため極めて強い感作性（Extreme）を示す。'
    },
    {
        '代表名': 'ビスフェノールAグリシジルメタクリレート',
        '別名リスト': ['BIS-GMA', 'BISGMA', 'ビスフェノールAグリシジルメタクリレート', 'Bis-GMA'],
        'MW': 512, 'EC3': 35.0, 'PII': 0.5,
        '備考': '【妥当性】高分子・剛直構造のため浸透せず非感作性。'
    },
    {
        '代表名': 'エトキシ化ビスフェノールAジメタクリレート',
        '別名リスト': ['BIS-EMA', 'BISEMA', 'エトキシ化ビスフェノールAジメタクリレート', 'ビスフェノールAポリエチレングリコールジエーテルジメタクリレート'],
        'MW': 452, 'EC3': 35.0, 'PII': 0.5,
        '備考': '【妥当性】Bis-GMA同様に分子量が大きく非感作性。'
    },
    {
        '代表名': 'アクリル酸2-ヒドロキシエチル',
        '別名リスト': ['HEA', 'アクリル酸2-ヒドロキシエチル', 'アクリル酸ヒドロキシエチル'],
        'MW': 116, 'EC3': 1.4, 'PII': 6.0,
        '備考': '【妥当性】HEMAのアクリレート版。メチル基がないだけで感作性が10倍以上跳ね上がる（極強感作性）。'
    },
    {
        '代表名': 'アクリル酸ヒドロキシプロピル',
        '別名リスト': ['HPA', 'アクリル酸ヒドロキシプロピル', 'アクリル酸2-ヒドロキシプロピル'],
        'MW': 130, 'EC3': 1.8, 'PII': 5.5,
        '備考': '【妥当性】HPMAのアクリレート版。こちらも極めて強い感作性を示す。'
    },
    {
        '代表名': 'アクリル酸ブチル',
        '別名リスト': ['BA', 'アクリル酸ブチル', 'アクリル酸n-ブチル'],
        'MW': 128, 'EC3': 11.0, 'PII': 3.5,
        '備考': '【妥当性】文献値11%前後。中等度の感作性を示す。'
    },
    {
        '代表名': 'メタクリル酸ブチル',
        '別名リスト': ['BMA', 'メタクリル酸ブチル', 'メタクリル酸n-ブチル'],
        'MW': 142, 'EC3': 35.0, 'PII': 2.0,
        '備考': '【妥当性】メタクリレート化によりLLNA感作性は非感作（陰性）レベルに低下。'
    },
    {
        '代表名': 'メタクリル酸ラウリル',
        '別名リスト': ['LMA', 'メタクリル酸ラウリル', 'メタクリル酸ドデシル'],
        'MW': 254, 'EC3': 35.0, 'PII': 1.0,
        '備考': '【妥当性】長鎖アルキル基により疎水性が高く、感作性・刺激性ともに低い。'
    },
    {
        '代表名': 'アクリル酸2-エチルヘキシル',
        '別名リスト': ['2-EHA', '2EHA', 'EHA', 'アクリル酸2-エチルヘキシル', 'アクリル酸エチルヘキシル'],
        'MW': 184, 'EC3': 9.7, 'PII': 3.0,
        '備考': '【妥当性】中等度〜やや強い感作性（EC3 約10%未満）。'
    },
    {
        '代表名': 'メタクリル酸2-エチルヘキシル',
        '別名リスト': ['2-EHMA', '2EHMA', 'EHMA', 'メタクリル酸2-エチルヘキシル', 'メタクリル酸エチルヘキシル'],
        'MW': 198, 'EC3': 30.0, 'PII': 1.5,
        '備考': '【妥当性】アクリレート版に比べ感作性は激減（弱感作性）。'
    },
    {
        '代表名': 'メタクリル酸ベンジル',
        '別名リスト': ['BZMA', 'BzMA', 'メタクリル酸ベンジル'],
        'MW': 176, 'EC3': 20.0, 'PII': 2.0,
        '備考': '【妥当性】弱い感作性を示す芳香族メタクリレート。'
    },
    {
        '代表名': 'アクリル酸2-フェノキシエチル',
        '別名リスト': ['PEA', '2-PEA', 'アクリル酸2-フェノキシエチル', 'アクリル酸フェノキシエチル'],
        'MW': 192, 'EC3': 8.5, 'PII': 2.5,
        '備考': '【妥当性】中〜強感作性（EC3 約8.5%）。希釈剤として多用されるため注意が必要。'
    },
    {
        '代表名': 'メタクリル酸グリシジル',
        '別名リスト': ['GMA', 'メタクリル酸グリシジル'],
        'MW': 142, 'EC3': 0.8, 'PII': 6.5,
        '備考': '【妥当性】エポキシ基＋メタクリロイル基の二重リスク。EC3<1%の超・強感作性（Extreme）。'
    },
    {
        '代表名': 'ジメタクリル酸1,4-ブタンジオール',
        '別名リスト': ['BDDMA', 'ジメタクリル酸1,4-ブタンジオール', 'ブタンジオールジメタクリレート'],
        'MW': 226, 'EC3': 25.0, 'PII': 1.5,
        '備考': '【妥当性】弱感作性。架橋剤の中では比較的安全。'
    },
    {
        '代表名': 'ジアクリル酸トリプロピレングリコール',
        '別名リスト': ['TPGDA', 'ジアクリル酸トリプロピレングリコール', 'トリプロピレングリコールジアクリレート'],
        'MW': 300, 'EC3': 12.0, 'PII': 3.5,
        '備考': '【妥当性】DPGDAより分子量が大きく、感作性はややマイルド（中等度）。'
    },
    {
        '代表名': 'ジアクリル酸ジプロピレングリコール',
        '別名リスト': ['DPGDA', 'ジアクリル酸ジプロピレングリコール', 'ジプロピレングリコールジアクリレート'],
        'MW': 242, 'EC3': 3.5, 'PII': 4.5,
        '備考': '【妥当性】TPGDAより低分子で強い感作性（強感作性）を示す定番希釈剤。'
    },
    {
        '代表名': 'テトラアクリル酸ペンタエリスリチル',
        '別名リスト': ['PETA', 'テトラアクリル酸ペンタエリスリチル', 'ペンタエリスリトールテトラアクリレート'],
        'MW': 352, 'EC3': 2.5, 'PII': 3.5,
        '備考': '【妥当性】4官能の強感作性アクリレート（EC3 約2.5%）。'
    },
    {
        '代表名': 'ヘキサアクリル酸ジペンタエリスリチル',
        '別名リスト': ['DPHA', 'ヘキサアクリル酸ジペンタエリスリチル', 'ジペンタエリスリトールヘキサアクリレート'],
        'MW': 578, 'EC3': 12.0, 'PII': 2.0,
        '備考': '【妥当性】6官能。分子量は大きいが架橋能が高いため中等度の感作性が残存。'
    },
    {
        '代表名': 'エポキシアクリレート',
        '別名リスト': ['EA', 'エポキシアクリレート', 'ビスフェノールAエポキシアクリレート', 'アクリル酸エポキシ'],
        'MW': 500, 'EC3': 25.0, 'PII': 1.0,
        '備考': '【妥当性】高分子オリゴマーであり感作性は弱い。'
    },
    {
        '代表名': 'ポリエステルアクリレート',
        '別名リスト': ['PEA-OLIGOMER', 'ポリエステルアクリレート'],
        'MW': 600, 'EC3': 30.0, 'PII': 0.5,
        '備考': '【妥当性】高分子オリゴマーであり実質非感作〜弱感作。'
    },
    {
        '代表名': 'メタクリル酸',
        '別名リスト': ['MAA', 'メタクリル酸'],
        'MW': 86, 'EC3': 20.0, 'PII': 7.5,
        '備考': '【妥当性】強酸による一次刺激・腐食性（高PII）が主たる毒性。純粋なアレルギー感作能（EC3）は中等度。'
    },
    {
        '代表名': 'トリメチルベンゾイルジフェニルホスフィンオキシド',
        '別名リスト': ['TPO', 'トリメチルベンゾイルジフェニルホスフィンオキシド', 'ジフェニル(2,4,6-トリメチルベンゾイル)ホスフィンオキシド'],
        'MW': 348, 'EC3': 8.0, 'PII': 1.5,
        '備考': '【妥当性】中〜強感作性（EC3 8%程度）。欧州では生殖毒性（CMR）も合わせて規制対象。'
    },
    {
        '代表名': 'ヒドロキシシクロヘキシルフェニルケトン',
        '別名リスト': ['HCPK', 'ヒドロキシシクロヘキシルフェニルケトン', '1-ヒドロキシシクロヘキシルフェニルケトン'],
        'MW': 204, 'EC3': 25.0, 'PII': 1.0,
        '備考': '【妥当性】汎用開始剤だが感作性は弱く（EC3 25%）、毒性リスクは低い。'
    },

    # --- 有機溶剤系（15成分） ---
    {
        '代表名': '酢酸エチル',
        '別名リスト': ['ETHYL ACETATE', '酢酸エチル'],
        'MW': 88, 'EC3': 35.0, 'PII': 3.0,
        '備考': '【妥当性】非感作性のエステル系溶剤。強力な脱脂作用による乾燥・刺激に注意。'
    },
    {
        '代表名': '酢酸ブチル',
        '別名リスト': ['BUTYL ACETATE', '酢酸ブチル', '酢酸n-ブチル'],
        'MW': 116, 'EC3': 35.0, 'PII': 2.5,
        '備考': '【妥当性】非感作性。酢酸エチルより揮発が遅く刺激もややマイルド。'
    },
    {
        '代表名': '酢酸プロピル',
        '別名リスト': ['PROPYL ACETATE', '酢酸プロピル', '酢酸n-プロピル'],
        'MW': 102, 'EC3': 35.0, 'PII': 2.5,
        '備考': '【妥当性】非感作性のエステル系溶剤。'
    },
    {
        '代表名': '酢酸イソプロピル',
        '別名リスト': ['ISOPROPYL ACETATE', '酢酸イソプロピル'],
        'MW': 102, 'EC3': 35.0, 'PII': 3.0,
        '備考': '【妥当性】非感作性のエステル系溶剤。'
    },
    {
        '代表名': 'エタノール',
        '別名リスト': ['ETHANOL', 'ALCOHOL', 'エタノール', '変性アルコール', '無水エタノール'],
        'MW': 46, 'EC3': 35.0, 'PII': 2.5,
        '備考': '【妥当性】非感作性。分子量は極小（浸透性大）だが毒性は脱脂乾燥のみ。'
    },
    {
        '代表名': 'イソプロパノール',
        '別名リスト': ['IPA', 'ISOPROPYL ALCOHOL', 'ISOPROPANOL', 'イソプロパノール', 'イソプロピルアルコール'],
        'MW': 60, 'EC3': 35.0, 'PII': 3.0,
        '備考': '【妥当性】非感作性。エタノールより脱脂力が強く刺激（PII）が高い。'
    },
    {
        '代表名': 'ブタノール',
        '別名リスト': ['BUTANOL', 'BUTYL ALCOHOL', 'ブタノール', 'n-ブタノール', '1-ブタノール'],
        'MW': 74, 'EC3': 35.0, 'PII': 4.0,
        '備考': '【妥当性】非感作性。粘膜や皮膚への一次刺激性が比較的強い。'
    },
    {
        '代表名': 'ジアセトンアルコール',
        '別名リスト': ['DAA', 'DIACETONE ALCOHOL', 'ジアセトンアルコール'],
        'MW': 116, 'EC3': 35.0, 'PII': 3.5,
        '備考': '【妥当性】非感作性溶剤。'
    },
    {
        '代表名': 'アセトン',
        '別名リスト': ['ACETONE', 'アセトン'],
        'MW': 58, 'EC3': 35.0, 'PII': 4.0,
        '備考': '【妥当性】非感作性。極小分子量で強力な脱脂力を持つため皮膚バリアを破壊（高PII）する。'
    },
    {
        '代表名': 'メチルエチルケトン',
        '別名リスト': ['MEK', 'METHYL ETHYL KETONE', 'メチルエチルケトン', 'ブタノン'],
        'MW': 72, 'EC3': 35.0, 'PII': 4.5,
        '備考': '【妥当性】非感作性だが揮発・刺激性がアセトンより強いケトン系溶剤。'
    },
    {
        '代表名': 'ヘプタン',
        '別名リスト': ['HEPTANE', 'ヘプタン', 'n-ヘプタン'],
        'MW': 100, 'EC3': 35.0, 'PII': 3.5,
        '備考': '【妥当性】非感作性の炭化水素系溶剤。'
    },
    {
        '代表名': 'トルエン',
        '別名リスト': ['TOLUENE', 'トルエン'],
        'MW': 92, 'EC3': 35.0, 'PII': 5.5,
        '備考': '【妥当性】非感作性。ただし神経毒性・生殖毒性および強い一次刺激を持つ要注意溶剤。'
    },
    {
        '代表名': '炭酸プロピレン',
        '別名リスト': ['PROPYLENE CARBONATE', '炭酸プロピレン', 'プロピレンカーボネート'],
        'MW': 102, 'EC3': 35.0, 'PII': 1.5,
        '備考': '【妥当性】非感作性。揮発しにくく刺激も少ない安全性の高い溶剤。'
    },
    {
        '代表名': 'エトキシジグリコール',
        '別名リスト': ['ETHOXYDIGLYCOL', 'ジエチレングリコールモノエチルエーテル', 'エトキシジグリコール'],
        'MW': 134, 'EC3': 35.0, 'PII': 1.5,
        '備考': '【妥当性】非感作性。浸透性は高いが刺激は少ない。'
    },
    {
        '代表名': 'テトラヒドロフラン',
        '別名リスト': ['THF', 'TETRAHYDROFURAN', 'テトラヒドロフラン'],
        'MW': 72, 'EC3': 35.0, 'PII': 5.0,
        '備考': '【妥当性】非感作性。強力な溶解力による脱脂刺激と揮発ガスに注意。'
    }
]

def normalize_text(text):
    return unicodedata.normalize('NFKC', text).strip().upper()

# -------------------------
# 2. リスク計算＆リスク指数算出関数
# -------------------------
def norm_mw(mw): return max(0.0, min(10.0, (500 - mw) / 45))
def norm_ec3(ec3): return max(0.0, min(10.0, (35 - ec3) / 3.5))
def norm_pii(pii): return max(0.0, min(10.0, pii * (10/8.0)))

def calc_risk_index(mw, ec3, pii):
    s1 = norm_mw(mw)
    s2 = norm_ec3(ec3)
    s3 = norm_pii(pii)
    risk_idx = ((s1 * s2 + s2 * s3 + s3 * s1) / 300.0) * 100.0
    return round(risk_idx, 1), (s1, s2, s3)

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
    
    max_score = max(s1, s2, s3)
    if risk_idx >= 40.0 or ec3 <= 2.5 or max_score >= 8.0:
        color = '#d62728'
    elif risk_idx >= 15.0 or ec3 <= 15.0 or max_score >= 5.0:
        color = '#ff7f0e'
    else:
        color = '#2ca02c'
        
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

if 'input_text' not in st.session_state:
    st.session_state['input_text'] = "ジカルバミン酸ジＨＥＭＡトリメチルヘキシル, メタクリル酸2-ヒドロキシエチル, 酢酸エチル, (未知の成分例) HEMA-P"

st.sidebar.title("📚 成分データベース情報")
st.sidebar.info(f"現在 **{len(db_data)}種類** の成分がローカル登録されています。")

search_filter = st.sidebar.text_input("データベース内をキーワード検索", placeholder="例: メタクリル酸, 酢酸")
with st.sidebar.expander("登録済み全成分リストを見る", expanded=True):
    for item in db_data:
        if not search_filter or normalize_text(search_filter) in normalize_text(item['代表名']) or any(normalize_text(search_filter) in normalize_text(a) for a in item['別名リスト']):
            st.caption(f"・**{item['代表名']}** ({item['別名リスト'][0]})")

st.sidebar.markdown("---")
st.sidebar.title("🔑 API連携ステータス")
api_key = ""
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("✅ Gemini APIキー（クラウド設定済み）\n\nOCRおよび未登録成分の自動AI検索が利用可能です。")
else:
    api_key = st.sidebar.text_input("Gemini APIキー", type="password", help="OCRや未登録成分のAI検索に必要です")

st.title("💅 ジェルネイル成分 リスク評価シミュレーター")
st.markdown("ローカルデータベースを優先参照し、**未登録の成分についてはAIが公的機関（ECHA等）のデータを元に毒性値を推論・補完**します。")

with st.expander("📸 パッケージの写真から全成分を読み取る", expanded=False):
    uploaded_file = st.file_uploader("全成分表示が写った画像をアップロード、またはカメラで撮影", type=['png', 'jpg', 'jpeg'])
    if st.button("画像をテキスト化する"):
        if not api_key:
            st.error("APIキーが設定されていません。")
        elif not uploaded_file:
            st.error("画像をアップロードしてください。")
        else:
            with st.spinner("AIが画像内の文字を解析中です..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-3.8-flash')
                    img = Image.open(uploaded_file).convert('RGB')
                    prompt = "この写真に含まれる「全成分表示」を読み取り、化粧品の成分名だけを抽出してカンマ(,)区切りで出力してください。関係ない文字は含めないでください。"
                    response = model.generate_content([prompt, img])
                    st.session_state['input_text'] = response.text.strip()
                    st.success("読み取りが完了しました！下の入力ボックスに自動反映されました。")
                    st.rerun()
                except Exception as e:
                    st.error(f"読み取り中にエラーが発生しました: {e}")

input_text = st.text_area("▼ ここに成分表をカンマ（,）または読点（、）区切りで入力", 
                          value=st.session_state['input_text'], height=100)

if st.button("リスクを解析する"):
    st.session_state['input_text'] = input_text
    components = [c.strip() for c in input_text.replace('、', ',').split(',') if c.strip()]
    
    if not components:
        st.warning("成分が入力されていません。")
        st.stop()

    matched_data = []
    unmatched_names = []
    
    # 1. ローカルデータベース検索
    for comp in components:
        norm_comp = normalize_text(comp)
        found_item = None
        for item in db_data:
            normalized_aliases = [normalize_text(alias) for alias in item['別名リスト']]
            if norm_comp in normalized_aliases:
                found_item = item
                break
        
        if found_item:
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

    # 2. 未登録成分に対するAI自動検索（フォールバック）
    if unmatched_names:
        if not api_key:
            st.warning(f"以下の成分はデータベースに未登録です。AIによる自動検索を行うにはAPIキーを設定してください。\n\n未登録成分: {', '.join(unmatched_names)}")
        else:
            with st.spinner("未登録成分の毒性データをAI（公的機関データソース）で検索しています..."):
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-3.8-flash')
                
                for u_comp in unmatched_names:
                    prompt = f"""
                    あなたは毒性学の専門家です。ジェルネイル・化粧品成分「{u_comp}」について、ECHA（欧州化学品庁）、CIR（化粧品成分審査パネル）、SCCS（欧州消費者安全科学委員会）などの公的機関のデータに基づき、以下の情報をJSON形式のみで出力してください。
                    正確な値が不明な場合は、類似の化学構造から推定されるワーストケースの値を出力してください。
                    
                    出力フォーマット（これ以外の文字やMarkdown記号は絶対に出力しないでください）:
                    {{
                        "JP_NAME": "日本語の正式な成分名",
                        "MW": 分子量(数値のみ),
                        "EC3": LLNA EC3値%(数値のみ、非感作性なら35.0),
                        "PII": 一次刺激性指数(0〜8の数値のみ),
                        "REMARKS": "【AI推論/出典: (参考にした公的機関名)】 ...妥当性の説明..."
                    }}
                    """
                    try:
                        response = model.generate_content(prompt)
                        json_str = response.text.replace('```json', '').replace('```', '').strip()
                        ai_data = json.loads(json_str)
                        
                        matched_data.append({
                            '成分名': ai_data.get('JP_NAME', u_comp),
                            '表示用名称': f"{ai_data.get('JP_NAME', u_comp)} (AI検索)",
                            'MW': float(ai_data.get('MW', 100)),
                            'EC3': float(ai_data.get('EC3', 1.5)),
                            'PII': float(ai_data.get('PII', 5.0)),
                            '備考': ai_data.get('REMARKS', '【AI推論】詳細不明')
                        })
                    except Exception as e:
                        st.error(f"「{u_comp}」のAI検索中にエラーが発生しました: {e}")

    # -------------------------
    # 4. 評価とチャート描画
    # -------------------------
    if matched_data:
        df_matched = pd.DataFrame(matched_data)
        
        st.divider()
        st.subheader("📊 1. 製品全体の総合評価（ワーストケース・ボトルネック）")
        
        base_components = df_matched.head(3)
        worst_mw = base_components['MW'].min()
        worst_mw_name = base_components.loc[base_components['MW'].idxmin()]['表示用名称']
        
        worst_ec3 = df_matched['EC3'].min()
        worst_ec3_name = df_matched.loc[df_matched['EC3'].idxmin()]['表示用名称']
        
        worst_pii = df_matched['PII'].max()
        worst_pii_name = df_matched.loc[df_matched['PII'].idxmax()]['表示用名称']
        
        total_risk_idx, (ts1, ts2, ts3) = calc_risk_index(worst_mw, worst_ec3, worst_pii)
        
        fig_total, _, _ = create_radar_chart(
            worst_mw, worst_ec3, worst_pii, 
            title=f"総合リスクプロファイル (リスク指数: {total_risk_idx}%)", 
            figsize=(5.5, 5.5), font_size=10
        )
        
        col0, col1, col2, col3 = st.columns(4)
        col0.metric("🔥 総合リスク指数", f"{total_risk_idx} %", "最大面積100%比", delta_color="inverse")
        col1.metric("浸透リスク (最小MW)", f"{worst_mw} Da", f"原因: {worst_mw_name}", delta_color="inverse")
        col2.metric("アレルギーリスク (最低EC3)", f"{worst_ec3} %", f"原因: {worst_ec3_name}", delta_color="inverse")
        col3.metric("刺激リスク (最高PII)", f"{worst_pii}", f"原因: {worst_pii_name}", delta_color="inverse")
        
        if total_risk_idx < 25.0 and max(ts1, ts2, ts3) >= 8.0:
            st.warning("⚠️ **単一リスク突出警告**: 総合リスク指数は低めですが、特定のリスク項目（アレルギー感作性や浸透性など）が極めて高い水準にあります。使用時は皮膚への付着に厳重な注意が必要です。")
        
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
                                item['MW'], 
                                item['EC3'], 
                                item['PII'], 
                                figsize=(4, 4), 
                                font_size=8
                            )
                            st.markdown(f"**{idx + j + 1}. {item['表示用名称']}**")
                            st.markdown(f"リスク指数: **`{indiv_risk_idx}%`**")
                            
                            if indiv_risk_idx < 25.0 and max(is1, is2, is3) >= 8.0:
                                st.caption("🚨 **単一リスク突出型（要注意）**")
                                
                            st.pyplot(fig_indiv)
                            plt.close(fig_indiv)
                            st.caption(f"MW: **{item['MW']} Da** | EC3: **{item['EC3']}%** | PII: **{item['PII']}**")
                            
                            # 未登録成分の場合、AIが取得した公的ソースの情報を表示
                            if "(AI検索)" in item['表示用名称']:
                                st.info(f"🤖 {item['備考']}")
                            else:
                                st.info(f"💡 {item['備考']}")
