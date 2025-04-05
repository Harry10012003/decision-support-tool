import streamlit as st
import pandas as pd
import numpy as np

# Hàm phân tích quyết định
def decision_making_analysis(data, probabilities=None, maximize=True, alpha=0.6):
    df = data.copy()
    criteria_cols = df.columns.tolist()

    # Xử lý Lạc Quan và Bi Quan
    df['Lạc Quan'] = df[criteria_cols].max(axis=1) if maximize else df[criteria_cols].min(axis=1)
    df['Bi Quan'] = df[criteria_cols].min(axis=1) if maximize else df[criteria_cols].max(axis=1)
    
    df['Trung Bình'] = df[criteria_cols].mean(axis=1)
    df['Realism'] = alpha * df['Lạc Quan'] + (1 - alpha) * df['Bi Quan']
    
    # Tính EMV
    df['EMV'] = df[criteria_cols].dot(probabilities) if probabilities is not None else None

    # Tính bảng Opportunity Loss
    opp_loss_table = df[criteria_cols].copy()
    for col in opp_loss_table.columns:
        best = opp_loss_table[col].max() if maximize else opp_loss_table[col].min()
        opp_loss_table[col] = abs(opp_loss_table[col] - best)
    df['Minimax Regret'] = opp_loss_table.max(axis=1)

    return df, opp_loss_table

# Hàm hiển thị bảng dữ liệu đầy đủ + 1 cột kết quả + kết luận
# Sửa
# def show_decision_full(data, result, column_name, maximize=True, label="Kết quả"):
#     df_display = data.copy()
#     df_display[column_name] = result[column_name]
#     df_display.index.name = None  # Ẩn tên index
def show_decision_full(data, result, column_name, maximize=True, label="Kết quả", include_all_rows=False):
    if include_all_rows:
        df_display = result.copy()
    else:
        df_display = data.copy()
        df_display[column_name] = result[column_name]

    df_display.index.name = None  # Ẩn tên index

    # Căn chỉnh bảng dữ liệu
    styled_df = df_display.style \
        .format(precision=0) \
        .set_properties(**{'text-align': 'center'})

    # Hiển thị bảng dữ liệu
    st.markdown(f"""
    <div style="padding:8px 20px;border-left:5px solid #fb8c00;margin-bottom:10px;">
        <h3 style="color:#fb8c00;margin:0;"> {label}</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(styled_df.to_html(), unsafe_allow_html=True)

    # Chỉ hiển thị kết luận nếu không bao gồm tất cả dòng (để tránh highlight "With perfect info")
    if not include_all_rows:
        if maximize:
            best_value = result[column_name].max()
        else:
            best_value = result[column_name].min()
        chosen_option = result[result[column_name] == best_value].index[0]

        st.markdown(f"""
        <div style="background-color:#dff0d8;padding:15px;border-radius:10px;border-left:5px solid #3c763d;">
            <h4 style="color:#2e5c2e;margin-bottom:10px;">
                ✅ <strong>Chọn phương án <span style='color:#205081'>{chosen_option}</span> theo quy tắc <span style='color:#205081'>{label}</span></strong>
            </h4>
            <p style="color:#1f2d1f;margin:0;font-size:16px;">
                Vì có giá trị <strong>{column_name}</strong> {'lớn nhất' if maximize else 'nhỏ nhất'} là: 
                <code style="color:#000;background-color:#f7f7f7;padding:4px 8px;border-radius:6px;font-weight:bold;font-size:15px;">{best_value:,.0f}</code>
            </p>
        </div>
        """, unsafe_allow_html=True)

st.set_page_config(page_title="Công cụ ra quyết định", layout="wide")

st.markdown(""" 
<div style="
    background-color:#ffe0b2;
    padding:20px;
    border-radius:12px;
    text-align:center;
    margin-bottom:30px;
    border-left:6px solid #ff9800;
">
    <h1 style="color:#333333; font-size:36px; margin:0;">
        DECISION ANALYSIS - Group 9
    </h1>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<h3 style='color:#ff9800;'>1. Mục tiêu của bạn là:</h3>", unsafe_allow_html=True)
    maximize = st.radio(
        label="",
        options=['Tối đa lợi nhuận', 'Tối thiểu chi phí'],
        index=0,
        horizontal=False
    )

    # Lưu giá trị lựa chọn vào session state
    st.session_state.maximize = True if maximize == 'Tối đa lợi nhuận' else False

    # 👇 Thêm hình minh họa ở đây
    st.image("reference.png", caption="Lưu ý: Chỉ copy phần dữ liệu như trên hình", width=350)

with col2:
    st.markdown("<h3 style='color:#ff9800;'>2. Nhập bảng dữ liệu:</h3>", unsafe_allow_html=True)

    user_input = st.text_area(
        "Dán dữ liệu vào đây:",
        value="Pro\t0.5\t0.3\t0.2\nA\t50000\t20000\t-10000\nB\t80000\t22000\t-20000\nC\t100000\t30000\t-40000\nD\t300000\t25000\t-100000",
        height=150
    )

data, probabilities = None, None

if user_input:
    try:
        from io import StringIO
        # Thay thế dấu '–' thành dấu '-'
        user_input_cleaned = user_input.replace('–', '-')
        
        data_raw = pd.read_csv(StringIO(user_input_cleaned), sep="\t", header=0, dtype=str)
        data_raw = data_raw.replace({",": ""}, regex=True)

        # Tách dòng xác suất
        probabilities_row = data_raw[data_raw.iloc[:, 0] == 'Pro']
        if probabilities_row.empty:
            st.error("❌ Không tìm thấy dòng 'Pro'!")
        else:
            probabilities = probabilities_row.iloc[0, 1:].astype(float).values
            data = data_raw[data_raw.iloc[:, 0] != 'Pro'].copy()
            data.set_index(data.columns[0], inplace=True)
            data = data.astype(float)
    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý dữ liệu: {e}")

# Tabs giao diện đơn giản hơn
if data is not None and probabilities is not None:
    st.markdown("<h3 style='color:#ff9800;'>3. Lựa chọn phương pháp ra quyết định:</h3>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔝 Lạc Quan", "🛡️ Bi Quan", "⚖️ Equally Likely",
        "🎯 Realism", "😰 Minimax Regret",
        "🔮 EVPI", "📘 EOL"
    ])

    with tab1:
        result, _ = decision_making_analysis(data, probabilities, st.session_state.maximize)  # Thay vì maximize=True
        show_decision_full(data, result, 'Lạc Quan', maximize=st.session_state.maximize, label="Lạc Quan")

    with tab2:
        result, _ = decision_making_analysis(data, probabilities, st.session_state.maximize)  # Thay vì maximize=True
        show_decision_full(data, result, 'Bi Quan', maximize=st.session_state.maximize, label="Bi Quan")

    with tab3:
        result, _ = decision_making_analysis(data, probabilities, st.session_state.maximize)  # Thay vì maximize=st.session_state.maximize
        show_decision_full(data, result, 'Trung Bình', maximize=st.session_state.maximize, label="Equally Likely")

    with tab4:
        alpha_Realism = st.number_input(
            "🎯 Nhập hệ số alpha cho Realism (0 < α < 1):",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.01,
            format="%.2f"
        )
        result, _ = decision_making_analysis(data, probabilities, st.session_state.maximize, alpha=alpha_Realism)
        show_decision_full(data, result, 'Realism', maximize=st.session_state.maximize, label=f"Realism (α = {alpha_Realism})")

    with tab5:
        result, opp_loss = decision_making_analysis(data, probabilities, st.session_state.maximize)

        # ❗️XÓA tên cột index (Unnamed: 0)
        opp_loss.index.name = None

        # 💅 Hiển thị bảng Opportunity Loss đẹp
        st.markdown("""<div style="padding:8px 20px;border-left:5px solid #e91e63;margin-bottom:10px;">
            <h3 style="color:#e91e63;margin:0;"> Bảng Opportunity Loss</h3></div>""", unsafe_allow_html=True)
        
        styled_opp_loss = opp_loss.style.format(precision=0).set_properties(**{'text-align': 'center'})
        
        st.markdown(styled_opp_loss.to_html(), unsafe_allow_html=True)

        # ✅ Kết luận
        show_decision_full(data, result, 'Minimax Regret', maximize=False, label="Minimax Regret")

    with tab6:
        # 1. Phân tích kết quả EMV
        result, _ = decision_making_analysis(data, probabilities, st.session_state.maximize)

        # 2. Tính With Perfect Info: giá trị tối ưu cho từng cột
        best_per_state = data.max(axis=0) if st.session_state.maximize else data.min(axis=0)
        EVwPI = np.dot(best_per_state.values, probabilities)

        # 3. Tính EMV từng phương án (nếu chưa có)
        result['EMV'] = data.dot(probabilities)

        # 4. Tính EVwoPI
        EVwoPI = result['EMV'].max() if st.session_state.maximize else result['EMV'].min()

        # 5. Tính EVPI
        EVPI = abs(EVwPI - EVwoPI)


        # 6. Thêm dòng "With perfect info" vào bảng kết quả
        result.loc["With perfect info", data.columns] = best_per_state
        result.loc["With perfect info", "EMV"] = EVwPI

        # 7. Giữ lại cột cần thiết
        cols_to_keep = list(data.columns) + ['EMV']
        result = result[cols_to_keep]

        # 8. Hiển thị bảng kết quả
        show_decision_full(
            data, result, 'EMV',
            maximize=st.session_state.maximize,
            label="Giá trị kỳ vọng (EMV) và Thông tin hoàn hảo",
            include_all_rows=True  # để không lọc dòng "With perfect info"
        )

        # 9. Hiển thị kết luận về EVPI
        st.markdown(f"""
        <div style="background-color:#e8f4fd;padding:15px;border-radius:10px;border-left:5px solid #1c3d5a; margin-top: 20px;">
            <h4 style="color:#1c3d5a;margin-bottom:10px;">
                🔮 <strong>Giá trị Thông tin Hoàn hảo (EVPI)</strong>
            </h4>
            <ul style="margin-left:20px;color:#1c3d5a;font-size:15px;">
                <li><strong>🧠 EVwPI</strong>: 
                    <code style="color:#000;background-color:#f7f7f7;padding:4px 8px;border-radius:6px;font-weight:bold;">{EVwPI:,.2f}</code>
                </li>
                <li><strong>🧮 EMV tốt nhất</strong>: 
                    <code style="color:#000;background-color:#f7f7f7;padding:4px 8px;border-radius:6px;font-weight:bold;">{EVwoPI:,.2f}</code>
                </li>
                <li><strong>💡 So the maximum you should pay for the additional information is </strong>: 
                    <code style="color:#000;background-color:#e0ffe0;padding:4px 8px;border-radius:6px;font-weight:bold;">{EVPI:,.2f}</code>
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab7:
        # Phân tích với trạng thái maximize hiện tại
        result, opp_loss = decision_making_analysis(data, probabilities, st.session_state.maximize)

        # Bảng EOL nhân với xác suất để tính EOL
        EOL_table_with_probabilities = opp_loss.copy()
        for i, col in enumerate(EOL_table_with_probabilities.columns):
            EOL_table_with_probabilities[col] *= probabilities[i]
        result['EOL'] = EOL_table_with_probabilities.sum(axis=1)

        # Bảng EOL không nhân xác suất – chỉ để hiển thị
        EOL_table_no_probabilities = opp_loss.copy()
        EOL_table_no_probabilities.index.name = None

        # Hiển thị bảng đẹp
        styled_eol = EOL_table_no_probabilities.style \
            .format(precision=0) \
            .set_properties(**{'text-align': 'center'}) \
            .set_table_styles([
                {'selector': 'th', 'props': [('color', '#0277bd'), ('font-weight', 'bold'), ('text-align', 'center')]}
            ])

        st.markdown("""
        <div style="padding:8px 20px;border-left:5px solid #29b6f6;margin-bottom:10px;">
            <h3 style="color:#29b6f6;margin:0;"> Bảng Expected Opportunity Loss (EOL)</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(styled_eol.to_html(), unsafe_allow_html=True)

        # Hiển thị kết quả lựa chọn phương án với EOL
        show_decision_full(data, result, 'EOL', maximize=False, label="Expected Opportunity Loss")
