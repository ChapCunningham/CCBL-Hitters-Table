import streamlit as st
import pandas as pd
import requests
from io import StringIO
import numpy as np

# Page config
st.set_page_config(
    page_title="CCBL Live Stats",
    page_icon="⚾",
    layout="wide"
)

# Custom CSS to match the College WAR table styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .dataframe {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 14px;
    }
    .stDataFrame > div {
        border: 1px solid #ddd;
        border-radius: 4px;
    }
    .stDataFrame table {
        border-collapse: collapse !important;
    }
    .stDataFrame th {
        background-color: #f8f9fa !important;
        border-bottom: 2px solid #ddd !important;
        font-weight: 600 !important;
        padding: 12px 8px !important;
        text-align: left !important;
    }
    .stDataFrame td {
        border-bottom: 1px solid #eee !important;
        padding: 8px !important;
        text-align: left !important;
    }
    .stDataFrame tr:hover {
        background-color: #f5f5f5 !important;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_csv_from_drive(drive_url):
    """Load CSV from Google Drive shareable link"""
    try:
        if '/file/d/' in drive_url:
            file_id = drive_url.split('/file/d/')[1].split('/')[0]
            direct_url = f"https://drive.google.com/uc?id={file_id}&export=download"
        else:
            direct_url = drive_url
        
        response = requests.get(direct_url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

def load_and_process_csv(df, split_by_handedness=False):
    """Process the raw CSV data into baseball statistics"""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df[df['Date'].notna()]

    # Create flags
    df['PA'] = df['PitchCall'].isin(['InPlay', 'HitByPitch']) | df['KorBB'].isin(['Strikeout', 'Walk'])
    df['AB'] = df['PitchCall'].eq('InPlay') | df['KorBB'].eq('Strikeout')
    df['BIP'] = df['PitchCall'].eq('InPlay')
    df['Hit'] = df['PlayResult'].isin(['Single', 'Double', 'Triple', 'HomeRun'])
    df['1B'] = df['PlayResult'].eq('Single')
    df['2B'] = df['PlayResult'].eq('Double')
    df['3B'] = df['PlayResult'].eq('Triple')
    df['HR'] = df['PlayResult'].eq('HomeRun')
    df['BB'] = df['KorBB'].eq('Walk')
    df['K'] = df['KorBB'].eq("Strikeout")
    df['HBP'] = df['PitchCall'].eq('HitByPitch')
    df['Barrel'] = (
        (df['PitchCall'] == 'InPlay') &
        (df['ExitSpeed'] >= 95) &
        (df['Angle'].between(10, 35)))
    df['swing'] = df['PitchCall'].isin(['StrikeSwinging', 'InPlay', 'FoulBallNotFieldable', 'FoulBallFieldable'])
    df['HardHit'] = (df['PitchCall'] == 'InPlay') & (df['ExitSpeed'] >= 95)

    # Classify BIP by launch angle
    df['GB'] = (df['PitchCall'] == 'InPlay') & (df['Angle'] < 10)
    df['LD'] = (df['PitchCall'] == 'InPlay') & (df['Angle'].between(10, 25, inclusive="left"))
    df['FB'] = (df['PitchCall'] == 'InPlay') & (df['Angle'].between(25, 50, inclusive="left"))
    df['PU'] = (df['PitchCall'] == 'InPlay') & (df['Angle'] > 50)

    # First Pitch Swing %
    df['FPSw'] = (df['Balls'] == 0) & (df['Strikes'] == 0) & df['swing']
    df['FirstPitch'] = (df['Balls'] == 0) & (df['Strikes'] == 0)

    # Define pitches in the strike zone
    df['zone'] = (
        (df['PlateLocHeight'] >= 1.5) &
        (df['PlateLocHeight'] <= 3.3775) &
        (df['PlateLocSide'] >= -0.83083) &
        (df['PlateLocSide'] <= 0.83083)
    )

    # Zone swings: swings on pitches in the strike zone
    df['zone_swing'] = df['swing'] & df['zone']

    df['contact'] = df['PitchCall'].isin(['InPlay', 'FoulBallNotFieldable', 'FoulBallFieldable'])
    df['whiff'] = df['PitchCall'].eq('StrikeSwinging')

    # Chase zone filter
    df['chase_zone'] = (
        (df['PlateLocHeight'] < 1.37750) |
        (df['PlateLocHeight'] > 3.50000) |
        (df['PlateLocSide'].abs() > 0.99750)
    )   

    # A chase attempt is a swing in the chase zone
    df['chase_swing'] = df['swing'] & df['chase_zone']
    df['OZone'] = ~df['zone']

    group_cols = ['Batter', 'BatterTeam']
    if split_by_handedness:
        group_cols.append('PitcherThrows')

    # Aggregations
    stats = df.groupby(group_cols).agg(
        PA=('PA', 'sum'),
        AB=('AB', 'sum'),
        BIP=('BIP', 'sum'),
        Hits=('Hit', 'sum'),
        BB=('BB', 'sum'),
        K=('K', 'sum'),
        HBP=('HBP', 'sum'),
        _1B=('1B', 'sum'),
        _2B=('2B', 'sum'),
        _3B=('3B', 'sum'),
        HR=('HR', 'sum'),
        LA=('Angle', 'mean'),
        Barrels=('Barrel', 'sum'),
        RV=('run_value', 'sum'),
        xwOBA=('xwOBA_result', 'mean'),
        PitchCount=('PitchCall', 'count'),
        Swings=('swing', 'sum'),
        ContactMade=('contact', 'sum'),
        Whiffs=('whiff', 'sum'),
        ChaseSwings=('chase_swing', 'sum'),
        ZoneSwings=('zone_swing', 'sum'),
        ZonePitches=('zone', 'sum'),
        OZonePitches=('OZone', 'sum'),
        FPSw=('FPSw', 'sum'),
        FirstPitchTotal=('FirstPitch', 'sum'),
        HardHits=('HardHit', 'sum'),
        GB=('GB', 'sum'),
        LD=('LD', 'sum'),
        FB=('FB', 'sum'),
        PU=('PU', 'sum'),
    ).reset_index()

    # Calculate EV stats (only non-bunt balls in play)
    ev_df = df[(df['PitchCall'] == 'InPlay') & (df['TaggedHitType'] != 'Bunt')]
    if not ev_df.empty:
        ev = ev_df.groupby(group_cols)['ExitSpeed'].mean().reset_index(name='EV')
        ev_90th = ev_df.groupby(group_cols)['ExitSpeed'].quantile(0.9).reset_index(name='90thEV')
        maxev = ev_df.groupby(group_cols)['ExitSpeed'].max().reset_index(name='maxEV')
        
        stats = stats.merge(ev, on=group_cols, how='left')
        stats = stats.merge(ev_90th, on=group_cols, how='left')
        stats = stats.merge(maxev, on=group_cols, how='left')
    else:
        stats['EV'] = np.nan
        stats['90thEV'] = np.nan
        stats['maxEV'] = np.nan

    # Calculate derived stats
    # wOBA Weights (CCBL 2024 or 2025)
    wBB = 0.673
    wHBP = 0.718
    w1B = 0.949
    w2B = 1.483
    w3B = 1.963
    wHR = 2.571

    # Compute wOBA
    stats['wOBA_numerator'] = (
        stats['BB'] * wBB +
        stats['HBP'] * wHBP +
        stats['_1B'] * w1B +
        stats['_2B'] * w2B +
        stats['_3B'] * w3B +
        stats['HR'] * wHR
    )   

    # Final wOBA
    stats['wOBA'] = stats['wOBA_numerator'] / stats['PA']
    stats['wOBA_diff'] = stats['xwOBA'] - stats['wOBA']

    stats['TB'] = stats['_1B'] + 2*stats['_2B'] + 3*stats['_3B'] + 4*stats['HR']
    stats['AVG'] = stats['Hits'] / stats['AB']
    stats['OBP'] = (stats['Hits'] + stats['BB'] + stats['HBP']) / stats['PA']
    stats['SLG'] = stats['TB'] / stats['AB']
    stats['OPS'] = stats['OBP'] + stats['SLG']
    stats['K%'] = stats['K'] / stats['PA']
    stats['BB%'] = stats['BB'] / stats['PA']
    stats['Barrel%'] = stats['Barrels'] / stats['BIP']
    stats['RV/100'] = stats['RV'] / stats['PitchCount'] * 100
    stats['Contact%'] = stats['ContactMade'] / stats['Swings']
    stats['Whiff%'] = stats['Whiffs'] / stats['Swings']
    stats['Swing%'] = stats['Swings'] / stats['PitchCount']
    stats['Chase%'] = stats['ChaseSwings'] / stats['OZonePitches']
    stats['ZSw%'] = stats['ZoneSwings'] / stats['ZonePitches']
    stats['SwStr%'] = stats['Whiffs'] / stats['PitchCount']
    stats['FPSw%'] = stats['FPSw'] / stats['FirstPitchTotal']
    stats['HardHit%'] = stats['HardHits'] / stats['BIP']
    stats['GB%'] = stats['GB'] / stats['BIP']
    stats['LD%'] = stats['LD'] / stats['BIP']
    stats['FB%'] = stats['FB'] / stats['BIP']
    stats['PU%'] = stats['PU'] / stats['BIP']

    if split_by_handedness:
        stats = stats.rename(columns={'PitchCount': 'P'})

    # Column ordering
    if split_by_handedness:
        ordered_cols = [
            'Batter', 'BatterTeam', 'PitcherThrows','PA','P', 'AVG', 'wOBA', 'xwOBA',
            'EV', 'LA', 'Contact%', 'Swing%', 'SwStr%', 'OPS', 'OBP', 'SLG',
            'Chase%', 'Whiff%', 'FPSw%', 'K%', 'BB%',
            'AB', 'BIP', 'Hits', 'K', 'BB', 'HBP',
            '_1B', '_2B', '_3B', 'HR', 'TB', 'wOBA_diff',
            'Barrels', 'Barrel%','HardHit%', 'GB%', 'LD%', 'FB%', 'PU%', '90thEV', 'maxEV', 'RV', 'RV/100',
        ]
    else:
        ordered_cols = [
            'Batter', 'BatterTeam',
            'PA', 'AB', 'PitchCount', 'BIP', 'Hits', 'K', 'BB', 'HBP',
            '_1B', '_2B', '_3B', 'HR', 'TB', 'AVG', 'OBP', 'SLG', 'OPS', 'wOBA', 'xwOBA', 'wOBA_diff',
            'K%', 'BB%', 'Swing%', 'FPSw%', 'Whiff%', 'SwStr%', 'Contact%', 'ZSw%', 'Chase%',
            'HardHit%','EV', '90thEV', 'maxEV', 'GB%', 'LD%', 'FB%', 'PU%', 'LA', 'Barrels', 'Barrel%', 'RV', 'RV/100'
        ]

    # Filter to only include columns that exist
    existing_cols = [col for col in ordered_cols if col in stats.columns]
    return stats[existing_cols].round(3)

def create_filters(df, prefix=""):
    """Create dynamic filters for all columns"""
    filters = {}
    
    # Create columns for filters
    num_cols = len(df.columns)
    cols_per_row = 4
    
    for i in range(0, num_cols, cols_per_row):
        cols = st.columns(min(cols_per_row, num_cols - i))
        
        for j, col_name in enumerate(df.columns[i:i+cols_per_row]):
            with cols[j]:
                if df[col_name].dtype in ['object', 'string']:
                    # Text/categorical filters
                    unique_values = ['All'] + sorted(df[col_name].dropna().unique().tolist())
                    selected = st.selectbox(
                        f"{col_name}",
                        unique_values,
                        key=f"{prefix}_{col_name}_filter"
                    )
                    if selected != 'All':
                        filters[col_name] = selected
                        
                elif df[col_name].dtype in ['int64', 'float64']:
                    # Numeric filters
                    if not df[col_name].isna().all():
                        min_val = float(df[col_name].min())
                        max_val = float(df[col_name].max())
                        
                        if min_val != max_val:
                            selected_range = st.slider(
                                f"{col_name}",
                                min_value=min_val,
                                max_value=max_val,
                                value=(min_val, max_val),
                                key=f"{prefix}_{col_name}_slider"
                            )
                            if selected_range != (min_val, max_val):
                                filters[col_name] = selected_range
    
    return filters

def apply_filters(df, filters):
    """Apply the selected filters to the dataframe"""
    filtered_df = df.copy()
    
    for col_name, filter_value in filters.items():
        if isinstance(filter_value, tuple):  # Numeric range
            filtered_df = filtered_df[
                (filtered_df[col_name] >= filter_value[0]) & 
                (filtered_df[col_name] <= filter_value[1])
            ]
        else:  # Categorical
            filtered_df = filtered_df[filtered_df[col_name] == filter_value]
    
    return filtered_df

def create_table_display(df, title, subtitle, prefix=""):
    """Create the styled table with filters"""
    st.markdown(f'<div class="main-header">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{subtitle}</div>', unsafe_allow_html=True)
    
    # Filters section
    with st.expander("🔍 Filters", expanded=False):
        filters = create_filters(df, prefix)
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    # Controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        entries_options = [25, 50, 100, 200, 500]
        show_entries = st.selectbox("Show entries:", entries_options, index=3, key=f"{prefix}_entries")
    
    with col2:
        search_term = st.text_input("Search:", key=f"{prefix}_search", placeholder="Search players...")
    
    # Apply search filter
    if search_term:
        mask = filtered_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
        filtered_df = filtered_df[mask]
    
    # Show results count
    st.write(f"Showing {min(show_entries, len(filtered_df))} of {len(filtered_df)} entries")
    
    # Display limited results
    display_df = filtered_df.head(show_entries)
    
    # Add ranking
    display_df = display_df.reset_index(drop=True)
    display_df.index = display_df.index + 1
    
    # Display table
    st.dataframe(
        display_df,
        use_container_width=True,
        height=600
    )
    
    return display_df

def main():
    # Sidebar
    st.sidebar.title("⚾ CCBL Stats Dashboard")
    
    # Google Drive URL input
    drive_url = st.sidebar.text_input(
        "Google Drive CSV URL:",
        placeholder="https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing",
        help="Make sure your CSV is set to 'Anyone with the link can view'"
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5 min)", value=True)
    
    if auto_refresh:
        st.sidebar.write("⏱️ Data refreshes every 5 minutes")
    
    if drive_url:
        # Load data
        with st.spinner("Loading data from Google Drive..."):
            raw_df = load_csv_from_drive(drive_url)
        
        if raw_df is not None:
            # Create tabs
            tab1, tab2 = st.tabs(["📊 Overall Stats", "🔄 By Handedness"])
            
            with tab1:
                # Process overall stats
                try:
                    overall_stats = load_and_process_csv(raw_df, split_by_handedness=False)
                    create_table_display(
                        overall_stats,
                        "CCBL Live Hitting Leaders",
                        "Cape Cod Baseball League Statistics",
                        "overall"
                    )
                except Exception as e:
                    st.error(f"Error processing overall stats: {str(e)}")
            
            with tab2:
                # Process handedness splits
                try:
                    handed_stats = load_and_process_csv(raw_df, split_by_handedness=True)
                    create_table_display(
                        handed_stats,
                        "CCBL Live Hitting Leaders - By Handedness",
                        "Split by Pitcher Handedness",
                        "handed"
                    )
                except Exception as e:
                    st.error(f"Error processing handedness stats: {str(e)}")
            
            # Sidebar stats
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Data Summary:**")
            st.sidebar.write(f"Raw Records: {len(raw_df):,}")
            st.sidebar.write(f"Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
            
    else:
        st.info("👈 Please enter your Google Drive CSV URL in the sidebar to get started.")
        
        # Instructions
        st.markdown("""
        ## How to get your Google Drive CSV link:
        
        1. Upload your CSV file to Google Drive
        2. Right-click on the file and select "Share"
        3. Change permissions to "Anyone with the link can view"
        4. Copy the shareable link
        5. Paste it in the sidebar
        
        The link should look like:
        `https://drive.google.com/file/d/1ABC123.../view?usp=sharing`
        """)

if __name__ == "__main__":
    main()
