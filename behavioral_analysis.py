import pandas as pd

# Complete Behavioral Analysis

def behavioral_analysis_gen(df_ga4, df_postback):
    """
    Behavioral analysis using the actual column names from the datasets
    """
    
    print("=== BEHAVIORAL FACTORS AND USER PATHS ANALYSIS ===")
    
    # Display user segments
    segment_counts = df_ga4.groupby('user_segment')['user_pseudo_id'].nunique()
    print("User segments:")
    for segment, count in segment_counts.items():
        print(f"  {segment}: {count:,}")
    
    return df_ga4

def session_analysis(df_ga4):
    """
    Session analysis using actual column names
    """
    
    print("\n=== SESSION ANALYSIS ===")
    
    # Calculate session metrics per user
    user_sessions = df_ga4.groupby(['user_pseudo_id', 'user_segment']).agg({
        'event_name': 'count',  # Total events
        'event_timestamp': ['min', 'max'],  # Session start/end
        'section_category': lambda x: len(x.unique()),  # Content variety
        'event_params_engagement_time_msec': 'sum'  # Total engagement time
    }).reset_index()
    
    # Fix column names
    user_sessions.columns = ['user_pseudo_id', 'user_segment', 'total_events', 
                           'session_start', 'session_end', 'content_variety', 
                           'total_engagement_ms']
    
    # Calculate session duration in minutes
    user_sessions['session_duration_minutes'] = (
        (user_sessions['session_end'] - user_sessions['session_start']) / 1000000 / 60
    )
    
    # Convert engagement time to minutes
    user_sessions['engagement_time_minutes'] = user_sessions['total_engagement_ms'].fillna(0) / 1000 / 60
    
    # Get page views specifically
    pageviews_per_user = df_ga4[df_ga4['event_name'] == 'page_view'].groupby(['user_pseudo_id', 'user_segment']).size().reset_index()
    pageviews_per_user.columns = ['user_pseudo_id', 'user_segment', 'page_views']
    
    # Merge
    user_sessions = user_sessions.merge(pageviews_per_user, on=['user_pseudo_id', 'user_segment'], how='left')
    user_sessions['page_views'] = user_sessions['page_views'].fillna(0)
    
    # Get clicks
    clicks_per_user = df_ga4[df_ga4['event_name'] == 'click_on_bk_web'].groupby(['user_pseudo_id', 'user_segment']).size().reset_index()
    clicks_per_user.columns = ['user_pseudo_id', 'user_segment', 'clicks']
    user_sessions = user_sessions.merge(clicks_per_user, on=['user_pseudo_id', 'user_segment'], how='left')
    user_sessions['clicks'] = user_sessions['clicks'].fillna(0)
    
    # Compare metrics by segment
    session_comparison = user_sessions.groupby('user_segment').agg({
        'page_views': ['mean', 'median', 'std'],
        'session_duration_minutes': ['mean', 'median', 'std'],
        'content_variety': ['mean', 'median', 'std'],
        'engagement_time_minutes': ['mean', 'median', 'std'],
        'clicks': ['mean', 'median', 'std']
    }).round(2)
    
    print("Session Metrics Comparison:")
    print(session_comparison)

    return user_sessions

def time_to_key_action_analysis(df_ga4, df_postback):
    """
    MISSING PIECE: Analyze time on site before key actions (registration, FTD)
    """
    
    print("\n=== TIME TO KEY ACTION ANALYSIS ===")
    
    # Get user session timing
    user_timing = df_ga4.groupby(['user_pseudo_id', 'user_segment']).agg({
        'event_timestamp': ['min', 'max'],  # First and last event
        'user_first_touch_timestamp': 'first',
        'event_date': ['min', 'max']  # First and last date
    }).reset_index()
    
    # Flatten column names
    user_timing.columns = ['user_pseudo_id', 'user_segment', 'session_start', 'session_end', 
                          'first_touch_timestamp', 'first_date', 'last_date']
    
    # Convert to datetime
    user_timing['session_start_dt'] = pd.to_datetime(user_timing['session_start'], unit='us')
    user_timing['session_end_dt'] = pd.to_datetime(user_timing['session_end'], unit='us')
    user_timing['first_touch_dt'] = pd.to_datetime(user_timing['first_touch_timestamp'], unit='us')
    
    # Calculate session duration
    user_timing['session_duration_minutes'] = (
        (user_timing['session_end_dt'] - user_timing['session_start_dt']).dt.total_seconds() / 60
    )
    
    # Get conversion timing from postback data
    conversion_timing = df_postback.groupby('meta_user_id').agg({
        'et': lambda x: x.tolist(),  # All event types
        'date': lambda x: x.tolist()  # All action dates
    }).reset_index()
    
    # Find registration and FTD timing
    def extract_action_date(events, dates, action_type):
        try:
            if action_type in events:
                idx = events.index(action_type)
                return pd.to_datetime(dates[idx])
        except:
            pass
        return None
    
    conversion_timing['reg_date'] = conversion_timing.apply(
        lambda row: extract_action_date(row['et'], row['date'], 'reg'), axis=1
    )
    conversion_timing['ftd_date'] = conversion_timing.apply(
        lambda row: extract_action_date(row['et'], row['date'], 'ftd'), axis=1
    )
    
    # Merge timing data
    timing_analysis = user_timing.merge(
        conversion_timing[['meta_user_id', 'reg_date', 'ftd_date']], 
        left_on='user_pseudo_id', right_on='meta_user_id', how='left'
    )
    
    # Calculate time to key actions
    timing_analysis['time_to_registration_hours'] = (
        (timing_analysis['reg_date'] - timing_analysis['session_start_dt']).dt.total_seconds() / 3600
    )
    timing_analysis['time_to_ftd_hours'] = (
        (timing_analysis['ftd_date'] - timing_analysis['session_start_dt']).dt.total_seconds() / 3600
    )
    
    # Days from first touch to conversion
    timing_analysis['days_first_touch_to_ftd'] = (
        (timing_analysis['ftd_date'] - timing_analysis['first_touch_dt']).dt.days
    )
    timing_analysis['days_first_touch_to_reg'] = (
        (timing_analysis['reg_date'] - timing_analysis['first_touch_dt']).dt.days
    )
    
    # Analysis by user segment
    print("⏱️ TIME TO KEY ACTIONS COMPARISON:")
    
    # Registration timing
    reg_users = timing_analysis[timing_analysis['time_to_registration_hours'].notna()]
    if len(reg_users) > 0:
        print("\n📝 REGISTRATION TIMING:")
        print(f"  Total users who registered: {len(reg_users)}")
        print(f"  Average time to registration: {reg_users['time_to_registration_hours'].mean():.1f} hours")
        print(f"  Median time to registration: {reg_users['time_to_registration_hours'].median():.1f} hours")
        
        # Quick vs slow registrations
        quick_reg = reg_users[reg_users['time_to_registration_hours'] <= 1]  # Within 1 hour
        slow_reg = reg_users[reg_users['time_to_registration_hours'] > 24]   # After 24 hours
        
        print(f"  Quick registrations (≤1h): {len(quick_reg)} ({len(quick_reg)/len(reg_users):.1%})")
        print(f"  Slow registrations (>24h): {len(slow_reg)} ({len(slow_reg)/len(reg_users):.1%})")
    
    # FTD timing
    ftd_users = timing_analysis[timing_analysis['time_to_ftd_hours'].notna()]
    if len(ftd_users) > 0:
        print("\n💰 FTD TIMING:")
        print(f"  Total users who made FTD: {len(ftd_users)}")
        print(f"  Average time to FTD: {ftd_users['time_to_ftd_hours'].mean():.1f} hours")
        print(f"  Median time to FTD: {ftd_users['time_to_ftd_hours'].median():.1f} hours")
        
        # Quick vs slow FTDs
        quick_ftd = ftd_users[ftd_users['time_to_ftd_hours'] <= 2]   # Within 2 hours
        slow_ftd = ftd_users[ftd_users['time_to_ftd_hours'] > 48]    # After 48 hours
        
        print(f"  Quick FTDs (≤2h): {len(quick_ftd)} ({len(quick_ftd)/len(ftd_users):.1%})")
        print(f"  Slow FTDs (>48h): {len(slow_ftd)} ({len(slow_ftd)/len(ftd_users):.1%})")
        
        # Days from first touch to FTD
        ftd_first_touch = ftd_users[ftd_users['days_first_touch_to_ftd'].notna()]
        if len(ftd_first_touch) > 0:
            print("\n🎯 FIRST TOUCH TO FTD:")
            print(f"  Average days from first touch to FTD: {ftd_first_touch['days_first_touch_to_ftd'].mean():.1f} days")
            print(f"  Median days from first touch to FTD: {ftd_first_touch['days_first_touch_to_ftd'].median():.1f} days")
            
            same_day_ftd = ftd_first_touch[ftd_first_touch['days_first_touch_to_ftd'] == 0]
            print(f"  Same-day conversions: {len(same_day_ftd)} ({len(same_day_ftd)/len(ftd_first_touch):.1%})")
    
    # Session duration comparison by conversion status
    print("\n⏰ SESSION DURATION BY CONVERSION STATUS:")
    session_comparison = timing_analysis.groupby('user_segment')['session_duration_minutes'].agg(['count', 'mean', 'median', 'std']).round(2)
    print(session_comparison)
    
    return timing_analysis

def traffic_source_analysis(df_ga4):
    """
    Analyze traffic sources - since source is fixed per session, focus on quality metrics
    """
    
    print("\n=== TRAFFIC SOURCE ANALYSIS ===")
    
    # Traffic source performance analysis
    traffic_analysis = df_ga4.groupby(['user_pseudo_id', 'user_segment']).agg({
        'traffic_source_source': 'first',  # Traffic source (same for whole session)
        'session_traffic_source_last_click_cross_channel_campaign_medium': 'first',  # Medium
        'event_name': 'count',  # Total events per user
        'event_params_engagement_time_msec': 'sum',  # Total engagement
        'section_category': lambda x: len(x.unique())  # Content variety
    }).reset_index()
    
    traffic_analysis.columns = ['user_pseudo_id', 'user_segment', 'traffic_source', 'campaign_medium', 
                               'total_events', 'total_engagement_ms', 'content_variety']
    
    # Add page views and clicks
    pageviews = df_ga4[df_ga4['event_name'] == 'page_view'].groupby('user_pseudo_id').size()
    clicks = df_ga4[df_ga4['event_name'] == 'click_on_bk_web'].groupby('user_pseudo_id').size()
    
    traffic_analysis = traffic_analysis.merge(pageviews.rename('page_views'), left_on='user_pseudo_id', right_index=True, how='left')
    traffic_analysis = traffic_analysis.merge(clicks.rename('clicks'), left_on='user_pseudo_id', right_index=True, how='left')
    traffic_analysis[['page_views', 'clicks']] = traffic_analysis[['page_views', 'clicks']].fillna(0)
    
    # Calculate engagement metrics
    traffic_analysis['engagement_time_sec'] = traffic_analysis['total_engagement_ms'].fillna(0) / 1000
    traffic_analysis['click_rate'] = traffic_analysis['clicks'] / traffic_analysis['page_views'].replace(0, 1)
    
    # 1. TRAFFIC SOURCE QUALITY ANALYSIS
    print("\n🎯 TRAFFIC SOURCE QUALITY ANALYSIS:")
    source_quality = traffic_analysis.groupby(['traffic_source', 'user_segment']).agg({
        'user_pseudo_id': 'count',
        'page_views': 'mean',
        'engagement_time_sec': 'mean', 
        'content_variety': 'mean',
        'click_rate': 'mean'
    }).reset_index()
    
    source_summary = source_quality.pivot(index='traffic_source', columns='user_segment', values='user_pseudo_id').fillna(0)
    if 'ftd_converted' in source_summary.columns:
        source_summary['total_users'] = source_summary.sum(axis=1)
        source_summary['ftd_rate'] = source_summary['ftd_converted'] / source_summary['total_users']
        source_summary = source_summary.sort_values('ftd_converted', ascending=False)
        
        print("Top Traffic Sources by Performance:")
        for source, row in source_summary.head(15).iterrows():
            print(f"  {source}: {int(row['ftd_converted'])} FTDs, {row['ftd_rate']:.2%} rate ({int(row['total_users'])} users)")
    
    # 2. CAMPAIGN MEDIUM ANALYSIS
    print("\n📱 CAMPAIGN MEDIUM ANALYSIS:")
    medium_quality = traffic_analysis.groupby(['campaign_medium', 'user_segment']).agg({
        'user_pseudo_id': 'count',
        'page_views': 'mean',
        'engagement_time_sec': 'mean'
    }).reset_index()
    
    medium_summary = medium_quality.pivot(index='campaign_medium', columns='user_segment', values='user_pseudo_id').fillna(0)
    if 'ftd_converted' in medium_summary.columns:
        medium_summary['total_users'] = medium_summary.sum(axis=1)
        medium_summary['ftd_rate'] = medium_summary['ftd_converted'] / medium_summary['total_users']
        medium_summary = medium_summary.sort_values('ftd_converted', ascending=False)
        
        print("Top Campaign Mediums by Performance:")
        for medium, row in medium_summary.head(10).iterrows():
            print(f"  {medium}: {int(row['ftd_converted'])} FTDs, {row['ftd_rate']:.2%} rate ({int(row['total_users'])} users)")
    
    # 3. SOURCE + CONTENT INTERACTION
    print("\n🔗 TRAFFIC SOURCE + CONTENT INTERACTION:")
    
    # Get most visited content category per user
    user_main_content = df_ga4.groupby('user_pseudo_id')['section_category'].apply(lambda x: x.value_counts().index[0]).reset_index()
    user_main_content.columns = ['user_pseudo_id', 'main_content_category']
    
    # Merge with traffic data
    traffic_content = traffic_analysis.merge(user_main_content, on='user_pseudo_id', how='left')
    
    # Analyze which traffic sources drive users to which content
    source_content_conv = traffic_content.groupby(['traffic_source', 'main_content_category', 'user_segment']).size().unstack(fill_value=0)
    
    if 'ftd_converted' in source_content_conv.columns:
        source_content_conv['total'] = source_content_conv.sum(axis=1)
        source_content_conv['ftd_rate'] = source_content_conv['ftd_converted'] / source_content_conv['total']
        
        # Show top combinations
        top_combinations = source_content_conv[source_content_conv['total'] >= 10].sort_values('ftd_converted', ascending=False).head(15)
        
        print("Top Traffic Source + Content Combinations:")
        for (source, content), row in top_combinations.iterrows():
            print(f"  {source} → {content}: {int(row['ftd_converted'])} FTDs, {row['ftd_rate']:.2%} rate ({int(row['total'])} users)")
    
    return source_summary, medium_summary, traffic_content

def compress_path_sequence(path_list):
    """
    Compress repeated content into readable format like 'bonuses*5 → predictions*2'
    """
    if not path_list or len(path_list) == 0:
        return ""
    
    compressed = []
    current_content = path_list[0]
    count = 1
    
    for i in range(1, len(path_list)):
        if path_list[i] == current_content:
            count += 1
        else:
            # Add the compressed segment
            if count == 1:
                compressed.append(current_content)
            else:
                compressed.append(f"{current_content}({count})")
            
            # Start new segment
            current_content = path_list[i]
            count = 1
    
    # Add the last segment
    if count == 1:
        compressed.append(current_content)
    else:
        compressed.append(f"{current_content}({count})")
    
    return " → ".join(compressed)

def path_analysis(df_ga4):
    """
    Path analysis with improved readable formatting
    """
    
    print("\n=== PATH ANALYSIS ===")
    
    # Create user journeys
    page_column = 'event_params_page_location' if 'event_params_page_location' in df_ga4.columns else 'event_params_page_title'
    
    user_paths = df_ga4.sort_values(['user_pseudo_id', 'event_timestamp']).groupby(['user_pseudo_id', 'user_segment']).agg({
        'section_category': list,
        'event_name': list,
        page_column: list
    }).reset_index()
    
    # Create COMPRESSED path sequences (much more readable)
    user_paths['path_sequence'] = user_paths['section_category'].apply(
        lambda x: compress_path_sequence(x[:8]) if len(x) > 0 else ""  # Take first 8 steps, compress
    )
    
    user_paths['has_click'] = user_paths['event_name'].apply(
        lambda x: 'click_on_bk_web' in x
    )
    
    # Filter out empty paths
    user_paths = user_paths[user_paths['path_sequence'] != ""]
    
    # Hot Paths Analysis
    print("\n🔥 HOT PATHS (Leading to FTD):")
    if 'ftd_converted' in user_paths['user_segment'].values:
        hot_paths = user_paths[user_paths['user_segment'] == 'ftd_converted']['path_sequence'].value_counts().head(15)
        
        for i, (path, count) in enumerate(hot_paths.items(), 1):
            print(f"  {i:2d}. {path} ({count} users)")
    
    # Dead-end Paths Analysis
    print("\n💀 DEAD-END PATHS (High volume, no conversion):")
    
    all_paths = user_paths['path_sequence'].value_counts()
    path_conversion = user_paths.groupby('path_sequence')['user_segment'].apply(
        lambda x: (x == 'ftd_converted').sum() / len(x) if len(x) > 0 else 0
    )
    
    dead_end_paths = all_paths[
        (all_paths >= 50) & 
        (path_conversion == 0)
    ].head(15)
    
    for i, (path, count) in enumerate(dead_end_paths.items(), 1):
        print(f"  {i:2d}. {path} ({count:,} users, 0% conversion)")
    
    # Path effectiveness analysis
    print("\n📊 PATH EFFECTIVENESS:")
    path_effectiveness = user_paths.groupby('path_sequence').agg({
        'user_pseudo_id': 'count',
        'user_segment': lambda x: (x == 'ftd_converted').sum(),
        'has_click': 'sum'
    }).reset_index()
    
    path_effectiveness.columns = ['path_sequence', 'total_users', 'ftd_users', 'users_with_clicks']
    path_effectiveness['conversion_rate'] = path_effectiveness['ftd_users'] / path_effectiveness['total_users']
    path_effectiveness['click_rate'] = path_effectiveness['users_with_clicks'] / path_effectiveness['total_users']
    
    # Top converting paths (min 10 users)
    top_converting = path_effectiveness[
        path_effectiveness['total_users'] >= 10
    ].sort_values('conversion_rate', ascending=False).head(15)
    
    print("🎯 Top Converting Path Patterns:")
    for i, (_, row) in enumerate(top_converting.iterrows(), 1):
        print(f"  {i:2d}. {row['path_sequence']} → {row['conversion_rate']:.1%} conversion ({row['total_users']} users)")
    
    # Content transition analysis
    print("\n🔄 CONTENT TRANSITION PATTERNS:")
    
    # Find paths with meaningful transitions (not just repetition)
    transition_paths = path_effectiveness[
        (path_effectiveness['total_users'] >= 5) &
        (path_effectiveness['path_sequence'].str.contains('→'))  # Has transitions
    ].sort_values('ftd_users', ascending=False).head(10)
    
    print("Most successful content transitions:")
    for i, (_, row) in enumerate(transition_paths.iterrows(), 1):
        print(f"  {i:2d}. {row['path_sequence']} → {row['ftd_users']} FTDs ({row['conversion_rate']:.1%})")
    
    return user_paths, path_effectiveness

def estimate_scroll_depth_analysis(df_ga4):
    """
    Estimate scroll depth and page engagement using available metrics
    """
    
    print("\n=== SCROLL DEPTH ESTIMATION (Key Pages) ===")
    
    # Use page_location if available, otherwise page_title
    page_column = 'event_params_page_location' if 'event_params_page_location' in df_ga4.columns else 'event_params_page_title'
    
    # Focus on key page types for scroll analysis
    key_page_types = ['bonuses', 'bk_rating', 'predictions', 'news', 'blog']
    
    scroll_estimates = []
    
    for page_type in key_page_types:
        page_data = df_ga4[df_ga4['section_category'] == page_type].copy()
        
        if len(page_data) == 0:
            continue
            
        # Calculate page-level engagement metrics
        page_engagement = page_data.groupby(['user_pseudo_id', 'user_segment', page_column]).agg({
            'event_params_engagement_time_msec': 'sum',  # Total time on page
            'event_name': 'count',  # Number of events on page
            'event_timestamp': ['min', 'max']  # Time span on page
        }).reset_index()
        
        # Flatten column names
        page_engagement.columns = ['user_pseudo_id', 'user_segment', 'page', 'engagement_time_ms', 
                                 'event_count', 'first_event', 'last_event']
        
        # Calculate time spent (in seconds) - fix data type issue
        page_engagement['time_on_page_sec'] = (
            (page_engagement['last_event'] - page_engagement['first_event']) / pd.Timedelta(seconds=1)
        ).astype(float)
        page_engagement['engagement_time_sec'] = page_engagement['engagement_time_ms'].fillna(0) / 1000
        
        # Estimate scroll engagement (higher engagement time + more events = more scrolling)
        page_engagement['estimated_scroll_score'] = (
            (page_engagement['engagement_time_sec'] / 10) +  # Normalize engagement time
            (page_engagement['event_count'] / 2) +  # More events = more interaction
            (page_engagement['time_on_page_sec'] / 30)  # Time spent
        )
        
        # Classify scroll depth (rough estimation)
        page_engagement['estimated_scroll_depth'] = pd.cut(
            page_engagement['estimated_scroll_score'], 
            bins=[0, 1, 3, 6, float('inf')], 
            labels=['Low (<25%)', 'Medium (25-50%)', 'High (50-75%)', 'Very High (75%+)']
        )
        
        # Analyze by user segment
        scroll_by_segment = page_engagement.groupby(['user_segment', 'estimated_scroll_depth']).size().unstack(fill_value=0)
        scroll_by_segment_pct = scroll_by_segment.div(scroll_by_segment.sum(axis=1), axis=0) * 100
        
        print(f"\n📜 {page_type.upper()} Pages - Estimated Scroll Engagement:")
        print("Scroll Depth Distribution by User Segment (%):")
        print(scroll_by_segment_pct.round(1))
        
        # Average engagement metrics
        avg_metrics = page_engagement.groupby('user_segment').agg({
            'engagement_time_sec': 'mean',
            'event_count': 'mean',
            'estimated_scroll_score': 'mean'
        }).round(2)
        
        print("\nAverage Engagement Metrics:")
        print(avg_metrics)
        
        scroll_estimates.append({
            'page_type': page_type,
            'scroll_distribution': scroll_by_segment_pct,
            'avg_metrics': avg_metrics,
            'raw_data': page_engagement
        })
    
    return scroll_estimates

def new_vs_returning_analysis(df_ga4):
    """
    Analyze new vs returning users using user_first_touch_timestamp
    """
    
    print("\n=== NEW VS RETURNING USER ANALYSIS ===")
    
    # Convert timestamps
    df_ga4['event_datetime'] = pd.to_datetime(df_ga4['event_timestamp'], unit='us')
    df_ga4['first_touch_datetime'] = pd.to_datetime(df_ga4['user_first_touch_timestamp'], unit='us')
    
    # Calculate time since first touch for each event
    df_ga4['days_since_first_touch'] = (df_ga4['event_datetime'] - df_ga4['first_touch_datetime']).dt.days
    
    # Classify users: if their current events are on the same day as first touch = new, otherwise = returning
    user_classification = df_ga4.groupby(['user_pseudo_id', 'user_segment']).agg({
        'days_since_first_touch': 'min',  # Minimum days since first touch
        'event_date': lambda x: len(x.unique())  # Number of different days active
    }).reset_index()
    
    user_classification['user_type'] = user_classification.apply(lambda row:
        'new' if row['days_since_first_touch'] == 0 and row['event_date'] == 1 else 'returning', axis=1
    )
    
    # Analyze conversion by user type
    user_type_analysis = user_classification.groupby(['user_type', 'user_segment']).size().unstack(fill_value=0)
    user_type_analysis['total'] = user_type_analysis.sum(axis=1)
    
    if 'ftd_converted' in user_type_analysis.columns:
        user_type_analysis['ftd_rate'] = user_type_analysis['ftd_converted'] / user_type_analysis['total']
    
    print("User Type Performance:")
    print(user_type_analysis.round(4))
    
    return user_classification

def complete_attribution_analysis(df_ga4, df_postback):
    """
    FIXED: Complete attribution analysis with all methods
    """
    
    print("\n=== COMPLETE ATTRIBUTION ANALYSIS ===")
    
    # Get user journey data
    user_journeys = df_ga4.sort_values(['user_pseudo_id', 'event_timestamp']) \
                          .groupby('user_pseudo_id').agg({
        'section_category': list,
        'event_name': list,
        'event_timestamp': list
    }).reset_index()
    
    # Get conversion data
    conversions = df_postback.groupby('meta_user_id').agg(
        has_registration=('et', lambda x: 'reg' in x.values),
        has_ftd=('et', lambda x: 'ftd' in x.values)
    ).reset_index()
    
    # Merge with conversions
    user_journeys = user_journeys.merge(
        conversions,
        left_on='user_pseudo_id',
        right_on='meta_user_id',
        how='left'
    ).fillna(False)

    # Journey Position Attribution
    print("\n=== JOURNEY POSITION ATTRIBUTION ===")
    position_f_results = []
    position_l_results = []
    
    for category in df_ga4['section_category'].unique():
        # Users who started with this category (first-touch)
        first_touch = user_journeys[
            user_journeys['section_category'].apply(lambda x: x[0] == category if len(x) > 0 else False)
        ]
        
        # Users who ended with this category (last-touch)
        last_touch = user_journeys[
            user_journeys['section_category'].apply(lambda x: x[-1] == category if len(x) > 0 else False)
        ]
        
        position_f_results.append({
            'content_category': category,
            'first_touch_users': len(first_touch),
            'first_touch_ftds': first_touch['has_ftd'].sum(),
            'first_touch_ftd_rate': first_touch['has_ftd'].sum() / len(first_touch) if len(first_touch) > 0 else 0
        })
        
        position_l_results.append({
            'content_category': category,
            'last_touch_users': len(last_touch),
            'last_touch_ftds': last_touch['has_ftd'].sum(),
            'last_touch_ftd_rate': last_touch['has_ftd'].sum() / len(last_touch) if len(last_touch) > 0 else 0
        })

    position_f_df = pd.DataFrame(position_f_results).sort_values('first_touch_ftds', ascending=False)
    position_l_df = pd.DataFrame(position_l_results).sort_values('last_touch_ftds', ascending=False)
    
    print("\n🎯 FIRST-TOUCH ATTRIBUTION (Best at attracting users):")
    print(position_f_df.head(10).round(4))
    
    print("\n🏁 LAST-TOUCH ATTRIBUTION (Best at converting users):")  
    print(position_l_df.head(10).round(4))
    
    return position_f_df, position_l_df

