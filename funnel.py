import pandas as pd

def analyze_user_journeys(df_ga4, df_postback):
    """
    Analyze user journeys with multiple attribution models
    """
    
    print("=== COMPREHENSIVE USER JOURNEY ANALYSIS ===")
    
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
    
    # Method 1: Content Participation Attribution
    print("\n=== METHOD 1: CONTENT PARTICIPATION (Any Interaction) ===")
    participation_results = []
    
    for category in df_ga4['section_category'].unique():
        # Users who interacted with this category at any point
        participated = user_journeys[
            user_journeys['section_category'].apply(lambda x: category in x)
        ]
        
        # Users who clicked after interacting with this category
        clicked_after = participated[
            participated.apply(lambda row: any(
                event == 'click_on_bk_web' and 
                i > 0 and 
                category in row['section_category'][:i+1]
                for i, event in enumerate(row['event_name'])
            ), axis=1)
        ]
        
        participation_results.append({
            'content_category': category,
            'total_participants': len(participated),
            'participants_who_clicked': len(clicked_after),
            'participants_who_registered': participated['has_registration'].sum(),
            'participants_who_ftd': participated['has_ftd'].sum(),
            'participation_click_rate': len(clicked_after) / len(participated) if len(participated) > 0 else 0,
            'participation_ftd_rate': participated['has_ftd'].sum() / len(participated) if len(participated) > 0 else 0
        })
    
    participation_df = pd.DataFrame(participation_results).sort_values('participants_who_ftd', ascending=False)
    print(participation_df.round(4))
    
    return participation_df


def page_level_analysis(df_ga4):
    """
    Analyze specific page performance using page URLs/titles
    Shows top 10 pages by both FTD count and conversion rate
    """
    
    print("\n=== PAGE-LEVEL CONVERSION ANALYSIS ===")
    
    # Use page_location if available, otherwise page_title  
    page_column = 'event_params_page_location'
    
    # Get page performance metrics
    page_performance = df_ga4.groupby([page_column, 'user_segment']).agg({
        'user_pseudo_id': 'nunique',
        'event_name': lambda x: (x == 'click_on_bk_web').sum(),  # Clicks on this page
        'event_params_engagement_time_msec': 'mean'  # Average engagement per visit
    }).reset_index()
    
    page_performance.columns = ['page', 'user_segment', 'unique_users', 'clicks', 'avg_engagement_ms']
    
    # Pivot to get conversion metrics
    page_conversion = page_performance.pivot(index='page', columns='user_segment', values='unique_users').fillna(0)
    page_clicks = page_performance.pivot(index='page', columns='user_segment', values='clicks').fillna(0)
    
    # Calculate conversion rates
    if 'ftd_converted' in page_conversion.columns:
        page_conversion['total_users'] = page_conversion.sum(axis=1)
        page_conversion['ftd_count'] = page_conversion['ftd_converted']  # Absolute FTD count
        page_conversion['ftd_rate'] = page_conversion['ftd_converted'] / page_conversion['total_users']
        page_conversion['total_clicks'] = page_clicks.sum(axis=1)  
        page_conversion['click_rate'] = page_conversion['total_clicks'] / page_conversion['total_users']
        
        # 1. TOP 10 PAGES BY ABSOLUTE FTD COUNT
        print("🥇 TOP 10 PAGES BY FTD COUNT:")
        print("-" * 50)
        top_ftd_count = page_conversion[
            page_conversion['total_users'] >= 50  # Minimum threshold
        ].sort_values('ftd_count', ascending=False).head(10)
        
        for i, (page, row) in enumerate(top_ftd_count.iterrows(), 1):
            page_short = str(page)[:60] + '...' if len(str(page)) > 60 else str(page)
            print(f"{i:2d}. {page_short}")
            print(f"     FTDs: {int(row['ftd_count'])} | Users: {int(row['total_users'])} | FTD Rate: {row['ftd_rate']:.2%}")
            print()
        
        # 2. TOP 10 PAGES BY FTD RATE (CONVERSION RATE)
        print("\n🎯 TOP 10 PAGES BY FTD RATE (Conversion Rate):")
        print("-" * 50)
        top_converting_pages = page_conversion[
            page_conversion['total_users'] >= 100  # Higher threshold for rate analysis
        ].sort_values('ftd_rate', ascending=False).head(10)
        
        for i, (page, row) in enumerate(top_converting_pages.iterrows(), 1):
            page_short = str(page)[:60] + '...' if len(str(page)) > 60 else str(page)
            print(f"{i:2d}. {page_short}")
            print(f"     FTD Rate: {row['ftd_rate']:.2%} | FTDs: {int(row['ftd_converted'])} | Users: {int(row['total_users'])}")
            print()
        
        # 3. KEY INSIGHTS
        print("\n🔍 KEY INSIGHTS:")
        print("-" * 30)
        total_ftds = page_conversion['ftd_count'].sum()
        median_rate = page_conversion['ftd_rate'].median()
        
        print(f"• Total FTDs across all pages: {int(total_ftds)}")
        print(f"• Median FTD rate: {median_rate:.2%}")
        print(f"• Top 10 pages by count represent: {(top_ftd_count['ftd_count'].sum() / total_ftds):.1%} of all FTDs")
        print(f"• Highest converting page: {top_converting_pages.iloc[0]['ftd_rate']:.2%} FTD rate")
        print(f"• Biggest FTD generator: {int(top_ftd_count.iloc[0]['ftd_count'])} FTDs")
    
    return page_conversion
