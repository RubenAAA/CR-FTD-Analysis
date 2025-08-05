import pandas as pd
import plotly.graph_objects as go

# Run the visualizations
def create_sankey_flow_diagram(df_ga4, df_postback):
    """
    Create a MUCH Sankey diagram showing meaningful user flows
    """
    
    print("🌊 Creating Sankey Flow Diagram...")
    
    # Get user journeys with conversion status
    user_journeys = df_ga4.sort_values(['user_pseudo_id', 'event_timestamp']) \
                          .groupby('user_pseudo_id').agg({
        'section_category': list,
        'event_name': list
    }).reset_index()
    
    # Get conversion data
    conversions = df_postback.groupby('meta_user_id').agg(
        has_ftd=('et', lambda x: 'ftd' in x.values)
    ).reset_index()
    
    user_journeys = user_journeys.merge(
        conversions, left_on='user_pseudo_id', right_on='meta_user_id', how='left'
    ).fillna(False)
    
    # Create SIMPLIFIED flows: Entry Point → Most Visited Category → Outcome
    flows = []
    
    for _, user in user_journeys.iterrows():
        sequence = user['section_category']
        converted = user['has_ftd']
        
        if isinstance(sequence, list) and len(sequence) >= 1:
            # Entry point (first category)
            entry_point = f"ENTRY: {sequence[0]}"
            
            # Most frequent category (shows user interest)
            if len(sequence) > 1:
                category_counts = pd.Series(sequence).value_counts()
                main_category = f"MAIN: {category_counts.index[0]}"
                
                # Don't create flow if entry and main are the same
                if sequence[0] != category_counts.index[0]:
                    flows.append({
                        'source': entry_point,
                        'target': main_category,
                        'converted': converted,
                        'user_id': user['user_pseudo_id'],
                        'session_length': len(sequence)
                    })
                else:
                    main_category = entry_point
            else:
                main_category = entry_point
            
            # Final outcome
            if converted:
                outcome = "✅ FTD CONVERTED"
            else:
                outcome = "❌ NO CONVERSION"
            
            flows.append({
                'source': main_category,
                'target': outcome,
                'converted': converted,
                'user_id': user['user_pseudo_id'],
                'session_length': len(sequence)
            })
    
    if not flows:
        print("❌ No flow data available")
        return None
    
    flows_df = pd.DataFrame(flows)
    
    # Aggregate flows and filter significant ones
    flow_summary = flows_df.groupby(['source', 'target']).agg({
        'user_id': 'count',
        'converted': 'sum',
        'session_length': 'mean'
    }).reset_index()
    flow_summary.columns = ['source', 'target', 'total_users', 'converted_users', 'avg_session_length']
    flow_summary['conversion_rate'] = flow_summary['converted_users'] / flow_summary['total_users']
    
    # Filter significant flows (min 10 users)
    significant_flows = flow_summary[flow_summary['total_users'] >= 10].copy()
    
    print(f"📊 Found {len(significant_flows)} significant user flows")
    
    # Create node mapping
    all_nodes = list(set(significant_flows['source'].tolist() + significant_flows['target'].tolist()))
    node_mapping = {node: i for i, node in enumerate(all_nodes)}
    
    # node colors
    node_colors = []
    for node in all_nodes:
        if "ENTRY:" in node:
            node_colors.append("lightblue")
        elif "MAIN:" in node:
            node_colors.append("orange") 
        elif "FTD CONVERTED" in node:
            node_colors.append("green")
        elif "NO CONVERSION" in node:
            node_colors.append("red")
        else:
            node_colors.append("lightgray")
    
    # link colors based on conversion performance
    link_colors = []
    for _, flow in significant_flows.iterrows():
        if "FTD CONVERTED" in flow['target']:
            link_colors.append("rgba(0,255,0,0.6)")  # Green for conversions
        elif "NO CONVERSION" in flow['target']:
            link_colors.append("rgba(255,0,0,0.3)")  # Red for non-conversions
        elif flow['conversion_rate'] > 0.05:  # High converting entry/main flows
            link_colors.append("rgba(0,200,0,0.4)")
        elif flow['conversion_rate'] > 0.02:  # Medium converting flows
            link_colors.append("rgba(255,165,0,0.4)")  # Orange
        else:
            link_colors.append("rgba(100,100,100,0.3)")  # Gray for low converting
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="black", width=2),
            label=[node.replace('ENTRY: ', '🚪 ').replace('MAIN: ', '🎯 ') for node in all_nodes],
            color=node_colors
        ),
        link=dict(
            source=[node_mapping[source] for source in significant_flows['source']],
            target=[node_mapping[target] for target in significant_flows['target']],
            value=significant_flows['total_users'].tolist(),
            color=link_colors,
            # Add hover info
            customdata=significant_flows[['conversion_rate', 'avg_session_length']].values,
            hovertemplate='<b>%{source.label}</b> → <b>%{target.label}</b><br>' +
                         'Users: %{value}<br>' +
                         'Conversion Rate: %{customdata[0]:.1%}<br>' +
                         'Avg Session Length: %{customdata[1]:.1f} pages<extra></extra>'
        )
    )])
    
    fig.update_layout(
        title_text="Simplified User Journey Flow<br>🚪 Entry Point → 🎯 Main Category → Outcome",
        font_size=12,
        height=600,
        width=1200
    )
    
    # Display with error handling
    try:
        fig.show()
    except Exception as e:
        print(f"Plotly display error: {e}")
        print("Saving as HTML file instead...")
        fig.write_html("sankey_diagram.html")
        print("✅ Sankey diagram saved as _sankey_diagram.html'")
    
    # Print insights
    print("\n🔍 KEY FLOW INSIGHTS:")
    print("-" * 40)
    
    # Best entry points
    entry_flows = significant_flows[significant_flows['source'].str.contains('ENTRY:')]
    if len(entry_flows) > 0:
        best_entries = entry_flows.groupby('source').agg({
            'total_users': 'sum',
            'converted_users': 'sum'
        }).reset_index()
        best_entries['conversion_rate'] = best_entries['converted_users'] / best_entries['total_users']
        best_entries = best_entries.sort_values('converted_users', ascending=False).head(5)
        
        print("🏆 Top 5 Entry Points by Conversions:")
        for _, row in best_entries.iterrows():
            entry_name = row['source'].replace('ENTRY: ', '')
            print(f"  {entry_name}: {int(row['converted_users'])} FTDs ({row['conversion_rate']:.1%} rate)")
    
    # Best content paths
    main_flows = significant_flows[significant_flows['source'].str.contains('MAIN:')]
    if len(main_flows) > 0:
        best_main = main_flows.groupby('source').agg({
            'total_users': 'sum', 
            'converted_users': 'sum'
        }).reset_index()
        best_main['conversion_rate'] = best_main['converted_users'] / best_main['total_users']
        best_main = best_main.sort_values('conversion_rate', ascending=False).head(5)
        
        print("\n🎯 Best Converting Content Categories:")
        for _, row in best_main.iterrows():
            category_name = row['source'].replace('MAIN: ', '')
            print(f"  {category_name}: {row['conversion_rate']:.1%} conversion rate ({int(row['total_users'])} users)")
    
    return significant_flows

# Create alternative flow visualization  
def create_conversion_path_analysis(df_ga4, df_postback):
    """
    Alternative: Show the most common successful conversion paths
    """
    
    print("\n🛣️ CONVERSION PATH ANALYSIS")
    print("-" * 40)
    
    # Get user journeys with conversion status
    user_journeys = df_ga4.sort_values(['user_pseudo_id', 'event_timestamp']) \
                          .groupby('user_pseudo_id').agg({
        'section_category': list
    }).reset_index()
    
    # Get conversions
    conversions = df_postback.groupby('meta_user_id').agg(
        has_ftd=('et', lambda x: 'ftd' in x.values)
    ).reset_index()
    
    user_journeys = user_journeys.merge(
        conversions, left_on='user_pseudo_id', right_on='meta_user_id', how='left'
    ).fillna(False)
    
    # Focus on converted users only
    converted_journeys = user_journeys[user_journeys['has_ftd'] == True]
    
    if len(converted_journeys) > 0:
        # Find most common conversion paths (first 3 steps)
        conversion_paths = []
        for _, user in converted_journeys.iterrows():
            sequence = user['section_category']
            if isinstance(sequence, list) and len(sequence) >= 1:
                # Take first 3 steps or all if less than 3
                path = ' → '.join(sequence[:3])
                conversion_paths.append(path)
        
        path_counts = pd.Series(conversion_paths).value_counts().head(10)
        
        print("🏆 Top 10 Successful Conversion Paths:")
        for i, (path, count) in enumerate(path_counts.items(), 1):
            print(f"{i:2d}. {path} ({count} conversions)")
        
        # Single-category converters
        single_category = converted_journeys[
            converted_journeys['section_category'].apply(lambda x: len(set(x)) == 1 if isinstance(x, list) else False)
        ]
        
        if len(single_category) > 0:
            single_cats = []
            for _, user in single_category.iterrows():
                single_cats.append(user['section_category'][0])
            
            single_cat_counts = pd.Series(single_cats).value_counts().head(5)
            print(f"\n🎯 Single-Category Converters ({len(single_category)} total):")
            for cat, count in single_cat_counts.items():
                print(f"  {cat}: {count} conversions")
    
    return converted_journeys

# Updated main function
def visualize_journey_flows(df_ga4, df_postback):
    """
    Run flow visualizations that actually show useful insights
    """
    
    print("🌊 CREATING USER JOURNEY FLOW VISUALIZATIONS")
    print("="*60)
    
    try:
        # 1. Sankey Flow Diagram
        print("\n1. SIMPLIFIED SANKEY FLOW DIAGRAM")
        print("-" * 40)
        sankey_data = create_sankey_flow_diagram(df_ga4, df_postback)
        
        # 2. Conversion Path Analysis
        print("\n2. CONVERSION PATH ANALYSIS")
        print("-" * 40)
        conversion_paths = create_conversion_path_analysis(df_ga4, df_postback)
        
        return sankey_data, conversion_paths
        
    except Exception as e:
        print(f"❌ Error in visualization pipeline: {e}")
        return None, None